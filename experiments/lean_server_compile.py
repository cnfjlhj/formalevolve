                      
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter

sys.dont_write_bytecode = True


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


_G_SUFFIX_PAT = re.compile(r"_g(?P<g>\d+)$")
_FINE_EVAL_BLOCK_COMMENT_RE = re.compile(r"/-[\s\S]*?-/")
_FINE_EVAL_LINE_COMMENT_RE = re.compile(r"(?m)^\s*--.*?$")
_FINE_EVAL_SORRY_TOKEN_RE = re.compile(r"\bsorry\b")
_FINE_EVAL_THEOREM_SPLIT_RE = re.compile(r"(?m)^\s*(?:noncomputable\s+)?theorem\b")
_LEADING_NUM_PREFIX_RE = re.compile(r"^\d+_")


def _combibench_theorem_name(problem_id: str) -> str:
    """
    Map our local CombiBench problem_id (e.g. '0000_hackmath_1') to HF theorem_name ('hackmath_1').
    """
    return _LEADING_NUM_PREFIX_RE.sub("", str(problem_id or "").strip())


@lru_cache(maxsize=8)
def _load_hf_answer_map(
    *,
    dataset_name: str,
    split: str,
    index_column: str,
    answer_column: str,
) -> Dict[str, Optional[List[str]]]:
    try:
        from datasets import load_dataset                
    except Exception as e:
        raise RuntimeError(f"datasets not available (needed for HF answer checks): {type(e).__name__}: {e}") from e

    ds = load_dataset(str(dataset_name), split=str(split))
    out: Dict[str, Optional[List[str]]] = {}
    for ex in ds:
        key = str(ex.get(index_column) or "").strip()
        if not key:
            continue
        raw = ex.get(answer_column)
        if raw is None:
            out[key] = None
            continue
        if isinstance(raw, list):
            vals = [str(x) for x in raw if x is not None and str(x).strip()]
        else:
            vals = [str(raw).strip()] if str(raw).strip() else []
        out[key] = vals if vals else None
    return out


_ABBREV_SOLUTION_NAME_RE = re.compile(r"(?m)^\s*abbrev\s+(?P<name>[A-Za-z0-9_]+_solution)\b")


def _fine_eval_extract_solution_tags(formal_statement: str) -> List[str]:
    """
    Extract `*_solution` abbrev names from the formal-statement template, mirroring CombiBench's
    fill-in-the-blank convention.
    """
    tmpl = _remove_comments_lean(formal_statement).strip()
    if not tmpl:
        return []
    tags: List[str] = []
    for chunk in tmpl.split("\n\n"):
        s = chunk.strip()
        if not s:
            continue
        if s.startswith(("import", "set_option", "open")):
            continue
        m = _ABBREV_SOLUTION_NAME_RE.search(s)
        if m:
            tags.append(str(m.group("name")))
    return tags


def _strip_g_suffix(name: str) -> str:
    return _G_SUFFIX_PAT.sub("", name or "")


def _parse_attempt_id(attempt_id: str) -> Tuple[str, Optional[str], Optional[int], Optional[int]]:
    base = _strip_g_suffix(attempt_id)
    parts = base.split("__")
    group_id = parts[0] if parts and parts[0] else base
    baseline = parts[1] if len(parts) >= 2 and parts[1] else None

    k_idx: Optional[int] = None
    for p in parts[2:]:
        if p.startswith("k") and p[1:].isdigit():
            k_idx = int(p[1:])
            break

    g_idx: Optional[int] = None
    m = _G_SUFFIX_PAT.search(attempt_id or "")
    if m:
        try:
            g_idx = int(m.group("g"))
        except Exception:
            g_idx = None
    return group_id, baseline, k_idx, g_idx


def _normalize_lean_server_url(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    if not (s.startswith("http://") or s.startswith("https://")):
        return ""
    return s.rstrip("/")


def _remove_comments_lean(code: str) -> str:
    """
    Best-effort comment stripper for Lean4 code.

    CombiBench's Fine-Eval removes comments before checking constraints, because models may
    try to "cheat" by commenting out code.
    """
    text = str(code or "")
    text = _FINE_EVAL_BLOCK_COMMENT_RE.sub("", text)
    text = _FINE_EVAL_LINE_COMMENT_RE.sub("", text)
    return text


def _fine_eval_precheck(*, full_code: str, formal_statement: str, forbid_keywords: List[str]) -> Tuple[bool, str]:
    """
    Lightweight Fine-Eval checks before calling the Lean server.

    - Forbid specific keywords (e.g. axiom/local_instance).
    - Ensure the generated code still contains the input formal statement skeleton
      (with all `sorry` removed), excluding imports/options/opens.
    """
    out = _remove_comments_lean(full_code).strip()
    if not out:
        return False, "FINE_EVAL: empty output code after comment stripping"

    for kw in forbid_keywords:
        if kw and str(kw) in out:
            return False, f"FINE_EVAL: forbidden keyword found: {kw!r}"

    tmpl = _remove_comments_lean(formal_statement).strip()
    if not tmpl:
                                                                       
        return True, ""

                                                                                
                                                                  
    parts: List[str] = []
    for chunk in tmpl.split("\n\n"):
        if not chunk.strip():
            continue
                                                                                      
        chunk = chunk.replace("\nnoncomputable theorem", "\n\nnoncomputable theorem")
        chunk = chunk.replace("\ntheorem", "\n\ntheorem")
        parts.append(chunk)

    filtered: List[str] = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        if s.startswith(("import", "set_option", "open")):
            continue
        s2 = _FINE_EVAL_SORRY_TOKEN_RE.sub("", s).strip()
        if s2:
            filtered.append(s2)

    missing = [s for s in filtered if s not in out]
    if missing:
                                                                       
        return False, f"FINE_EVAL: missing {len(missing)} formal-statement chunks in output"
    return True, ""


def _append_solution_answer_checks(
    *,
    full_code: str,
    formal_statement: str,
    problem_id: str,
    answer_map: Dict[str, Optional[List[str]]],
) -> Tuple[bool, str, str]:
    """
    Append CombiBench-style answer equality checks for fill-in-the-blank problems.

    Mirrors CombiBench's `one_stage_verify` behavior:
      - Only trigger when the formal statement contains `abbrev *_solution ...` tags.
      - If both tags and ground-truth answers exist, add `example: tag = gt := by try rfl; try norm_num`.
    """
    tags = _fine_eval_extract_solution_tags(formal_statement)
    if not tags:
                                                                                                          
        return True, "", full_code

    theorem_name = _combibench_theorem_name(problem_id)
    ground_truths = answer_map.get(theorem_name)
    if not ground_truths:
                                                                                              
        return True, "", full_code

    if len(tags) != len(ground_truths):
        return (
            False,
            f"FINE_EVAL: answer/tag arity mismatch tags={len(tags)} ground_truths={len(ground_truths)}",
            full_code,
        )

    out = str(full_code)
    for tag, gt in zip(tags, ground_truths):
        gt_s = str(gt or "").strip()
        if not gt_s:
            return False, "FINE_EVAL: empty ground_truth answer string", full_code
                                                                                         
        out += f"\n\nexample: {tag} = {gt_s} := by\n  try rfl\n  try norm_num"
    return True, "", out


def _msg_to_dict(m: Any) -> Dict[str, Any]:
    if not isinstance(m, dict):
        return {
            "severity": None,
            "pos": {"line": None, "column": None},
            "endPos": {"line": None, "column": None},
            "data": str(m),
        }

    start_pos = m.get("pos") or m.get("start_pos") or {}
    end_pos = m.get("endPos") or m.get("end_pos") or {}
    if not isinstance(start_pos, dict):
        start_pos = {}
    if not isinstance(end_pos, dict):
        end_pos = {}
    return {
        "severity": m.get("severity"),
        "pos": {"line": start_pos.get("line"), "column": start_pos.get("column")},
        "endPos": {"line": end_pos.get("line"), "column": end_pos.get("column")},
        "data": m.get("data"),
    }


def _partition_messages(messages: List[Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    infos: List[Dict[str, Any]] = []
    for m in messages:
        d = _msg_to_dict(m)
        sev = str(d.get("severity") or "").lower()
        if sev == "error":
            errors.append(d)
        elif sev == "warning":
            warnings.append(d)
        else:
            infos.append(d)
    return errors, warnings, infos


def _compile_via_http(
    session: requests.Session,
    *,
    lean_server_url: str,
    code: str,
    timeout_s: int,
) -> Dict[str, Any]:
    snippet_id = f"proof-compile-{uuid.uuid4().hex}"
    resp: Optional[requests.Response] = None
    try:
        resp = session.post(
            lean_server_url,
            json={"snippets": [{"id": snippet_id, "code": code}], "timeout": int(timeout_s)},
            timeout=float(timeout_s + 30),
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.exceptions.Timeout:
        return {
            "pass": False,
            "complete": False,
            "system_errors": f"HTTP TIMEOUT after {timeout_s}s",
        }
    except requests.exceptions.ConnectionError as e:
        return {
            "pass": False,
            "complete": False,
            "system_errors": f"HTTP CONNECTION ERROR: {type(e).__name__}: {e}",
        }
    except requests.exceptions.RequestException as e:
        return {
            "pass": False,
            "complete": False,
            "system_errors": f"HTTP REQUEST ERROR: {type(e).__name__}: {e}",
        }
    except Exception as e:
        return {
            "pass": False,
            "complete": False,
            "system_errors": f"HTTP EXCEPTION: {type(e).__name__}: {e}",
        }
    finally:
                                                                               
                                                                            
                                         
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass

    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not results:
        return {
            "pass": False,
            "complete": False,
            "system_errors": "Invalid Lean server response: missing/empty results",
        }

    check_result = results[0] if isinstance(results[0], dict) else {}
    response = check_result.get("response") if isinstance(check_result, dict) else None
    if not isinstance(response, dict):
        response = {}
    messages = response.get("messages") or []
    if not isinstance(messages, list):
        messages = []

    errors, warnings, infos = _partition_messages(messages)
    sorries = [w for w in warnings if "declaration uses 'sorry'" in str(w.get("data") or "")]

    pass_ok = bool(len(errors) == 0)
                                                                                                         
    complete_ok = bool(pass_ok and not sorries)

    return {
        "sorries": sorries,
        "tactics": [],
        "errors": errors,
        "warnings": warnings,
        "infos": infos,
        "ast": {},
        "system_errors": None,
        "pass": pass_ok,
        "complete": complete_ok,
    }


def _result_record(name: str, code: str, comp: Dict[str, Any]) -> Dict[str, Any]:
    return {"name": name, "problem_id": name, "code": code, "compilation_result": comp}


@dataclass(frozen=True)
class GroupKey:
    group_id: str
    baseline: str


def _group_key_for_attempt(name: str) -> GroupKey:
    group_id, baseline, _k, _g = _parse_attempt_id(name)
    return GroupKey(group_id=str(group_id), baseline=str(baseline or ""))


def _sorted_attempts(attempts: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _key(r: Dict[str, Any]) -> Tuple[int, int, str]:
        name = str(r.get("name") or r.get("problem_id") or "").strip()
        _gid, _base, k_idx, g_idx = _parse_attempt_id(name)
        return (k_idx if k_idx is not None else 1_000_000, g_idx if g_idx is not None else 1_000_000, name)

    return sorted(list(attempts), key=_key)


def _load_existing_results(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for r in raw:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name") or r.get("problem_id") or "").strip()
        if not name:
            continue
        out[name] = r
    return out


def _group_solved(
    *,
    existing_by_name: Dict[str, Dict[str, Any]],
    group_attempts: List[Dict[str, Any]],
    early_stop_field: str,
) -> bool:
    if str(early_stop_field) == "none":
        return False
    for r in group_attempts:
        name = str(r.get("name") or r.get("problem_id") or "").strip()
        if not name:
            continue
        prev = existing_by_name.get(name)
        if not isinstance(prev, dict):
            continue
        comp = prev.get("compilation_result") or {}
        if isinstance(comp, dict) and bool(comp.get(early_stop_field, False)):
            return True
    return False


def _compile_group(
    *,
    lean_server_url: str,
    timeout_s: int,
    early_stop_field: str,
    group_attempts: List[Dict[str, Any]],
    existing_by_name: Dict[str, Dict[str, Any]],
    fine_eval: bool,
    fine_eval_forbid_keywords: List[str],
    fine_eval_answer_map: Optional[Dict[str, Optional[List[str]]]],
) -> List[Dict[str, Any]]:
    session = requests.Session()
                                                                             
                                                                             
                                                                    
                                                                            
                                                                       
    adapter = HTTPAdapter(pool_connections=2, pool_maxsize=2, max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    out: List[Dict[str, Any]] = []
    early_stop_enabled = str(early_stop_field) != "none"

    for r in group_attempts:
        name = str(r.get("name") or r.get("problem_id") or "").strip()
        if not name:
            continue

        prev = existing_by_name.get(name)
        if isinstance(prev, dict):
            comp_prev = prev.get("compilation_result") or {}
            if early_stop_enabled and isinstance(comp_prev, dict) and bool(comp_prev.get(early_stop_field, False)):
                break
                              
                                                                                                 
                                                                                                                 
                                                                                                         
                                                                                                       
            prev_code = prev.get("full_code")
            if prev_code is None:
                prev_code = prev.get("code")
            if prev_code is None:
                prev_code = ""
            prev_code_str = str(prev_code)
            cur_code = r.get("full_code")
            if cur_code is None:
                cur_code = ""
            cur_code_str = str(cur_code)
                                                                                                           
                                                                       
            sys_err = ""
            if isinstance(comp_prev, dict):
                sys_err = str(comp_prev.get("system_errors") or "").strip()
            if prev_code_str == cur_code_str and not sys_err:
                continue

        full_code = r.get("full_code")
        if full_code is None:
            full_code = ""
        code_str = str(full_code)
        if code_str.strip() in ("", "None"):
            comp = {
                "sorries": [],
                "tactics": [],
                "errors": [
                    {
                        "severity": "error",
                        "pos": {"line": None, "column": None},
                        "endPos": {"line": None, "column": None},
                        "data": "Empty/None code",
                    }
                ],
                "warnings": [],
                "infos": [],
                "ast": {},
                "system_errors": None,
                "pass": False,
                "complete": False,
            }
        else:
            if fine_eval:
                tmpl = str(r.get("lean4_code") or "")
                ok, msg = _fine_eval_precheck(
                    full_code=code_str,
                    formal_statement=tmpl,
                    forbid_keywords=fine_eval_forbid_keywords,
                )
                if not ok:
                    comp = {
                        "sorries": [],
                        "tactics": [],
                        "errors": [
                            {
                                "severity": "error",
                                "pos": {"line": None, "column": None},
                                "endPos": {"line": None, "column": None},
                                "data": msg or "FINE_EVAL: failed",
                            }
                        ],
                        "warnings": [],
                        "infos": [],
                        "ast": {},
                        "system_errors": None,
                        "pass": False,
                        "complete": False,
                    }
                    out.append(_result_record(name, code_str, comp))
                    continue
                if fine_eval_answer_map is not None:
                    gid, _base, _k, _g = _parse_attempt_id(name)
                    ok2, msg2, code_str2 = _append_solution_answer_checks(
                        full_code=code_str,
                        formal_statement=tmpl,
                        problem_id=str(gid),
                        answer_map=fine_eval_answer_map,
                    )
                    if not ok2:
                        comp = {
                            "sorries": [],
                            "tactics": [],
                            "errors": [
                                {
                                    "severity": "error",
                                    "pos": {"line": None, "column": None},
                                    "endPos": {"line": None, "column": None},
                                    "data": msg2 or "FINE_EVAL: answer check failed",
                                }
                            ],
                            "warnings": [],
                            "infos": [],
                            "ast": {},
                            "system_errors": None,
                            "pass": False,
                            "complete": False,
                        }
                        out.append(_result_record(name, code_str, comp))
                        continue
                    code_str = code_str2
            comp = _compile_via_http(session, lean_server_url=lean_server_url, code=code_str, timeout_s=timeout_s)
        out.append(_result_record(name, code_str, comp))

        if early_stop_enabled and isinstance(comp, dict) and bool(comp.get(early_stop_field, False)):
            break

    try:
        session.close()
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Compile Goedel-Prover-V2 inference outputs using an HTTP Lean server (ONLY).\n"
            "\n"
            "Features:\n"
            "- Per-problem early-stop: stop compiling remaining attempts once a problem is solved\n"
            "  (success criterion selectable via --early_stop_field).\n"
            "- Per-problem parallelism: compile different problems concurrently via --workers.\n"
        )
    )
    ap.add_argument("--input_path", required=True, help="Path to to_inference_codes.json (list of dicts with full_code).")
    ap.add_argument("--output_path", required=True, help="Path to write code_compilation_repl.json")
    ap.add_argument(
        "--lean_server_url",
        default=os.environ.get("LEAN_SERVER_URL", "").strip(),
        help="HTTP Lean server URL (e.g. http://127.0.0.1:8002/api/check). Required.",
    )
    ap.add_argument("--timeout_s", type=int, default=120, help="Per-proof timeout seconds (default: 120).")
    ap.add_argument("--workers", type=int, default=8, help="Parallel workers over problems (default: 8).")
    ap.add_argument(
        "--early_stop_field",
        default="complete",
        choices=["complete", "pass", "none"],
        help="Per-problem early-stop success criterion (default: complete). Use 'none' to compile all attempts.",
    )
    ap.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Write intermediate output every N completed problems (default: 10).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output_path by skipping already-compiled names.",
    )
    ap.add_argument(
        "--fine_eval",
        action="store_true",
        help=(
            "Enable CombiBench-style Fine-Eval checks before Lean compilation: "
            "forbid `axiom`/`local_instance` and require the output to contain the input statement skeleton "
            "(with `sorry` removed)."
        ),
    )
    ap.add_argument(
        "--fine_eval_answer_hf_dataset",
        type=str,
        default="",
        help=(
            "Optional HF dataset id to enforce CombiBench-style answer checks for fill-in-the-blank tasks "
            "(e.g. AI-MO/CombiBench). Requires --fine_eval."
        ),
    )
    ap.add_argument(
        "--fine_eval_answer_hf_split",
        type=str,
        default="",
        help=(
            "HF split name to load answers from (e.g. test). Requires --fine_eval and --fine_eval_answer_hf_dataset."
        ),
    )
    ap.add_argument(
        "--fine_eval_answer_index_column",
        type=str,
        default="theorem_name",
        help="HF index column for answer map (default: theorem_name).",
    )
    ap.add_argument(
        "--fine_eval_answer_column",
        type=str,
        default="answer",
        help="HF answer column (default: answer).",
    )
    ap.add_argument(
        "--fine_eval_forbid",
        type=str,
        default="axiom,local_instance",
        help="Comma-separated forbidden keyword substrings when --fine_eval is enabled (default: axiom,local_instance).",
    )
    ap.add_argument(
        "--write_stdout_log",
        action="store_true",
        help="Also write progress lines to stdout (useful when wrapped by other scripts).",
    )
    args = ap.parse_args()

    input_path = Path(str(args.input_path)).expanduser().resolve()
    if not input_path.exists():
        _warn(f"input not found: {input_path}")
        return 2
    output_path = Path(str(args.output_path)).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lean_server_url = _normalize_lean_server_url(str(args.lean_server_url))
    if not lean_server_url:
        _warn(
            "Invalid --lean_server_url. This script ONLY supports HTTP Lean server.\n"
            "Set env LEAN_SERVER_URL or pass --lean_server_url http(s)://.../api/check"
        )
        return 2

    try:
        raw = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as e:
        _warn(f"failed to parse input: {type(e).__name__}: {e}")
        return 2
    if not isinstance(raw, list):
        _warn(f"expected a JSON list: {input_path}")
        return 2

    existing_by_name: Dict[str, Dict[str, Any]] = _load_existing_results(output_path) if bool(args.resume) else {}
    if existing_by_name:
        _warn(f"resume: loaded {len(existing_by_name)} existing records from {output_path}")

    grouped: Dict[GroupKey, List[Dict[str, Any]]] = {}
    for r in raw:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name") or r.get("problem_id") or "").strip()
        if not name:
            continue
        gk = _group_key_for_attempt(name)
        grouped.setdefault(gk, []).append(r)

    groups: List[Tuple[GroupKey, List[Dict[str, Any]]]] = []
    for gk, attempts in grouped.items():
        groups.append((gk, _sorted_attempts(attempts)))
    groups.sort(key=lambda x: (x[0].baseline, x[0].group_id))

    early_stop_field = str(args.early_stop_field)
    fine_eval = bool(args.fine_eval)
    fine_eval_forbid_keywords = [s.strip() for s in str(args.fine_eval_forbid or "").split(",") if s.strip()]
    fine_eval_answer_map: Optional[Dict[str, Optional[List[str]]]] = None
    if fine_eval and str(args.fine_eval_answer_hf_dataset or "").strip() and str(args.fine_eval_answer_hf_split or "").strip():
        fine_eval_answer_map = _load_hf_answer_map(
            dataset_name=str(args.fine_eval_answer_hf_dataset),
            split=str(args.fine_eval_answer_hf_split),
            index_column=str(args.fine_eval_answer_index_column),
            answer_column=str(args.fine_eval_answer_column),
        )

    def _write_checkpoint() -> None:
        ordered: List[Dict[str, Any]] = []
        for r in raw:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name") or r.get("problem_id") or "").strip()
            if not name:
                continue
            rec = existing_by_name.get(name)
            if rec is not None:
                ordered.append(rec)
        _atomic_write_json(output_path, ordered)

                                                                           
    input_code_by_name: Dict[str, str] = {}
    for r in raw:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name") or r.get("problem_id") or "").strip()
        if not name:
            continue
        code = r.get("full_code")
        if code is None:
            code = ""
        input_code_by_name[name] = str(code)

    total_groups = len(groups)
    groups_to_run: List[Tuple[GroupKey, List[Dict[str, Any]]]] = []
    num_skipped_solved = 0
    num_skipped_all_done = 0
    for gk, attempts in groups:
        if _group_solved(existing_by_name=existing_by_name, group_attempts=attempts, early_stop_field=early_stop_field):
            num_skipped_solved += 1
            continue
        names = [str(r.get("name") or r.get("problem_id") or "").strip() for r in attempts]
        names = [n for n in names if n]
        if names and all(n in existing_by_name for n in names):
                                                                                              
                                                                                       
            any_changed = False
            any_retryable = False
            for n in names:
                prev = existing_by_name.get(n)
                if not isinstance(prev, dict):
                    any_changed = True
                    break
                                                                                                  
                prev_code = prev.get("full_code")
                if prev_code is None:
                    prev_code = prev.get("code")
                if prev_code is None:
                    prev_code = ""
                if str(prev_code) != input_code_by_name.get(n, ""):
                    any_changed = True
                    break
                comp_prev = prev.get("compilation_result") or {}
                if isinstance(comp_prev, dict) and str(comp_prev.get("system_errors") or "").strip():
                    any_retryable = True
                    break
            if not any_changed and not any_retryable:
                num_skipped_all_done += 1
                continue
        groups_to_run.append((gk, attempts))

    _warn(
        f"start: groups_total={total_groups} groups_to_run={len(groups_to_run)} "
        f"resume={bool(args.resume)} skipped_solved={num_skipped_solved} skipped_all_done={num_skipped_all_done} "
        f"workers={int(args.workers)} early_stop_field={early_stop_field} timeout_s={int(args.timeout_s)}"
    )
    if bool(args.write_stdout_log):
        print(
            f"[INFO] start groups_total={total_groups} groups_to_run={len(groups_to_run)} "
            f"workers={int(args.workers)} early_stop_field={early_stop_field} timeout_s={int(args.timeout_s)}",
            flush=True,
        )

    t0 = time.time()
    done_groups = 0
    new_records = 0

    def _count_success(field: str) -> int:
        ok = 0
        for rec in existing_by_name.values():
            comp = rec.get("compilation_result") or {}
            if isinstance(comp, dict) and bool(comp.get(field, False)):
                ok += 1
        return ok

    pass_n = _count_success("pass")
    complete_n = _count_success("complete")

    save_every = max(1, int(args.save_every))
    workers = max(1, int(args.workers))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for gk, attempts in groups_to_run:
            fut = ex.submit(
                _compile_group,
                lean_server_url=lean_server_url,
                timeout_s=int(args.timeout_s),
                early_stop_field=early_stop_field,
                group_attempts=attempts,
                existing_by_name=existing_by_name,
                fine_eval=fine_eval,
                fine_eval_forbid_keywords=fine_eval_forbid_keywords,
                fine_eval_answer_map=fine_eval_answer_map,
            )
            futs[fut] = gk

        for fut in as_completed(futs):
            gk = futs[fut]
            try:
                rows = fut.result()
            except Exception as e:
                _warn(f"group failed: {gk.baseline}/{gk.group_id}: {type(e).__name__}: {e}")
                rows = []

            for r in rows:
                name = str(r.get("name") or r.get("problem_id") or "").strip()
                if not name:
                    continue
                old = existing_by_name.get(name)
                old_comp = (old or {}).get("compilation_result") if isinstance(old, dict) else {}
                old_pass = bool(old_comp.get("pass", False)) if isinstance(old_comp, dict) else False
                old_complete = bool(old_comp.get("complete", False)) if isinstance(old_comp, dict) else False
                existing_by_name[name] = r
                if old is None:
                    new_records += 1

                comp = r.get("compilation_result") or {}
                new_pass = bool(comp.get("pass", False)) if isinstance(comp, dict) else False
                new_complete = bool(comp.get("complete", False)) if isinstance(comp, dict) else False
                pass_n += int(new_pass) - int(old_pass)
                complete_n += int(new_complete) - int(old_complete)

            done_groups += 1
            if done_groups % 5 == 0 or done_groups == len(groups_to_run):
                _warn(
                    f"progress: groups={done_groups}/{len(groups_to_run)} new_records={new_records} "
                    f"pass={pass_n} complete={complete_n} elapsed_s={time.time()-t0:.1f}"
                )
                if bool(args.write_stdout_log):
                    print(
                        f"[INFO] progress groups={done_groups}/{len(groups_to_run)} new_records={new_records} "
                        f"pass={pass_n} complete={complete_n} elapsed_s={time.time()-t0:.1f}",
                        flush=True,
                    )
            if done_groups % save_every == 0 or done_groups == len(groups_to_run):
                _write_checkpoint()
                _warn(f"checkpoint: wrote {len(existing_by_name)} total records -> {output_path}")
                if bool(args.write_stdout_log):
                    print(f"[INFO] checkpoint wrote={len(existing_by_name)} path={output_path}", flush=True)

    _write_checkpoint()
    _warn(f"wrote: {output_path}")
    _warn(
        f"summary: groups_total={total_groups} groups_ran={len(groups_to_run)} new_records={new_records} "
        f"pass={pass_n} complete={complete_n} elapsed_s={time.time()-t0:.1f}"
    )
    if bool(args.write_stdout_log):
        print(
            f"[INFO] summary groups_total={total_groups} groups_ran={len(groups_to_run)} new_records={new_records} "
            f"pass={pass_n} complete={complete_n} elapsed_s={time.time()-t0:.1f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
