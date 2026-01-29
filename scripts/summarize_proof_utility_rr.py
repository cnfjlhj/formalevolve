                      
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.dont_write_bytecode = True


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


@dataclass(frozen=True)
class Row:
    method: str
    tag: str
    field: str
    budget: int
    problems: int
    solved: int
    solved_ratio: float
    solved_aligned_complete: int
    solved_aligned_complete_ratio: float
    solved_theorem_complete: int
    solved_theorem_complete_ratio: float
    problems_alloc_gt0: int
    solved_alloc_gt0: int
    solved_ratio_alloc_gt0: float
    solved_aligned_complete_alloc_gt0: int
    solved_aligned_complete_ratio_alloc_gt0: float
    solved_theorem_complete_ratio_alloc_gt0: float


def _infer_budget_from_index(index_json: Path) -> int:
    try:
        obj = _read_json(index_json)
    except Exception:
        return 0
    if not isinstance(obj, dict):
        return 0
    return _safe_int(obj.get("budget"), 0)


def _load_index_problem_sets(index_json: Path) -> Tuple[Set[str], Set[str]]:
    """
    Returns (all_problem_ids, active_problem_ids) from an RR index.

    - all_problem_ids: any entry with a non-empty `problem_id`
    - active_problem_ids: subset where `allocations` is a non-empty list
      (i.e., the method produced candidates for that problem)
    """
    if not index_json.exists():
        return set(), set()
    try:
        obj = _read_json(index_json)
    except Exception:
        return set(), set()
    if not isinstance(obj, dict):
        return set(), set()
    probs = obj.get("problems") or []
    if not isinstance(probs, list):
        return set(), set()
    all_ids: Set[str] = set()
    active_ids: Set[str] = set()
    for pr in probs:
        if not isinstance(pr, dict):
            continue
        pid = str(pr.get("problem_id") or "").strip()
        if not pid:
            continue
        all_ids.add(pid)
        alloc = pr.get("allocations") or []
        if isinstance(alloc, list) and len(alloc) > 0:
            active_ids.add(pid)
    return all_ids, active_ids


_FINE_EVAL_BLOCK_COMMENT_RE = re.compile(r"/-[\s\S]*?-/")
_FINE_EVAL_LINE_COMMENT_RE = re.compile(r"(?m)^\s*--.*?$")
_FINE_EVAL_SORRY_TOKEN_RE = re.compile(r"\bsorry\b")


def _remove_comments_lean(code: str) -> str:
    text = str(code or "")
    if "/-" not in text and "--" not in text:
        return text
    text = _FINE_EVAL_BLOCK_COMMENT_RE.sub("", text)
    text = _FINE_EVAL_LINE_COMMENT_RE.sub("", text)
    return text


def _remove_comments_lean_escaped(content_escaped: str) -> str:
    """
    Best-effort comment stripper for a JSON-escaped Lean string content (i.e., without the outer quotes).

    - Line comments: remove `-- ...` until the next literal `\\n` escape.
    - Block comments: remove `/- ... -/`.

    This avoids `json.loads` on very large code strings.
    """
    s = str(content_escaped or "")
    if not s:
        return ""

                
    if "--" not in s and "/-" not in s:
        return s

    out: List[str] = []
    i = 0
    n = len(s)
    while i < n:
                        
        if i + 1 < n and s[i] == "/" and s[i + 1] == "-":
            j = s.find("-/", i + 2)
            if j == -1:
                                                        
                break
            i = j + 2
            continue

                       
        if i + 1 < n and s[i] == "-" and s[i + 1] == "-":
            j = s.find(r"\n", i + 2)
            if j == -1:
                                         
                break
                                      
            out.append(r"\n")
            i = j + 2
            continue

        out.append(s[i])
        i += 1

    return "".join(out)


def _fine_eval_precheck_like(*, full_code: str, formal_statement: str, forbid_keywords: List[str]) -> bool:
    """
    A lightweight, offline version of `experiments/lean_server_compile.py --fine_eval`:

    - Strip Lean comments from output and template.
    - Forbid keyword substrings (default: axiom/local_instance).
    - Require the output to contain the input statement skeleton (with all `sorry` removed),
      excluding import/open/set_option chunks.
    """
    out = _remove_comments_lean(full_code).strip()
    if not out:
        return False
    for kw in forbid_keywords:
        if kw and str(kw) in out:
            return False

    chunks = _fine_eval_template_chunks(formal_statement)
    return all(s in out for s in chunks)


def _fine_eval_template_chunks(formal_statement: str) -> List[str]:
    """
    Precompute the list of skeleton chunks required for Fine-Eval-style containment.
    """
    tmpl = _remove_comments_lean(formal_statement).strip()
    if not tmpl:
        return []

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
    return filtered


def _fine_eval_template_chunks_escaped(formal_statement: str) -> List[str]:
    """
    Precompute required skeleton chunks in JSON-escaped form (without outer quotes).

    This matches how `code_compilation_repl.json` stores `code` (as a JSON string).
    """
    return [json.dumps(s, ensure_ascii=False)[1:-1] for s in _fine_eval_template_chunks(formal_statement)]


def _load_rr64_statement_by_problem_id(rr64_dataset_jsonl: Path) -> Dict[str, str]:
    """
    Map rr64 statement-level problem_id -> statement skeleton.

    Note: compilation outputs use attempt names with a `_g{k}` suffix, so callers should strip it
    before looking up in this mapping.
    """
    if not rr64_dataset_jsonl.exists():
        return {}
    out: Dict[str, str] = {}
    try:
        with rr64_dataset_jsonl.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                pid = str(obj.get("problem_id") or "").strip()
                if not pid or pid in out:
                    continue
                out[pid] = str(obj.get("statement") or "")
    except Exception:
        return {}
    return out


def _strip_attempt_suffix(attempt_id: str) -> str:
    return re.sub(r"_g\d+$", "", str(attempt_id or ""))


def _scan_compilation_hit_sets(compilation_json: Path) -> Tuple[Set[str], Set[str]]:
    """
    Returns (pass_groups, complete_groups), each a set of group_id (problem_id)
    that has at least one attempt compiled as pass/complete.

    Implemented as a streaming scan over pretty-printed JSON to avoid loading huge files.
    """
    pass_groups, complete_groups, _aligned = _scan_compilation_group_sets(
        compilation_json=compilation_json,
        rr64_dataset_jsonl=None,
    )
    return pass_groups, complete_groups


def _scan_compilation_group_sets(
    *,
    compilation_json: Path,
    rr64_dataset_jsonl: Optional[Path],
    forbid_keywords: Optional[List[str]] = None,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Returns (pass_groups, complete_groups, aligned_complete_groups).

    - pass_groups/complete_groups: as in `_scan_compilation_hit_sets`
    - aligned_complete_groups: complete + Fine-Eval-style statement containment (requires rr64_dataset_jsonl)
    """
    if forbid_keywords is None:
        forbid_keywords = ["axiom", "local_instance"]
    if not compilation_json.exists():
        return set(), set(), set()

    template_chunks_by_pid: Dict[str, List[str]] = {}
    if rr64_dataset_jsonl is not None:
        statement_by_pid = _load_rr64_statement_by_problem_id(rr64_dataset_jsonl)
        template_chunks_by_pid = {pid: _fine_eval_template_chunks(stmt) for pid, stmt in statement_by_pid.items()}

    want_aligned = bool(template_chunks_by_pid) and any(template_chunks_by_pid.values())

    pass_groups: Set[str] = set()
    complete_groups: Set[str] = set()
    aligned_groups: Set[str] = set()

                                                                                               
                                                                                      
    try:
        obj = _read_json(compilation_json)
        if isinstance(obj, list):
            for rec in obj:
                if not isinstance(rec, dict):
                    continue
                attempt = str(rec.get("name") or "").strip()
                if not attempt:
                    continue
                group = (attempt.split("__", 1)[0] or "").strip()
                if not group:
                    continue
                cr = rec.get("compilation_result") or {}
                if isinstance(cr, dict):
                    if bool(cr.get("pass")):
                        pass_groups.add(group)
                    if bool(cr.get("complete")):
                        complete_groups.add(group)
                        if want_aligned:
                            base_pid = _strip_attempt_suffix(attempt)
                            tmpl_chunks = template_chunks_by_pid.get(base_pid) or []
                            if tmpl_chunks:
                                code = str(rec.get("code") or "")
                                out = _remove_comments_lean(code).strip()
                                if out:
                                    bad_kw = any(kw and str(kw) in out for kw in forbid_keywords)
                                    if (not bad_kw) and all(s in out for s in tmpl_chunks):
                                        aligned_groups.add(group)
            return pass_groups, complete_groups, aligned_groups
    except Exception:
                                      
        pass

    cur_attempt: Optional[str] = None
    cur_group: Optional[str] = None
    cur_code_raw: Optional[str] = None
    cur_pass: bool = False
    cur_complete: bool = False
    processed_aligned: bool = False

    pat_obj_start = re.compile(r"^\s{2}\{\s*$")
    pat_obj_end = re.compile(r"^\s{2}\}\s*,?\s*$")
    pat_name = re.compile(r'^\s*"name"\s*:\s*"([^"]+)"')
    pat_pass_true = re.compile(r'^\s*"pass"\s*:\s*true\b')
    pat_complete_true = re.compile(r'^\s*"complete"\s*:\s*true\b')
    pat_code = re.compile(r'^\s*"code"\s*:\s*(.+?)\s*,?\s*$')

    def maybe_process_aligned() -> None:
        nonlocal processed_aligned
        if not want_aligned or processed_aligned:
            return
        if cur_group is None or cur_attempt is None or cur_code_raw is None or not cur_complete:
            return
        base_pid = _strip_attempt_suffix(cur_attempt)
        tmpl_chunks = template_chunks_by_pid.get(base_pid) or []
        if not tmpl_chunks:
            processed_aligned = True
            return
        if not cur_code_raw.startswith('"'):
            processed_aligned = True
            return

                                                                                           
        try:
            code = json.loads(cur_code_raw)
        except Exception:
            processed_aligned = True
            return
        out = _remove_comments_lean(code).strip()
        if not out:
            processed_aligned = True
            return
        for kw in forbid_keywords:
            if kw and str(kw) in out:
                processed_aligned = True
                return
        if all(s in out for s in tmpl_chunks):
            aligned_groups.add(cur_group)
        processed_aligned = True

    def finalize_object() -> None:
        if cur_group is None:
            return
        if cur_pass:
            pass_groups.add(cur_group)
        if cur_complete:
            complete_groups.add(cur_group)
        maybe_process_aligned()

    try:
        with compilation_json.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if pat_obj_start.match(line):
                    cur_attempt = None
                    cur_group = None
                    cur_code_raw = None
                    cur_pass = False
                    cur_complete = False
                    processed_aligned = False
                    continue

                m = pat_name.match(line)
                if m:
                    cur_attempt = m.group(1)
                    cur_group = (cur_attempt.split("__", 1)[0] or "").strip() or None
                                                                             
                    if cur_group is not None:
                        if cur_pass:
                            pass_groups.add(cur_group)
                        if cur_complete:
                            complete_groups.add(cur_group)
                    maybe_process_aligned()
                    continue

                if pat_pass_true.match(line):
                    cur_pass = True
                    if cur_group is not None:
                        pass_groups.add(cur_group)
                    continue

                if pat_complete_true.match(line):
                    cur_complete = True
                    if cur_group is not None:
                        complete_groups.add(cur_group)
                    maybe_process_aligned()
                    continue

                if want_aligned:
                    m = pat_code.match(line)
                    if m and cur_code_raw is None:
                        raw = (m.group(1) or "").strip()
                        if raw.endswith(","):
                            raw = raw[:-1].rstrip()
                        cur_code_raw = raw
                        maybe_process_aligned()
                        continue

                if pat_obj_end.match(line):
                    finalize_object()
                    continue
    except Exception:
        return set(), set(), set()
    return pass_groups, complete_groups, aligned_groups


def _scan_compilation_group_sets_with_aligned_counts(
    *,
    compilation_json: Path,
    rr64_dataset_jsonl: Optional[Path],
    forbid_keywords: Optional[List[str]] = None,
) -> Tuple[Set[str], Set[str], Set[str], Dict[str, Dict[str, int]]]:
    """
    Like `_scan_compilation_group_sets`, but also returns per-group attempt counts:

      - total_attempts: number of attempts observed in `code_compilation_repl.json`
      - aligned_complete_attempts: attempts that are complete and pass Fine-Eval-style containment

    This function uses the fast `json.load` path; if parsing fails, it falls back to
    `_scan_compilation_group_sets` and returns empty counts.
    """
    if forbid_keywords is None:
        forbid_keywords = ["axiom", "local_instance"]
    if not compilation_json.exists():
        return set(), set(), set(), {}

    template_chunks_by_pid: Dict[str, List[str]] = {}
    if rr64_dataset_jsonl is not None and rr64_dataset_jsonl.exists():
        statement_by_pid = _load_rr64_statement_by_problem_id(rr64_dataset_jsonl)
        template_chunks_by_pid = {pid: _fine_eval_template_chunks(stmt) for pid, stmt in statement_by_pid.items()}
    want_aligned = bool(template_chunks_by_pid) and any(template_chunks_by_pid.values())

    pass_groups: Set[str] = set()
    complete_groups: Set[str] = set()
    aligned_groups: Set[str] = set()
    counts: Dict[str, Dict[str, int]] = {}

    try:
        obj = _read_json(compilation_json)
    except Exception:
        p, c, a = _scan_compilation_group_sets(compilation_json=compilation_json, rr64_dataset_jsonl=rr64_dataset_jsonl)
        return p, c, a, {}
    if not isinstance(obj, list):
        return set(), set(), set(), {}

    for rec in obj:
        if not isinstance(rec, dict):
            continue
        attempt = str(rec.get("name") or "").strip()
        if not attempt:
            continue
        group = (attempt.split("__", 1)[0] or "").strip()
        if not group:
            continue
        row = counts.get(group)
        if row is None:
            row = {"total_attempts": 0, "aligned_complete_attempts": 0}
            counts[group] = row
        row["total_attempts"] += 1

        cr = rec.get("compilation_result") or {}
        if isinstance(cr, dict) and bool(cr.get("pass")):
            pass_groups.add(group)
        is_complete = bool(isinstance(cr, dict) and bool(cr.get("complete")))
        if is_complete:
            complete_groups.add(group)
        if not (want_aligned and is_complete):
            continue

        base_pid = _strip_attempt_suffix(attempt)
        tmpl_chunks = template_chunks_by_pid.get(base_pid) or []
        if not tmpl_chunks:
            continue
        code = str(rec.get("code") or "")
        out = _remove_comments_lean(code).strip()
        if not out:
            continue
        if any(kw and str(kw) in out for kw in forbid_keywords):
            continue
        if all(s in out for s in tmpl_chunks):
            aligned_groups.add(group)
            row["aligned_complete_attempts"] += 1

    return pass_groups, complete_groups, aligned_groups, counts


def _scan_compilation_aligned_complete_groups(
    *,
    compilation_json: Path,
    rr64_dataset_jsonl: Path,
    forbid_keywords: Optional[List[str]] = None,
) -> Set[str]:
                                  
    _pass_groups, _complete_groups, aligned_groups = _scan_compilation_group_sets(
        compilation_json=compilation_json,
        rr64_dataset_jsonl=rr64_dataset_jsonl,
        forbid_keywords=forbid_keywords,
    )
    return aligned_groups


def _load_summary(summary_json: Path) -> Optional[Tuple[str, Dict[str, Any]]]:
    try:
        obj = _read_json(summary_json)
    except Exception as e:
        _warn(f"failed to parse {summary_json}: {type(e).__name__}: {e}")
        return None
    if not isinstance(obj, dict):
        return None
    field = str(obj.get("field") or "").strip()
    overall = obj.get("overall") or {}
    if not field or not isinstance(overall, dict):
        return None
    return field, overall


def _load_solved_theorem_complete_set(per_problem_json: Path) -> Set[str]:
    """
    Load the set of group_ids where `solved_theorem_complete` is true.

    This is typically a small file (<= number of benchmark problems).
    """
    if not per_problem_json.exists():
        return set()
    try:
        obj = _read_json(per_problem_json)
    except Exception:
        return set()
    if not isinstance(obj, list):
        return set()
    out: Set[str] = set()
    for row in obj:
        if not isinstance(row, dict):
            continue
        gid = str(row.get("group_id") or "").strip()
        if not gid:
            continue
        if bool(row.get("solved_theorem_complete")):
            out.add(gid)
    return out


def _load_per_problem_attempt_counts(per_problem_json: Path) -> Dict[str, Dict[str, int]]:
    """
    Returns per-group attempt counts from `portfolio_summary_pass/per_problem.json`:
      - total_attempts
      - pass_attempts
      - complete_attempts
      - theorem_complete_attempts
    """
    if not per_problem_json.exists():
        return {}
    try:
        obj = _read_json(per_problem_json)
    except Exception:
        return {}
    if not isinstance(obj, list):
        return {}
    out: Dict[str, Dict[str, int]] = {}
    for row in obj:
        if not isinstance(row, dict):
            continue
        gid = str(row.get("group_id") or "").strip()
        if not gid:
            continue
        attempts = row.get("attempts") or []
        if not isinstance(attempts, list):
            continue
        total = len(attempts)
        pass_n = 0
        complete_n = 0
        theorem_n = 0
        for a in attempts:
            if not isinstance(a, dict):
                continue
            if bool(a.get("pass_ok")):
                pass_n += 1
            if bool(a.get("complete_ok")):
                complete_n += 1
            if bool(a.get("theorem_complete_ok")):
                theorem_n += 1
        out[gid] = {
            "total_attempts": int(total),
            "pass_attempts": int(pass_n),
            "complete_attempts": int(complete_n),
            "theorem_complete_attempts": int(theorem_n),
        }
    return out


def _summarize_density(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    xs = sorted(values)
    n = len(xs)
    def q(p: float) -> float:
        if n == 1:
            return float(xs[0])
        idx = int(round(p * float(n - 1)))
        idx = max(0, min(n - 1, idx))
        return float(xs[idx])
    return {
        "n": n,
        "mean": float(sum(xs) / float(n)),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
    }


def _paired_density_comparison(
    *,
    ours_counts: Dict[str, Dict[str, int]],
    base_counts: Dict[str, Dict[str, int]],
    denom: Set[str],
    key_num: str,
    key_den: str = "total_attempts",
) -> Dict[str, Any]:
    """
    Compare per-problem densities on a shared denominator set.

    Returns summary stats of per-problem ratios and paired differences.
    """
    denom2 = sorted(set(denom or set()))
    ours_vals: List[float] = []
    base_vals: List[float] = []
    diffs: List[float] = []
    wins = 0
    losses = 0
    ties = 0
    for gid in denom2:
        o = ours_counts.get(gid) or {}
        b = base_counts.get(gid) or {}
        od = _safe_int(o.get(key_den), 0)
        bd = _safe_int(b.get(key_den), 0)
        if od <= 0 or bd <= 0:
                                                                       
            continue
        ov = float(_safe_int(o.get(key_num), 0)) / float(od)
        bv = float(_safe_int(b.get(key_num), 0)) / float(bd)
        ours_vals.append(ov)
        base_vals.append(bv)
        d = ov - bv
        diffs.append(d)
        if d > 0:
            wins += 1
        elif d < 0:
            losses += 1
        else:
            ties += 1
    if not diffs:
        return {"n": 0}
    diff_sorted = sorted(diffs)
    n = len(diff_sorted)
    return {
        "n": n,
        "ours": _summarize_density(ours_vals),
        "baseline": _summarize_density(base_vals),
        "diff": {
            "mean": float(sum(diffs) / float(n)),
            "p50": float(diff_sorted[n // 2]),
            "wins": wins,
            "losses": losses,
            "ties": ties,
        },
    }


def _density_on_subset(
    *,
    counts: Dict[str, Dict[str, int]],
    subset: Set[str],
    key_num: str,
    key_den: str = "total_attempts",
) -> List[float]:
    vals: List[float] = []
    for gid in subset:
        row = counts.get(gid) or {}
        den = _safe_int(row.get(key_den), 0)
        if den <= 0:
            continue
        vals.append(float(_safe_int(row.get(key_num), 0)) / float(den))
    return vals


def _head_to_head(*, ours: Set[str], baseline: Set[str], denom: Set[str]) -> Dict[str, Any]:
    """
    Pairwise comparison on a fixed denominator set of problem IDs.
    """
    denom2 = set(denom or set())
    ours2 = set(ours or set()) & denom2
    base2 = set(baseline or set()) & denom2
    ours_only = ours2 - base2
    base_only = base2 - ours2
    both = ours2 & base2
    neither = denom2 - (ours2 | base2)

    wins = len(ours_only)
    losses = len(base_only)
    decisive = wins + losses
    return {
        "total": len(denom2),
        "ours_solved": len(ours2),
        "baseline_solved": len(base2),
        "ours_rate": (float(len(ours2)) / float(len(denom2))) if denom2 else 0.0,
        "baseline_rate": (float(len(base2)) / float(len(denom2))) if denom2 else 0.0,
        "wins": wins,
        "losses": losses,
        "decisive": decisive,
        "win_rate": (float(wins) / float(decisive)) if decisive > 0 else None,
        "both": len(both),
        "neither": len(neither),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate proof-utility RR results into a single CSV/JSON table.")
    ap.add_argument("--out_root", required=True, help="Root directory containing per-method RR outputs.")
    ap.add_argument(
        "--methods",
        default="ours,sample,strong_compile,strong_semantic",
        help="Comma-separated method directories to scan.",
    )
    ap.add_argument("--tags", default="rr64", help="Comma-separated tags to scan.")
    args = ap.parse_args()

    out_root = Path(str(args.out_root)).expanduser().resolve()
    if not out_root.exists():
        _warn(f"out_root not found: {out_root}")
        return 2

    methods = [s.strip() for s in str(args.methods or "").split(",") if s.strip()]
    tags = [s.strip() for s in str(args.tags or "").split(",") if s.strip()]
    if not methods or not tags:
        _warn("empty --methods or --tags")
        return 2

    rows: List[Row] = []
    benchmark_all_ids_by_tag: Dict[str, Set[str]] = {}
    for tag in tags:
        preferred = out_root / "ours" / tag / f"{tag}_dataset.index.json"
        if preferred.exists():
            all_ids, _active = _load_index_problem_sets(preferred)
            if all_ids:
                benchmark_all_ids_by_tag[tag] = all_ids
                continue
        for method in methods:
            cand = out_root / method / tag / f"{tag}_dataset.index.json"
            if cand.exists():
                all_ids, _active = _load_index_problem_sets(cand)
                if all_ids:
                    benchmark_all_ids_by_tag[tag] = all_ids
                    break

    for method in methods:
        for tag in tags:
            tag_dir = out_root / method / tag
            if not tag_dir.exists():
                continue

            index_json = tag_dir / f"{tag}_dataset.index.json"
            budget = _infer_budget_from_index(index_json) if index_json.exists() else 0
            all_ids_method, active_ids_method = _load_index_problem_sets(index_json)

            benchmark_all_ids = benchmark_all_ids_by_tag.get(tag, set()) or all_ids_method
            problems_all = len(benchmark_all_ids)
            problems_alloc_gt0 = len(active_ids_method)

            compilation_json = tag_dir / "goedel_out" / "code_compilation_repl.json"
            pass_groups, complete_groups, aligned_groups = _scan_compilation_group_sets(
                compilation_json=compilation_json,
                rr64_dataset_jsonl=tag_dir / f"{tag}_dataset.jsonl",
            )
            pass_hit_all = len(pass_groups & benchmark_all_ids) if benchmark_all_ids else len(pass_groups)
            complete_hit_all = len(complete_groups & benchmark_all_ids) if benchmark_all_ids else len(complete_groups)
            aligned_hit_all = len(aligned_groups & benchmark_all_ids) if benchmark_all_ids else len(aligned_groups)
            pass_hit_alloc = len(pass_groups & active_ids_method) if active_ids_method else 0
            complete_hit_alloc = len(complete_groups & active_ids_method) if active_ids_method else 0
            aligned_hit_alloc = len(aligned_groups & active_ids_method) if active_ids_method else 0

            for field in ["pass", "complete"]:
                summary_json = tag_dir / "goedel_out" / f"portfolio_summary_{field}" / "summary.json"
                if not summary_json.exists():
                    continue
                loaded = _load_summary(summary_json)
                if not loaded:
                    continue
                field2, overall = loaded

                solved_all = pass_hit_all if field == "pass" else complete_hit_all
                solved_ratio_all = float(solved_all) / float(problems_all) if problems_all > 0 else 0.0
                solved_aligned_ratio_all = float(aligned_hit_all) / float(problems_all) if problems_all > 0 else 0.0

                solved_alloc = pass_hit_alloc if field == "pass" else complete_hit_alloc
                solved_ratio_alloc = float(solved_alloc) / float(problems_alloc_gt0) if problems_alloc_gt0 > 0 else 0.0
                solved_aligned_ratio_alloc = (
                    float(aligned_hit_alloc) / float(problems_alloc_gt0) if problems_alloc_gt0 > 0 else 0.0
                )

                solved_theorem_complete = _safe_int(overall.get("solved_theorem_complete"), 0)
                solved_theorem_complete_ratio = (
                    float(solved_theorem_complete) / float(problems_all) if problems_all > 0 else 0.0
                )
                solved_theorem_complete_ratio_alloc = (
                    float(solved_theorem_complete) / float(problems_alloc_gt0) if problems_alloc_gt0 > 0 else 0.0
                )
                rows.append(
                    Row(
                        method=method,
                        tag=tag,
                        field=field2 or field,
                        budget=budget,
                        problems=int(problems_all),
                        solved=int(solved_all),
                        solved_ratio=float(solved_ratio_all),
                        solved_aligned_complete=int(aligned_hit_all),
                        solved_aligned_complete_ratio=float(solved_aligned_ratio_all),
                        solved_theorem_complete=solved_theorem_complete,
                        solved_theorem_complete_ratio=solved_theorem_complete_ratio,
                        problems_alloc_gt0=int(problems_alloc_gt0),
                        solved_alloc_gt0=int(solved_alloc),
                        solved_ratio_alloc_gt0=float(solved_ratio_alloc),
                        solved_aligned_complete_alloc_gt0=int(aligned_hit_alloc),
                        solved_aligned_complete_ratio_alloc_gt0=float(solved_aligned_ratio_alloc),
                        solved_theorem_complete_ratio_alloc_gt0=solved_theorem_complete_ratio_alloc,
                    )
                )

    if not rows:
        _warn("no summary.json files found under out_root")
        return 2

                                                                                                       
                                                                          
                                              
                                                                               
    head_to_head: Dict[str, Any] = {}
    for tag in tags:
        benchmark_all_ids = benchmark_all_ids_by_tag.get(tag, set())
        if not benchmark_all_ids:
            continue

        per_method: Dict[str, Dict[str, Set[str]]] = {}
        per_method_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
        for method in methods:
            tag_dir = out_root / method / tag
            index_json = tag_dir / f"{tag}_dataset.index.json"
            _all_ids_method, active_ids_method = _load_index_problem_sets(index_json)
            compilation_json = tag_dir / "goedel_out" / "code_compilation_repl.json"
            pass_groups, complete_groups, aligned_groups = _scan_compilation_group_sets(
                compilation_json=compilation_json,
                rr64_dataset_jsonl=tag_dir / f"{tag}_dataset.jsonl",
            )
            per_problem_json = tag_dir / "goedel_out" / "portfolio_summary_pass" / "per_problem.json"
            theorem_complete_groups = _load_solved_theorem_complete_set(per_problem_json)
            per_method_counts[method] = _load_per_problem_attempt_counts(per_problem_json)

            per_method[method] = {
                "active": active_ids_method & benchmark_all_ids,
                "pass": pass_groups & benchmark_all_ids,
                "complete": complete_groups & benchmark_all_ids,
                "aligned_complete": aligned_groups & benchmark_all_ids,
                "theorem_complete": theorem_complete_groups & benchmark_all_ids,
            }

        if "ours" not in per_method:
            continue

        head_to_head[tag] = {}
        ours_active = per_method["ours"]["active"]
        for baseline in [m for m in methods if m != "ours" and m in per_method]:
            base_active = per_method[baseline]["active"]
            overlap_attempted = ours_active & base_active
            entry: Dict[str, Any] = {
                "baseline": baseline,
                "denominators": {
                    "all": len(benchmark_all_ids),
                    "overlap_attempted": len(overlap_attempted),
                },
                "metrics": {},
            }
            for metric in ["pass", "complete", "aligned_complete", "theorem_complete"]:
                entry["metrics"][metric] = {
                    "all": _head_to_head(
                        ours=per_method["ours"][metric],
                        baseline=per_method[baseline][metric],
                        denom=benchmark_all_ids,
                    ),
                    "overlap_attempted": _head_to_head(
                        ours=per_method["ours"][metric],
                        baseline=per_method[baseline][metric],
                        denom=overlap_attempted,
                    ),
                }

                                                                                                         
                                                                                                    
                                                                                                      
                                                               
            ours_counts = per_method_counts.get("ours") or {}
            base_counts = per_method_counts.get(baseline) or {}

                                                                                                             
            _p, _c, _a, ours_aligned_counts = _scan_compilation_group_sets_with_aligned_counts(
                compilation_json=out_root / "ours" / tag / "goedel_out" / "code_compilation_repl.json",
                rr64_dataset_jsonl=out_root / "ours" / tag / f"{tag}_dataset.jsonl",
            )
            _p, _c, _a, base_aligned_counts = _scan_compilation_group_sets_with_aligned_counts(
                compilation_json=out_root / baseline / tag / "goedel_out" / "code_compilation_repl.json",
                rr64_dataset_jsonl=out_root / baseline / tag / f"{tag}_dataset.jsonl",
            )
            entry["densities"] = {
                "pass": {
                    "overlap_attempted": _paired_density_comparison(
                        ours_counts=ours_counts, base_counts=base_counts, denom=overlap_attempted, key_num="pass_attempts"
                    )
                },
                "complete": {
                    "overlap_attempted": _paired_density_comparison(
                        ours_counts=ours_counts,
                        base_counts=base_counts,
                        denom=overlap_attempted,
                        key_num="complete_attempts",
                    )
                },
                "aligned_complete": {
                    "overlap_attempted": _paired_density_comparison(
                        ours_counts=ours_aligned_counts,
                        base_counts=base_aligned_counts,
                        denom=overlap_attempted,
                        key_num="aligned_complete_attempts",
                    )
                },
                "theorem_complete": {
                    "overlap_attempted": _paired_density_comparison(
                        ours_counts=ours_counts,
                        base_counts=base_counts,
                        denom=overlap_attempted,
                        key_num="theorem_complete_attempts",
                    )
                },
            }

                                                                                                              
                                                                                     
            common_correct: Dict[str, Any] = {}
            for metric, num_key, solved_key in [
                ("pass", "pass_attempts", "pass"),
                ("complete", "complete_attempts", "complete"),
                ("aligned_complete", "aligned_complete_attempts", "aligned_complete"),
                ("theorem_complete", "theorem_complete_attempts", "theorem_complete"),
            ]:
                both_solved = (
                    (per_method["ours"][solved_key] & per_method[baseline][solved_key]) & overlap_attempted
                )
                if metric == "aligned_complete":
                    ours_vals = _density_on_subset(counts=ours_aligned_counts, subset=both_solved, key_num=num_key)
                    base_vals = _density_on_subset(counts=base_aligned_counts, subset=both_solved, key_num=num_key)
                else:
                    ours_vals = _density_on_subset(counts=ours_counts, subset=both_solved, key_num=num_key)
                    base_vals = _density_on_subset(counts=base_counts, subset=both_solved, key_num=num_key)
                common_correct[metric] = {
                    "both_solved_n": len(both_solved),
                    "ours": _summarize_density(ours_vals),
                    "baseline": _summarize_density(base_vals),
                    "diff_mean": (float(sum(ours_vals) / len(ours_vals)) - float(sum(base_vals) / len(base_vals)))
                    if (ours_vals and base_vals)
                    else None,
                }
            entry["common_correct_density"] = common_correct
            head_to_head[tag][baseline] = entry

    rows.sort(key=lambda r: (r.method, r.tag, r.field))

    csv_path = out_root / "proof_utility_rr_table.csv"
    json_path = out_root / "proof_utility_rr_table.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "tag",
                "field",
                "budget",
                "problems",
                "solved",
                "solved_ratio",
                "solved_aligned_complete",
                "solved_aligned_complete_ratio",
                "solved_theorem_complete",
                "solved_theorem_complete_ratio",
                "problems_alloc_gt0",
                "solved_alloc_gt0",
                "solved_ratio_alloc_gt0",
                "solved_aligned_complete_alloc_gt0",
                "solved_aligned_complete_ratio_alloc_gt0",
                "solved_theorem_complete_ratio_alloc_gt0",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.method,
                    r.tag,
                    r.field,
                    int(r.budget),
                    int(r.problems),
                    int(r.solved),
                    f"{r.solved_ratio:.6g}",
                    int(r.solved_aligned_complete),
                    f"{r.solved_aligned_complete_ratio:.6g}",
                    int(r.solved_theorem_complete),
                    f"{r.solved_theorem_complete_ratio:.6g}",
                    int(r.problems_alloc_gt0),
                    int(r.solved_alloc_gt0),
                    f"{r.solved_ratio_alloc_gt0:.6g}",
                    int(r.solved_aligned_complete_alloc_gt0),
                    f"{r.solved_aligned_complete_ratio_alloc_gt0:.6g}",
                    f"{r.solved_theorem_complete_ratio_alloc_gt0:.6g}",
                ]
            )

    json_path.write_text(
        json.dumps(
            [
                {
                    "method": r.method,
                    "tag": r.tag,
                    "field": r.field,
                    "budget": r.budget,
                    "problems": r.problems,
                    "solved": r.solved,
                    "solved_ratio": r.solved_ratio,
                    "solved_aligned_complete": r.solved_aligned_complete,
                    "solved_aligned_complete_ratio": r.solved_aligned_complete_ratio,
                    "solved_theorem_complete": r.solved_theorem_complete,
                    "solved_theorem_complete_ratio": r.solved_theorem_complete_ratio,
                    "problems_alloc_gt0": r.problems_alloc_gt0,
                    "solved_alloc_gt0": r.solved_alloc_gt0,
                    "solved_ratio_alloc_gt0": r.solved_ratio_alloc_gt0,
                    "solved_aligned_complete_alloc_gt0": r.solved_aligned_complete_alloc_gt0,
                    "solved_aligned_complete_ratio_alloc_gt0": r.solved_aligned_complete_ratio_alloc_gt0,
                    "solved_theorem_complete_ratio_alloc_gt0": r.solved_theorem_complete_ratio_alloc_gt0,
                }
                for r in rows
            ],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    meta = {
        "out_root": str(out_root),
        "rows": len(rows),
        "methods_scanned": methods,
        "tags_scanned": tags,
        "notes": {
            "problems/solved/solved_ratio": "All-denominator (full benchmark problems, counts missing-candidate problems as failures).",
            "solved_aligned_complete_ratio": "Fine-Eval-style aligned complete: compilation_result.complete + output contains the input rr64 statement skeleton (with `sorry` removed), excluding import/open/set_option chunks.",
            "problems_alloc_gt0/solved_alloc_gt0/solved_ratio_alloc_gt0": "alloc>0 denominator (problems with non-empty allocations in method index).",
            "solved_aligned_complete_ratio_alloc_gt0": "solved_aligned_complete / problems_alloc_gt0.",
            "solved_theorem_complete_ratio": "solved_theorem_complete / problems (all-denominator).",
            "solved_theorem_complete_ratio_alloc_gt0": "solved_theorem_complete / problems_alloc_gt0.",
            "head_to_head": "Pairwise win/loss stats between ours and baselines on (a) all problems and (b) overlap-attempted problems (active_ids_ours ∩ active_ids_baseline), to reduce variance from empty repertoires and coverage differences.",
        },
        "head_to_head": head_to_head,
    }
    (out_root / "proof_utility_rr_table.csv.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
