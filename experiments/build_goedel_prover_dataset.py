#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from functools import lru_cache
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.dont_write_bytecode = True

LEAN_DECL_KEYWORDS = [
    "theorem",
    "lemma",
    "def",
    "example",
    "axiom",
    "abbrev",
    "opaque",
    "instance",
]

_DECL_START_PAT = re.compile(rf"^\s*(?:noncomputable\s+)?(?:{'|'.join(LEAN_DECL_KEYWORDS)})\b")
_DECL_NAME_PAT = re.compile(
    rf"^\s*(?:noncomputable\s+)?(?P<kw>{'|'.join(LEAN_DECL_KEYWORDS)})\b(?:\s+(?P<name>[^\s(:]+))?"
)


@dataclass(frozen=True)
class ProgramRow:
    program_id: str
    code: str
    public_metrics: Dict[str, Any]
    combined_score: float
    timestamp: float


@dataclass(frozen=True)
class SelectedStatement:
    program_id: str
    statement: str
    signature: str
    combined_score: float
    public_metrics: Dict[str, Any]
    code_len: int
    timestamp: float


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


_ABBREV_SORRY_LINE_RE = re.compile(
    r"^(?P<indent>\s*)abbrev\s+(?P<name>[^\s:]+)\s*:\s*(?P<type>.+?)\s*:=\s*sorry\s*$"
)


def _sanitize_header_sorry_to_opaque(header: str) -> str:
    """
    Make headers compatible with Goedel's `complete` metric by removing `:= sorry`.

    CombiBench headers typically define a `_solution` constant as:
      abbrev foo_solution : ℕ := sorry

    This causes `complete=false` even when the theorem proof itself has no `sorry`.
    Here we rewrite those lines into an opaque constant declaration:
      opaque foo_solution : ℕ
    """
    lines = str(header or "").splitlines()
    out: List[str] = []
    for line in lines:
        m = _ABBREV_SORRY_LINE_RE.match(line.rstrip())
        if not m:
            out.append(line.rstrip())
            continue
        indent = m.group("indent") or ""
        name = m.group("name")
        ty = m.group("type")
        out.append(f"{indent}opaque {name} : {ty}")
    return "\n".join(out).strip()


def _compose_lean_file(*, header: str, statement: str) -> str:
    h = str(header or "").strip()
    s = str(statement or "").strip()
    if h:
        return h.rstrip() + "\n\n" + s
    return s


_THEOREM_START_RE = re.compile(r"(?m)^\s*(?:noncomputable\s+)?theorem\b")
_LEADING_NUM_PREFIX_RE = re.compile(r"^\d+_")


def _split_header_and_theorem(formal_statement: str) -> Tuple[str, str]:
    """
    Split a full Lean file into (header, theorem_block) by the first `theorem` keyword.

    This matches CombiBench's convention: the header may contain `abbrev *_solution := sorry`
    lines (fill-in-the-blank), while the theorem block contains `:= by sorry`.
    """
    text = str(formal_statement or "").strip()
    if not text:
        return "", ""
    m = _THEOREM_START_RE.search(text)
    if not m:
        return text.strip(), ""
    header = text[: m.start()].strip()
    theorem = text[m.start() :].strip()
    return header, theorem


def _combibench_theorem_name(problem_id: str) -> str:
    """
    Map our local CombiBench problem_id (e.g. '0000_hackmath_1') to HF theorem_name ('hackmath_1').
    """
    s = str(problem_id or "").strip()
    return _LEADING_NUM_PREFIX_RE.sub("", s)


@lru_cache(maxsize=8)
def _load_hf_formal_map(
    *,
    dataset_name: str,
    split: str,
    index_column: str,
    formal_column: str,
) -> Dict[str, str]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency for --hf_dataset: `datasets`. "
            "Install it or run without HF overrides."
        ) from e

    ds = load_dataset(str(dataset_name), split=str(split))
    out: Dict[str, str] = {}
    for ex in ds:
        try:
            key = str(ex.get(index_column) or "").strip()
            val = str(ex.get(formal_column) or "").strip()
        except Exception:
            continue
        if not key or not val:
            continue
        out[key] = val
    return out


def _resolve_manifest_path(run_root: Path, raw_path: Any, *, fallback: Optional[Path] = None) -> Optional[Path]:
    """
    Resolve a path recorded in manifest.json / problem_config.json.

    - Old runs may contain absolute paths from a different machine/user.
    - Newer runs may store relative paths for portability.
    """
    candidates: List[Path] = []
    if raw_path:
        try:
            p = Path(str(raw_path)).expanduser()
            candidates.append(p)
            if not p.is_absolute():
                candidates.append(run_root / p)
        except Exception:
            pass
    if fallback is not None:
        candidates.append(fallback)
    for c in candidates:
        try:
            if c.exists():
                return c.resolve()
        except Exception:
            continue
    return None


def _resolve_run_dir(run_root: Path, baseline: str, problem_id: str, run_dir_raw: Any) -> Optional[Path]:
    """
    Resolve a problem's run_dir from manifest.

    - Old manifests may contain absolute paths from a different machine/user.
    - Newer manifests may store relative paths for portability.
    - Fallback to the canonical layout under run_root: runs/<baseline>/<problem_id>.
    """
    candidates: List[Path] = []

    if run_dir_raw:
        try:
            raw = Path(str(run_dir_raw)).expanduser()
            candidates.append(raw)
            if not raw.is_absolute():
                candidates.append(run_root / raw)
        except Exception:
            pass

    candidates.append(run_root / "runs" / str(baseline) / str(problem_id))

    for c in candidates:
        try:
            if c.exists():
                return c
        except Exception:
            continue
    return None


def _is_ok(pm: Dict[str, Any], key: str) -> bool:
    try:
        return int(pm.get(key, 0) or 0) == 1
    except Exception:
        return False


def _clean_code(code: str) -> str:
    if not code:
        return ""
    cleaned_lines: List[str] = []
    for line in code.splitlines():
        if re.match(r"^\s*(?:#|//|--)?\s*EVOLVE-BLOCK-(?:START|END)\s*$", line):
            continue
        s = line.strip()
        if s.startswith("import ") or s.startswith("open ") or s.startswith("open scoped "):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _extract_first_decl(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    decl_idx = None
    for i, line in enumerate(lines):
        if _DECL_START_PAT.search(line):
            decl_idx = i
            break
    if decl_idx is None:
        return ""
    start = decl_idx
    while start > 0 and lines[start - 1].lstrip().startswith("@["):
        start -= 1
    next_idx = None
    for j in range(decl_idx + 1, len(lines)):
        if _DECL_START_PAT.search(lines[j]):
            next_idx = j
            break
    end = next_idx if next_idx is not None else len(lines)
    return "\n".join(lines[start:end]).strip()


def _decl_signature(code: str) -> str:
    decl = _extract_first_decl(_clean_code(code))
    if not decl:
        return ""

    lines = decl.splitlines()
    for i, line in enumerate(lines):
        m = _DECL_NAME_PAT.match(line)
        if not m:
            continue
        kw = m.group("kw")
        name = m.group("name")
        if kw and name:
            lines[i] = re.sub(
                rf"\b{re.escape(kw)}\s+{re.escape(name)}\b",
                f"{kw} _",
                line,
                count=1,
            )
        break

    decl2 = "\n".join(lines)
    decl2 = decl2.split(":=", 1)[0].rstrip()
    return re.sub(r"\s+", " ", decl2).strip()


def _force_by_sorry(stmt: str) -> str:
    stmt = (stmt or "").strip()
    if not stmt:
        return ""
    # Normalize all declarations to the same prover-friendly skeleton.
    #
    # Why:
    # - Some exported statements end with `:= sorry` (no `by`), while Goedel's prompting/merging logic
    #   expects a `:= by` marker.
    # - Even when `:= by` exists, we want to strip any existing body and canonicalize to `by sorry`.
    #
    # This makes downstream proving robust and keeps k-selection comparable.
    if ":=" not in stmt:
        return stmt
    head = stmt.split(":=", 1)[0].rstrip()
    return head + " := by sorry"


def _read_program_rows(run_dir: Path) -> List[ProgramRow]:
    db_path = run_dir / "evolution_db.sqlite"
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT id, code, public_metrics, combined_score, timestamp FROM programs")
        rows = cur.fetchall()
        conn.close()
    except Exception:
        return []

    out: List[ProgramRow] = []
    for program_id, code, pm_raw, combined_score, ts in rows:
        if pm_raw is None:
            continue
        try:
            pm = json.loads(pm_raw) if isinstance(pm_raw, str) else dict(pm_raw)
        except Exception:
            continue
        try:
            cs = float(combined_score or 0.0)
        except Exception:
            cs = 0.0
        try:
            ts_f = float(ts or 0.0)
        except Exception:
            ts_f = 0.0
        out.append(
            ProgramRow(
                program_id=str(program_id or ""),
                code=str(code or ""),
                public_metrics=pm,
                combined_score=cs,
                timestamp=ts_f,
            )
        )
    return out


def _select_topk_unique(
    programs: Iterable[ProgramRow],
    *,
    k: int,
    prefer_semantic: bool,
    require_semantic: bool,
) -> List[SelectedStatement]:
    compile_ok = [p for p in programs if _is_ok(p.public_metrics, "compile_ok")]

    def best_per_signature(cands: Iterable[ProgramRow]) -> Dict[str, SelectedStatement]:
        best: Dict[str, SelectedStatement] = {}
        for p in cands:
            stmt = _force_by_sorry(_extract_first_decl(_clean_code(p.code)))
            sig = _decl_signature(p.code)
            if not stmt or not sig:
                continue
            cand = SelectedStatement(
                program_id=p.program_id,
                statement=stmt,
                signature=sig,
                combined_score=float(p.combined_score),
                public_metrics=p.public_metrics,
                code_len=len(p.code or ""),
                timestamp=float(p.timestamp),
            )
            prev = best.get(sig)
            if prev is None:
                best[sig] = cand
                continue

            def _rank(x: SelectedStatement) -> Tuple[int, float, int, int, int, float]:
                return (
                    int(_is_ok(x.public_metrics, "semantic_ok")),
                    float(x.combined_score),
                    -len(x.signature or ""),
                    -len(x.statement or ""),
                    -int(x.code_len),
                    -float(x.timestamp),
                )

            if _rank(cand) > _rank(prev):
                best[sig] = cand
        return best

    if require_semantic:
        semantic_ok = [p for p in compile_ok if _is_ok(p.public_metrics, "semantic_ok")]
        uniq = list(best_per_signature(semantic_ok).values())
        uniq.sort(
            key=lambda x: (
                int(_is_ok(x.public_metrics, "semantic_ok")),
                float(x.combined_score),
                -len(x.signature or ""),
                -len(x.statement or ""),
                -int(x.code_len),
            ),
            reverse=True,
        )
        return uniq[:k]

    semantic_ok = [p for p in compile_ok if _is_ok(p.public_metrics, "semantic_ok")]
    uniq_sem = list(best_per_signature(semantic_ok).values())
    uniq_all = list(best_per_signature(compile_ok).values())

    uniq_sem.sort(
        key=lambda x: (
            int(_is_ok(x.public_metrics, "semantic_ok")),
            float(x.combined_score),
            -len(x.signature or ""),
            -len(x.statement or ""),
            -int(x.code_len),
        ),
        reverse=True,
    )
    uniq_all.sort(
        key=lambda x: (
            int(_is_ok(x.public_metrics, "semantic_ok")),
            float(x.combined_score),
            -len(x.signature or ""),
            -len(x.statement or ""),
            -int(x.code_len),
        ),
        reverse=True,
    )

    out: List[SelectedStatement] = []
    seen_sig: set[str] = set()
    if prefer_semantic:
        for c in uniq_sem:
            if len(out) >= k:
                break
            if c.signature in seen_sig:
                continue
            out.append(c)
            seen_sig.add(c.signature)
    for c in uniq_all:
        if len(out) >= k:
            break
        if c.signature in seen_sig:
            continue
        out.append(c)
        seen_sig.add(c.signature)
    return out


def _read_manifest_run_root(run_root: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    manifest = _read_json(run_root / "manifest.json") or {}
    baselines = manifest.get("baselines")
    if not isinstance(baselines, list) or not baselines:
        runs_dir = run_root / "runs"
        if runs_dir.is_dir():
            baselines = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()])
        else:
            baselines = []
    problems = manifest.get("problems") or []
    if not isinstance(problems, list):
        problems = []
    return [str(x) for x in baselines], [p for p in problems if isinstance(p, dict)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build a Goedel-Prover-V2 input JSONL from autoformalization run roots.\n"
            "Metric: statement-portfolio proof_hit@K (K statements / problem, one prover sample each).\n"
            "Note: Goedel's inference.py renames each input as <problem_id>_g{j}, so we encode grouping into problem_id.\n"
            "- with_solution toggle (ground_truth): optionally fill *_solution from GT to enable true `complete`.\n"
        )
    )
    ap.add_argument("--run_root", type=str, required=True, help="Run root containing manifest.json + runs/.")
    ap.add_argument(
        "--mode",
        type=str,
        default="selected",
        choices=["selected", "ground_truth"],
        help="Dataset mode: selected (from evolution_db.sqlite) or ground_truth (from inputs/*/problem_config.json).",
    )
    ap.add_argument(
        "--solution_mode",
        type=str,
        default="without_solution",
        choices=["without_solution", "with_solution"],
        help=(
            "When mode=ground_truth, choose whether to keep *_solution := sorry (without_solution) "
            "or to fill the solution from GT (with_solution, useful for true `complete`)."
        ),
    )
    ap.add_argument(
        "--baseline",
        type=str,
        default="",
        help="Baseline name under runs/. If empty and only one exists, auto-pick it.",
    )
    ap.add_argument("--k", type=int, default=16, help="Max statements per problem (default: 16).")
    ap.add_argument(
        "--prefer_semantic",
        action="store_true",
        help="Prefer semantic_ok=1 statements; fill remaining slots with compile_ok=1.",
    )
    ap.add_argument(
        "--require_semantic",
        action="store_true",
        help="Only output semantic_ok=1 statements (may output fewer than K per problem).",
    )
    ap.add_argument(
        "--max_problems",
        type=int,
        default=0,
        help="If >0, only process the first N problems in manifest order.",
    )
    ap.add_argument(
        "--problems_file",
        type=str,
        default="",
        help="Optional file with one problem_id per line (only process those).",
    )
    ap.add_argument(
        "--require_header_no_sorry",
        action="store_true",
        help="Skip problems whose header contains the substring 'sorry' (useful when evaluating `complete`).",
    )
    ap.add_argument(
        "--sanitize_header_sorry",
        action="store_true",
        help="Rewrite `abbrev x : T := sorry` in headers into `opaque x : T` (to make `complete` meaningful).",
    )
    ap.add_argument(
        "--hf_dataset",
        type=str,
        default="",
        help=(
            "Optional HuggingFace dataset id to override ground-truth formal statements "
            "(e.g. AI-MO/CombiBench). Only applies to --mode ground_truth."
        ),
    )
    ap.add_argument(
        "--hf_split",
        type=str,
        default="",
        help=(
            "HF split name when using --hf_dataset (e.g. test or test_with_solution). "
            "Only applies to --mode ground_truth."
        ),
    )
    ap.add_argument("--hf_index_column", type=str, default="theorem_name", help="HF index column (default: theorem_name).")
    ap.add_argument(
        "--hf_formal_column",
        type=str,
        default="formal_statement",
        help="HF formal statement column (default: formal_statement).",
    )
    ap.add_argument("--out", type=str, required=True, help="Output JSONL path.")
    ap.add_argument(
        "--index_out",
        type=str,
        default="",
        help="Optional JSON output path for a structured selection index.",
    )
    args = ap.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    if not (run_root / "manifest.json").exists():
        _warn(f"manifest.json not found under: {run_root}")
        return 2

    baselines, problems = _read_manifest_run_root(run_root)
    if str(args.mode) == "selected":
        if not baselines:
            _warn(f"no baselines found under: {run_root / 'runs'}")
            return 2

    baseline = str(args.baseline or "").strip()
    if str(args.mode) == "ground_truth":
        if not baseline:
            baseline = "ground_truth"
    else:
        if not baseline:
            if len(baselines) != 1:
                _warn(f"--baseline required; available baselines={baselines}")
                return 2
            baseline = baselines[0]
        if baseline not in baselines:
            _warn(f"baseline not found: {baseline}; available baselines={baselines}")
            return 2

    allow: Optional[set[str]] = None
    if args.problems_file:
        p = Path(args.problems_file).expanduser()
        allow = {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}

    selected_index: Dict[str, Any] = {
        "run_root": str(run_root),
        "baseline": baseline,
        "k": int(args.k),
        "solution_mode": str(args.solution_mode),
        "problems": [],
    }
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prefer_semantic = bool(args.prefer_semantic)
    require_semantic = bool(args.require_semantic)
    if require_semantic:
        prefer_semantic = True

    num_written = 0
    num_problems = 0
    num_missing = 0
    num_empty = 0
    num_under_k = 0
    num_header_sorry_skipped = 0
    num_missing_problem_config = 0
    hf_formal_by_name: Dict[str, str] = {}
    # Auto-fill HF defaults for with_solution if none provided.
    if str(args.mode) == "ground_truth" and str(args.solution_mode) == "with_solution":
        if not str(args.hf_dataset or "").strip():
            args.hf_dataset = "AI-MO/CombiBench"
        if not str(args.hf_split or "").strip():
            args.hf_split = "test_with_solution"

    use_hf = bool(str(args.hf_dataset or "").strip()) and bool(str(args.hf_split or "").strip())
    if use_hf and str(args.mode) != "ground_truth":
        _warn("--hf_dataset/--hf_split only apply to --mode ground_truth; ignoring for selected mode.")
        use_hf = False
    if use_hf:
        _warn(f"loading HF dataset: {args.hf_dataset} split={args.hf_split}")
        hf_formal_by_name = _load_hf_formal_map(
            dataset_name=str(args.hf_dataset),
            split=str(args.hf_split),
            index_column=str(args.hf_index_column),
            formal_column=str(args.hf_formal_column),
        )
        _warn(f"HF loaded: {len(hf_formal_by_name)} formal statements")

    with out_path.open("w", encoding="utf-8") as out_f:
        for i, p in enumerate(problems):
            if args.max_problems and i >= int(args.max_problems):
                break
            problem_id = str(p.get("problem_id") or "").strip()
            if not problem_id:
                continue
            if allow is not None and problem_id not in allow:
                continue
            runs = p.get("runs") or {}
            if not isinstance(runs, dict):
                runs = {}
            input_dir = _resolve_manifest_path(
                run_root,
                p.get("input_dir"),
                fallback=run_root / "inputs" / problem_id,
            )
            if input_dir is None:
                num_missing_problem_config += 1
                continue
            cfg_path = input_dir / "problem_config.json"
            cfg = _read_json(cfg_path) or {}
            header_raw = str(cfg.get("header") or "").strip()
            header_curr = header_raw
            header_has_sorry = "sorry" in header_curr
            header_prover = header_curr
            if bool(args.require_header_no_sorry) and "sorry" in header_curr and str(args.mode) != "ground_truth":
                num_header_sorry_skipped += 1
                continue

            selected: List[SelectedStatement]
            run_dir: Optional[Path] = None
            run_dir_raw = runs.get(baseline)
            if str(args.mode) == "ground_truth":
                gt = str(cfg.get("ground_truth") or "").strip()
                header_curr = header_raw

                theo_name = _combibench_theorem_name(problem_id)
                formal_full = ""
                if str(args.solution_mode) == "with_solution" and use_hf:
                    formal_full = hf_formal_by_name.get(theo_name, "").strip()
                    if not formal_full:
                        _warn(
                            f"HF (with_solution) missing theorem_name={theo_name!r} for problem_id={problem_id!r}; "
                            "falling back to local problem_config.json"
                        )
                elif use_hf:
                    formal_full = hf_formal_by_name.get(theo_name, "").strip()
                    if not formal_full:
                        _warn(
                            f"HF missing theorem_name={theo_name!r} for problem_id={problem_id!r}; "
                            "falling back to local problem_config.json"
                        )

                if formal_full:
                    header_raw_hf, theorem_hf = _split_header_and_theorem(formal_full)
                    if theorem_hf:
                        gt = theorem_hf
                    if header_raw_hf:
                        header_curr = header_raw_hf
                        header_has_sorry = header_has_sorry or ("sorry" in header_raw_hf)

                if bool(args.require_header_no_sorry) and "sorry" in header_curr:
                    num_header_sorry_skipped += 1
                    continue

                header_prover = (
                    _sanitize_header_sorry_to_opaque(header_curr)
                    if bool(args.sanitize_header_sorry)
                    else header_curr
                )

                stmt = _force_by_sorry(gt)
                sig = _decl_signature(stmt)
                if stmt and sig:
                    selected = [
                        SelectedStatement(
                            program_id="ground_truth",
                            statement=stmt,
                            signature=sig,
                            combined_score=0.0,
                            public_metrics={"compile_ok": 1, "semantic_ok": 1},
                            code_len=len(stmt),
                            timestamp=0.0,
                        )
                    ]
                else:
                    selected = []
            else:
                run_dir = _resolve_run_dir(run_root, baseline, problem_id, run_dir_raw)
                if run_dir is None:
                    num_missing += 1
                    continue
                programs = _read_program_rows(run_dir)
                selected = _select_topk_unique(
                    programs,
                    k=int(args.k),
                    prefer_semantic=prefer_semantic,
                    require_semantic=require_semantic,
                )
                header_prover = (
                    _sanitize_header_sorry_to_opaque(header_curr)
                    if bool(args.sanitize_header_sorry)
                    else header_curr
                )

            num_problems += 1
            if not selected:
                num_empty += 1
            if len(selected) < int(args.k):
                num_under_k += 1

            idx_row: Dict[str, Any] = {
                "problem_id": problem_id,
                "mode": str(args.mode),
                "input_dir": str(input_dir),
                "run_dir": str(run_dir) if run_dir is not None else None,
                "header_has_sorry": bool(header_has_sorry),
                "hf_dataset": str(args.hf_dataset) if use_hf else None,
                "hf_split": str(args.hf_split) if use_hf else None,
                "solution_mode": str(args.solution_mode) if str(args.mode) == "ground_truth" else None,
                "selected": [],
            }

            for j, s in enumerate(selected):
                attempt_id = f"{problem_id}__{baseline}__k{j:02d}"
                full_lean4_code = _compose_lean_file(header=header_prover, statement=s.statement)
                rec = {
                    "problem_id": attempt_id,
                    # For Goedel's inference.py (local backend): expects a full Lean file.
                    "lean4_code": full_lean4_code,
                    # For our remote backend / analysis: keep statement-only + raw/sanitized headers.
                    "statement": s.statement,
                        "header": header_curr,
                        "header_prover": header_prover,
                    "group_problem_id": problem_id,
                    "source_baseline": baseline,
                    "rank": int(j),
                    "signature": s.signature,
                    "combined_score": float(s.combined_score),
                    "source_program_id": s.program_id,
                    "compile_ok": 1 if _is_ok(s.public_metrics, "compile_ok") else 0,
                    "semantic_ok": 1 if _is_ok(s.public_metrics, "semantic_ok") else 0,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                num_written += 1
                idx_row["selected"].append(rec)

            selected_index["problems"].append(idx_row)

    selected_index["stats"] = {
        "problems_processed": int(num_problems),
        "problems_missing_run_dir": int(num_missing),
        "problems_zero_selected": int(num_empty),
        "problems_under_k": int(num_under_k),
        "problems_skipped_header_sorry": int(num_header_sorry_skipped),
        "problems_missing_problem_config": int(num_missing_problem_config),
        "attempts_written": int(num_written),
        "prefer_semantic": bool(prefer_semantic),
        "require_semantic": bool(require_semantic),
        "sanitize_header_sorry": bool(args.sanitize_header_sorry),
        "mode": str(args.mode),
        "solution_mode": str(args.solution_mode),
    }

    if args.index_out:
        idx_out = Path(args.index_out).expanduser()
        idx_out.parent.mkdir(parents=True, exist_ok=True)
        idx_out.write_text(json.dumps(selected_index, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(selected_index["stats"], indent=2, ensure_ascii=False))
    print(f"Wrote dataset: {out_path}")
    if args.index_out:
        print(f"Wrote index: {Path(args.index_out).expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
