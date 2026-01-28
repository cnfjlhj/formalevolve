#!/usr/bin/env python3
"""
Paper-aligned summary metrics from per-problem runs.

This script is intended for ICML-style "code supplementary" releases:
- It consumes the outputs produced by `run_dataset_pilot.py`
  (i.e., a directory containing `runs/<problem_id>/evolution_db.sqlite`).
- It computes the core paper metrics at the *run budget* T (the run's stop point):
  - CH@T: fraction of problems with >=1 compile-ok candidate
  - SH@T: fraction of problems with >=1 semantic-ok candidate
  - SemOK_total: total number of *deduplicated* semantic-ok candidates across problems
  - Uniformity: Gini coefficient + Top-10% share over per-problem SemOK counts

Deduplication is implemented via a canonicalized statement string derived from
the stored program code:
- Extract the first theorem/lemma (fallback: first declaration)
- Normalize the top-level declaration name to a constant (default: my_theorem)

This matches the "deduplicated under canonicalization" wording used for
cross-problem concentration metrics in the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _detect_runs_dir(run_root: Path) -> Path:
    candidates = [
        run_root / "runs" / "ours",  # legacy wrapper layout
        run_root / "runs",  # run_dataset_pilot layout
    ]
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            if any(p.is_dir() for p in cand.iterdir()):
                return cand
        except Exception:
            continue
    raise FileNotFoundError(
        "Missing runs dir. Expected one of: " + ", ".join(str(p) for p in candidates)
    )


def _gini(values: Sequence[int], eps: float = 1e-12) -> float:
    """Gini coefficient for non-negative values."""
    n = int(len(values))
    if n <= 0:
        return 0.0
    total = float(sum(int(v) for v in values))
    if total <= 0:
        return 0.0
    xs = sorted(float(v) for v in values)
    weighted_sum = 0.0
    for i, v in enumerate(xs, start=1):
        weighted_sum += float(i) * float(v)
    # Equivalent to the pairwise-absolute-difference definition.
    return (2.0 * weighted_sum) / (float(n) * (total + eps)) - (float(n) + 1.0) / float(n)


def _topk_share(values: Sequence[int], frac: float = 0.10) -> float:
    n = int(len(values))
    if n <= 0:
        return 0.0
    total = float(sum(int(v) for v in values))
    if total <= 0:
        return 0.0
    k = max(1, int(math.ceil(float(frac) * float(n))))
    topk = sorted((int(v) for v in values), reverse=True)[:k]
    return float(sum(topk)) / total


def _import_statement_utils() -> Tuple[Any, Any]:
    """Import statement normalization helpers from the evaluator."""
    here = Path(__file__).resolve()
    # examples/icml2026_autoformalization/scripts -> examples/icml2026_autoformalization
    root = here.parents[1]
    sys.path.insert(0, str(root))
    from evaluate import normalize_lean_statement, normalize_decl_name_for_cycle_prompt  # type: ignore

    return normalize_lean_statement, normalize_decl_name_for_cycle_prompt


def _canonicalize_statement(
    code: str,
    *,
    normalize_lean_statement: Any,
    normalize_decl_name_for_cycle_prompt: Any,
    canonical_decl_name: str,
) -> str:
    stmt = normalize_lean_statement(code or "")
    if not stmt:
        return ""
    stmt2, _orig = normalize_decl_name_for_cycle_prompt(stmt, normalized_name=canonical_decl_name)
    # Whitespace normalization for stable dedup keys.
    return "\n".join(ln.rstrip() for ln in (stmt2 or "").splitlines()).strip()


@dataclass(frozen=True)
class PerProblemRow:
    problem_id: str
    compile_hit: int
    semantic_hit: int
    semok_dedup_count: int
    db_programs_total: int
    term_total_budget_calls: Optional[int]
    term_total_llm_calls: Optional[int]


def _iter_programs(conn: sqlite3.Connection) -> Iterable[Tuple[str, Dict[str, Any]]]:
    cur = conn.execute("SELECT code, public_metrics FROM programs")
    for row in cur.fetchall():
        code = str(row["code"] or "")
        public_raw = row["public_metrics"]
        try:
            public = json.loads(public_raw) if public_raw else {}
        except Exception:
            public = {}
        if not isinstance(public, dict):
            public = {}
        yield code, public


def _summarize_run_root(run_root: Path, canonical_decl_name: str) -> Tuple[List[PerProblemRow], Dict[str, Any]]:
    normalize_lean_statement, normalize_decl_name_for_cycle_prompt = _import_statement_utils()

    runs_dir = _detect_runs_dir(run_root)
    manifest_path = run_root / "manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}

    rows: List[PerProblemRow] = []
    missing_db: List[str] = []
    missing_term: List[str] = []

    for problem_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        pid = problem_dir.name
        db_path = problem_dir / "evolution_db.sqlite"
        term_path = problem_dir / "termination_log.json"
        if not db_path.exists():
            missing_db.append(pid)
            continue
        if not term_path.exists():
            missing_term.append(pid)
            term = {}
        else:
            term = _load_json(term_path)

        compile_hit = 0
        semok_set: set[str] = set()
        total_programs = 0

        with _open_db(db_path) as conn:
            for code, public in _iter_programs(conn):
                total_programs += 1
                c_ok = int(public.get("compile_ok", 0) or 0)
                s_ok = int(public.get("semantic_ok", 0) or 0)
                if c_ok == 1:
                    compile_hit = 1
                if c_ok == 1 and s_ok == 1:
                    key = _canonicalize_statement(
                        code,
                        normalize_lean_statement=normalize_lean_statement,
                        normalize_decl_name_for_cycle_prompt=normalize_decl_name_for_cycle_prompt,
                        canonical_decl_name=canonical_decl_name,
                    )
                    if key:
                        semok_set.add(key)

        semantic_hit = 1 if semok_set else 0
        rows.append(
            PerProblemRow(
                problem_id=pid,
                compile_hit=int(compile_hit),
                semantic_hit=int(semantic_hit),
                semok_dedup_count=int(len(semok_set)),
                db_programs_total=int(total_programs),
                term_total_budget_calls=_safe_int(term.get("total_budget_calls")),
                term_total_llm_calls=_safe_int(term.get("total_llm_calls")),
            )
        )

    semok_counts = [int(r.semok_dedup_count) for r in rows]
    compile_hits = [int(r.compile_hit) for r in rows]
    semantic_hits = [int(r.semantic_hit) for r in rows]

    n = int(len(rows))
    ch = float(sum(compile_hits)) / float(n) if n > 0 else 0.0
    sh = float(sum(semantic_hits)) / float(n) if n > 0 else 0.0
    semok_total = int(sum(semok_counts))

    def _safe_stats_int(vals: List[int]) -> Dict[str, Any]:
        if not vals:
            return {"count": 0}
        vals_sorted = sorted(int(v) for v in vals)
        return {
            "count": len(vals_sorted),
            "min": vals_sorted[0],
            "max": vals_sorted[-1],
            "mean": float(sum(vals_sorted)) / float(len(vals_sorted)),
            "median": float(statistics.median(vals_sorted)),
        }

    meta = {
        "run_root": str(run_root),
        "runs_dir": str(runs_dir),
        "canonical_decl_name": canonical_decl_name,
        "manifest_preview": {
            "dataset_path": manifest.get("dataset_path"),
            "num_problems": manifest.get("num_problems"),
            "baseline_mode": manifest.get("baseline_mode"),
            "seed": manifest.get("seed"),
            "max_llm_calls": manifest.get("max_llm_calls"),
            "num_generations": manifest.get("num_generations"),
            "llm_mode": manifest.get("llm_mode"),
            "llm_models": manifest.get("llm_models"),
            "patch_llm_models": manifest.get("patch_llm_models"),
            "use_semantic": manifest.get("use_semantic"),
            "use_cycle_consistency": manifest.get("use_cycle_consistency"),
        },
        "missing_db": missing_db,
        "missing_termination_log": missing_term,
        "num_problems_present": n,
        "CH_at_T": ch,
        "SH_at_T": sh,
        "SemOK_total": semok_total,
        "gini_semok": float(_gini(semok_counts)),
        "top10_share_semok": float(_topk_share(semok_counts, frac=0.10)),
        "per_problem_semok_stats": _safe_stats_int(semok_counts),
        "note": (
            "SemOK counts are deduplicated by canonicalized statement extracted from the stored code. "
            "If semantic judging was disabled during the run, SH@T and SemOK_total will be 0."
        ),
    }
    return rows, meta


def _write_csv(path: Path, rows: List[PerProblemRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "problem_id",
                "compile_hit",
                "semantic_hit",
                "semok_dedup_count",
                "db_programs_total",
                "term_total_budget_calls",
                "term_total_llm_calls",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "problem_id": r.problem_id,
                    "compile_hit": r.compile_hit,
                    "semantic_hit": r.semantic_hit,
                    "semok_dedup_count": r.semok_dedup_count,
                    "db_programs_total": r.db_programs_total,
                    "term_total_budget_calls": "" if r.term_total_budget_calls is None else r.term_total_budget_calls,
                    "term_total_llm_calls": "" if r.term_total_llm_calls is None else r.term_total_llm_calls,
                }
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_root",
        type=str,
        required=True,
        help="A pilot output dir containing `runs/<problem_id>/evolution_db.sqlite` (or `runs/ours/<problem_id>/...`).",
    )
    ap.add_argument(
        "--canonical_decl_name",
        type=str,
        default="my_theorem",
        help="Canonical declaration name used for deduplication (default: my_theorem).",
    )
    ap.add_argument(
        "--out_json",
        type=str,
        default="",
        help="Optional output JSON path (default: <run_root>/paper_summary.json).",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="",
        help="Optional output CSV path (default: <run_root>/paper_per_problem.csv).",
    )
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    rows, meta = _summarize_run_root(run_root, canonical_decl_name=str(args.canonical_decl_name))

    out_json = Path(args.out_json).resolve() if args.out_json else (run_root / "paper_summary.json")
    out_csv = Path(args.out_csv).resolve() if args.out_csv else (run_root / "paper_per_problem.csv")

    out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_csv, rows)

    # Also print a concise summary (stdout) for quick CLI usage.
    n = int(meta.get("num_problems_present", 0) or 0)
    print(f"[paper_summary] run_root={run_root}")
    print(f"[paper_summary] problems={n} CH@T={meta.get('CH_at_T'):.3f} SH@T={meta.get('SH_at_T'):.3f}")
    print(
        f"[paper_summary] SemOK_total={meta.get('SemOK_total')} "
        f"Gini={meta.get('gini_semok'):.4f} Top10%={meta.get('top10_share_semok'):.4f}"
    )
    print(f"[paper_summary] wrote: {out_json}")
    print(f"[paper_summary] wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

