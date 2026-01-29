#!/usr/bin/env python3
"""
Strict evaluation for an evolution run.

Why "strict":
- Computes success metrics from `evolution_db.sqlite` (source of truth),
  instead of relying on the optional `best/` export folder.
- Adds per-problem *budget accounting* from `termination_log.json`, where
  `total_llm_calls = raw_llm_api_calls + seedbank_debited_calls` (when enabled).

Outputs (written under --run_root):
- strict_metrics_per_problem.csv
- strict_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ProblemStrictRow:
    problem_id: str
    best_program_id: str
    best_generation: int
    best_island_idx: Optional[int]
    best_combined_score: float
    compile_hit_db: int
    semantic_hit_db: int
    best_export_dir_exists: int
    best_export_error_matching_seed: int
    total_budget_calls: Optional[int]
    total_llm_calls: Optional[int]
    raw_llm_api_calls: Optional[int]
    seedbank_debited_calls: Optional[int]
    total_evals: Optional[int]
    total_effective_evals: Optional[int]
    elapsed_time_seconds: Optional[float]
    generations_completed: Optional[int]


def _safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _has_best_export_error(err_path: Path) -> bool:
    if not err_path.exists():
        return False
    try:
        s = err_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    return "Could not find matching seed for best program" in s


def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _query_one(conn: sqlite3.Connection, sql: str, args: Tuple[Any, ...] = ()) -> Any:
    cur = conn.execute(sql, args)
    row = cur.fetchone()
    return None if row is None else row[0]


def _query_best_program(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    # Prefer non-null combined_score; break ties deterministically by timestamp then id.
    cur = conn.execute(
        """
        SELECT id, generation, island_idx, combined_score, timestamp
        FROM programs
        WHERE combined_score IS NOT NULL
        ORDER BY combined_score DESC, timestamp ASC, id ASC
        LIMIT 1
        """
    )
    row = cur.fetchone()
    return None if row is None else row


def _bool_hit(conn: sqlite3.Connection, json_key: str) -> int:
    sql = f"""
        SELECT 1
        FROM programs
        WHERE json_extract(public_metrics, '$.{json_key}') = 1
        LIMIT 1
    """
    return 1 if _query_one(conn, sql) is not None else 0


def _max_combined_score(conn: sqlite3.Connection) -> float:
    val = _query_one(conn, "SELECT MAX(combined_score) FROM programs")
    return float(val or 0.0)


def _numeric_summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    values_sorted = sorted(values)
    mean = sum(values_sorted) / len(values_sorted)
    med = statistics.median(values_sorted)
    return {
        "count": len(values_sorted),
        "min": values_sorted[0],
        "max": values_sorted[-1],
        "mean": mean,
        "median": med,
    }


def _collect_rows(run_root: Path) -> Tuple[List[ProblemStrictRow], Dict[str, Any]]:
    # Support both layouts:
    # - New/pilot layout (run_dataset_pilot.py):   <run_root>/runs/<problem_id>/
    # - Legacy layout (some trend wrappers):       <run_root>/runs/ours/<problem_id>/
    runs_dir_candidates = [
        run_root / "runs" / "ours",
        run_root / "runs",
    ]
    runs_dir: Optional[Path] = None
    for cand in runs_dir_candidates:
        if not cand.exists():
            continue
        try:
            if any(p.is_dir() for p in cand.iterdir()):
                runs_dir = cand
                break
        except Exception:
            continue
    if runs_dir is None:
        raise FileNotFoundError(
            "Missing runs dir. Expected one of: "
            + ", ".join(str(p) for p in runs_dir_candidates)
        )

    missing_db: List[str] = []
    missing_termination_log: List[str] = []
    missing_best_dir: List[str] = []
    best_export_error_matching_seed: List[str] = []

    rows: List[ProblemStrictRow] = []

    for problem_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        pid = problem_dir.name
        db_path = problem_dir / "evolution_db.sqlite"
        term_path = problem_dir / "termination_log.json"
        best_dir = problem_dir / "best"
        err_path = problem_dir / "run_evo.err"

        if not db_path.exists():
            missing_db.append(pid)
            continue
        if not term_path.exists():
            missing_termination_log.append(pid)
            continue

        if not best_dir.exists():
            missing_best_dir.append(pid)
        has_export_err = _has_best_export_error(err_path)
        if has_export_err:
            best_export_error_matching_seed.append(pid)

        term = _load_json(term_path)
        with _open_db(db_path) as conn:
            best_row = _query_best_program(conn)
            if best_row is None:
                best_id = ""
                best_generation = -1
                best_island_idx_int = None
            else:
                best_id = str(best_row["id"])
                best_generation = int(best_row["generation"])
                best_island_idx = best_row["island_idx"]
                best_island_idx_int = None if best_island_idx is None else int(best_island_idx)

            rows.append(
                ProblemStrictRow(
                    problem_id=pid,
                    best_program_id=best_id,
                    best_generation=best_generation,
                    best_island_idx=best_island_idx_int,
                    best_combined_score=_max_combined_score(conn),
                    compile_hit_db=_bool_hit(conn, "compile_ok"),
                    semantic_hit_db=_bool_hit(conn, "semantic_ok"),
                    best_export_dir_exists=1 if best_dir.exists() else 0,
                    best_export_error_matching_seed=1 if has_export_err else 0,
                    total_budget_calls=_safe_int(term.get("total_budget_calls")),
                    total_llm_calls=_safe_int(term.get("total_llm_calls")),
                    raw_llm_api_calls=_safe_int(term.get("raw_llm_api_calls")),
                    seedbank_debited_calls=_safe_int(term.get("seedbank_debited_calls")),
                    total_evals=_safe_int(term.get("total_evals")),
                    total_effective_evals=_safe_int(term.get("total_effective_evals")),
                    elapsed_time_seconds=_safe_float(term.get("elapsed_time_seconds")),
                    generations_completed=_safe_int(term.get("generations_completed")),
                )
            )

    meta = {
        "missing_db": missing_db,
        "missing_termination_log": missing_termination_log,
        "missing_best_dir": missing_best_dir,
        "best_export_error_matching_seed": best_export_error_matching_seed,
    }
    return rows, meta


def _write_csv(out_path: Path, rows: List[ProblemStrictRow]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "problem_id",
        "best_combined_score",
        "compile_hit_db",
        "semantic_hit_db",
        "best_generation",
        "best_island_idx",
        "best_program_id",
        "best_export_dir_exists",
        "best_export_error_matching_seed",
        "total_budget_calls",
        "total_llm_calls",
        "raw_llm_api_calls",
        "seedbank_debited_calls",
        "total_evals",
        "total_effective_evals",
        "elapsed_time_seconds",
        "generations_completed",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "problem_id": r.problem_id,
                    "best_combined_score": r.best_combined_score,
                    "compile_hit_db": r.compile_hit_db,
                    "semantic_hit_db": r.semantic_hit_db,
                    "best_generation": r.best_generation,
                    "best_island_idx": r.best_island_idx,
                    "best_program_id": r.best_program_id,
                    "best_export_dir_exists": r.best_export_dir_exists,
                    "best_export_error_matching_seed": r.best_export_error_matching_seed,
                    "total_budget_calls": r.total_budget_calls,
                    "total_llm_calls": r.total_llm_calls,
                    "raw_llm_api_calls": r.raw_llm_api_calls,
                    "seedbank_debited_calls": r.seedbank_debited_calls,
                    "total_evals": r.total_evals,
                    "total_effective_evals": r.total_effective_evals,
                    "elapsed_time_seconds": r.elapsed_time_seconds,
                    "generations_completed": r.generations_completed,
                }
            )


def _write_summary(out_path: Path, run_root: Path, rows: List[ProblemStrictRow], meta: Dict[str, Any]) -> None:
    best_score_counts: Dict[str, int] = {}
    for r in rows:
        key = str(int(r.best_combined_score))
        best_score_counts[key] = best_score_counts.get(key, 0) + 1

    def _vals_int(field: str) -> List[float]:
        out: List[float] = []
        for r in rows:
            v = getattr(r, field)
            if v is None:
                continue
            out.append(float(v))
        return out

    def _vals_float(field: str) -> List[float]:
        out: List[float] = []
        for r in rows:
            v = getattr(r, field)
            if v is None:
                continue
            out.append(float(v))
        return out

    summary = {
        "run_dir": str(run_root.resolve()),
        "num_problems": len(rows),
        "rows_written": len(rows),
        "missing_db": meta.get("missing_db", []),
        "missing_termination_log": meta.get("missing_termination_log", []),
        "missing_best_dir": meta.get("missing_best_dir", []),
        "best_export_error_matching_seed": meta.get("best_export_error_matching_seed", []),
        "compile_hit_db": sum(r.compile_hit_db for r in rows),
        "semantic_hit_db": sum(r.semantic_hit_db for r in rows),
        "best_score_db_counts": best_score_counts,
        "budget_strict": {
            "total_budget_calls": _numeric_summary(_vals_int("total_budget_calls")),
            "total_llm_calls": _numeric_summary(_vals_int("total_llm_calls")),
            "raw_llm_api_calls": _numeric_summary(_vals_int("raw_llm_api_calls")),
            "seedbank_debited_calls": _numeric_summary(_vals_int("seedbank_debited_calls")),
            "total_effective_evals": _numeric_summary(_vals_int("total_effective_evals")),
            "elapsed_time_seconds": _numeric_summary(_vals_float("elapsed_time_seconds")),
            "generations_completed": _numeric_summary(_vals_int("generations_completed")),
        },
        "note": (
            "Strict metrics computed from evolution_db.sqlite (programs.public_metrics + combined_score) "
            "and budget from termination_log.json (seedbank debits included in total_llm_calls). "
            "best/ is treated as optional export."
        ),
    }

    out_path.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", type=str, required=True)
    args = ap.parse_args()

    run_root = Path(args.run_root).expanduser()
    if not run_root.exists():
        raise FileNotFoundError(run_root)

    rows, meta = _collect_rows(run_root)

    # Keep stable ordering in CSV.
    rows_sorted = sorted(rows, key=lambda r: r.problem_id)

    _write_csv(run_root / "strict_metrics_per_problem.csv", rows_sorted)
    _write_summary(run_root / "strict_summary.json", run_root, rows_sorted, meta)


if __name__ == "__main__":
    main()
