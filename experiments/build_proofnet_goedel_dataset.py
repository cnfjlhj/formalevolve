                      
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.dont_write_bytecode = True


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _compose_lean_file(*, header: str, statement: str) -> str:
    h = str(header or "").strip()
    s = str(statement or "").strip()
    if h:
        return h.rstrip() + "\n\n" + s
    return s


def _normalize_statement_to_by_sorry(stmt: str) -> str:
    s = str(stmt or "").strip()
    if not s:
        return ""
    if ":=" in s:
        head = s.split(":=", 1)[0].rstrip()
        return head + " := by sorry"
                                                                   
    if s.endswith("sorry"):
        s = s[: -len("sorry")].rstrip()
    return s.rstrip() + " := by sorry"


@dataclass(frozen=True)
class Problem:
    problem_id: str
    header: str
    statement: str


def _problem_id_from_row(row: Dict[str, Any]) -> str:
                                                                                 
                                                                    
    for k in ("full_name", "problem_name", "name", "id"):
        v = str(row.get(k) or "").strip()
        if v:
            return v
    return ""


def _load_problems(path: Path, *, max_problems: int) -> List[Problem]:
    raw = _read_jsonl(path)
    if max_problems > 0:
        raw = raw[:max_problems]
    out: List[Problem] = []
    for r in raw:
        pid = _problem_id_from_row(r)
        header = str(r.get("header") or "").strip()
        stmt = str(r.get("formal_stmt") or "").strip()
        stmt = _normalize_statement_to_by_sorry(stmt)
        if not pid or not stmt:
            continue
        out.append(Problem(problem_id=pid, header=header, statement=stmt))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build a Goedel remote-inference input JSONL from ProofNet jsonl.\n"
            "This exports statement-only Lean problems (theorem := by sorry), suitable for\n"
            "experiments/goedel_remote_inference.py (output_mode=merge) + lean_server_compile.py.\n"
        )
    )
    ap.add_argument("--input_jsonl", required=True, help="Path to benchmark/proofnet_lean4.15.0.jsonl")
    ap.add_argument("--out", required=True, help="Output dataset JSONL path.")
    ap.add_argument("--index_out", default="", help="Optional selection index JSON output path.")
    ap.add_argument("--baseline", default="ground_truth", help="Baseline name used in attempt ids (default: ground_truth).")
    ap.add_argument("--k", type=int, default=1, help="Statements per problem (default: 1).")
    ap.add_argument("--max_problems", type=int, default=0, help="If >0, only export the first N problems.")
    args = ap.parse_args()

    in_path = Path(str(args.input_jsonl)).expanduser().resolve()
    if not in_path.exists():
        _warn(f"Input not found: {in_path}")
        return 2

    out_path = Path(str(args.out)).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    baseline = str(args.baseline or "").strip() or "ground_truth"
    k = max(1, int(args.k))
    problems = _load_problems(in_path, max_problems=int(args.max_problems))
    if not problems:
        _warn("No problems parsed from input.")
        return 2

    selected_index: Dict[str, Any] = {
        "baseline": baseline,
        "source": str(in_path),
        "mode": "benchmark_proofnet",
        "problems": [],
        "stats": {},
    }

    num_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for p in problems:
            idx_row: Dict[str, Any] = {
                "problem_id": p.problem_id,
                "mode": "benchmark_proofnet",
                "input_dir": None,
                "run_dir": None,
                "header_has_sorry": ("sorry" in p.header),
                "selected": [],
            }
            for j in range(k):
                attempt_id = f"{p.problem_id}__{baseline}__k{j:02d}"
                full_lean4_code = _compose_lean_file(header=p.header, statement=p.statement)
                rec: Dict[str, Any] = {
                    "problem_id": attempt_id,
                    "lean4_code": full_lean4_code,
                    "statement": p.statement,
                    "header": p.header,
                    "header_prover": p.header,
                    "group_problem_id": p.problem_id,
                    "source_baseline": baseline,
                    "rank": int(j),
                    "signature": "",
                    "combined_score": 0.0,
                    "source_program_id": "",
                    "compile_ok": 1,
                    "semantic_ok": 1,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                num_written += 1
                idx_row["selected"].append(rec)
            selected_index["problems"].append(idx_row)

    selected_index["stats"] = {
        "problems_processed": int(len(problems)),
        "attempts_written": int(num_written),
        "k": int(k),
        "max_problems": int(args.max_problems),
    }

    if str(args.index_out or "").strip():
        idx_path = Path(str(args.index_out)).expanduser().resolve()
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        idx_path.write_text(json.dumps(selected_index, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote index: {idx_path}")

    print(json.dumps(selected_index["stats"], indent=2, ensure_ascii=False))
    print(f"Wrote dataset: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
