#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.dont_write_bytecode = True


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


def _group_problem_id(obj: Dict[str, Any]) -> str:
    gid = str(obj.get("group_problem_id") or "").strip()
    if gid:
        return gid
    pid = str(obj.get("problem_id") or "").strip()
    if not pid:
        return ""
    return pid.split("__", 1)[0]


def _source_baseline(obj: Dict[str, Any]) -> str:
    s = str(obj.get("source_baseline") or obj.get("baseline") or "").strip()
    if s:
        return s
    pid = str(obj.get("problem_id") or "").strip()
    parts = pid.split("__")
    if len(parts) >= 2 and parts[1].strip():
        return parts[1].strip()
    return ""


def _rank(obj: Dict[str, Any]) -> int:
    try:
        return int(obj.get("rank") if obj.get("rank") is not None else 1_000_000_000)
    except Exception:
        return 1_000_000_000


def _allocations_round_robin(*, budget: int, m: int) -> List[int]:
    if m <= 0:
        return []
    if budget <= 0:
        return [0 for _ in range(m)]
    base = budget // m
    rem = budget % m
    return [base + (1 if i < rem else 0) for i in range(m)]


def _format_s_idx(i: int) -> str:
    return f"s{i:02d}" if i < 100 else f"s{i}"


@dataclass(frozen=True)
class ProblemInfo:
    problem_id: str
    baseline: str
    statements: int
    budget: int
    allocations: List[int]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Expand a per-problem selected statement set into a fixed-budget round-robin schedule.\n"
            "Writes `n` per statement so that per-problem total completions = budget."
        )
    )
    ap.add_argument("--input_jsonl", required=True, help="Selected statements JSONL (one row per statement).")
    ap.add_argument("--base_index_json", required=True, help="Index JSON produced alongside input_jsonl.")
    ap.add_argument("--budget", type=int, required=True, help="Total prover attempts per problem (sum over statements).")
    ap.add_argument("--tag", type=str, required=True, help="Tag to encode into statement ids (e.g. rr64).")
    ap.add_argument(
        "--max_statements",
        type=int,
        default=0,
        help="If >0, cap statements per problem to this number (default: 0 = no cap).",
    )
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument("--index_out", required=True, help="Output index JSON path.")
    args = ap.parse_args()

    input_jsonl = Path(str(args.input_jsonl)).expanduser().resolve()
    base_index_json = Path(str(args.base_index_json)).expanduser().resolve()
    out_path = Path(str(args.out)).expanduser().resolve()
    index_out = Path(str(args.index_out)).expanduser().resolve()

    if not input_jsonl.exists():
        _warn(f"input_jsonl not found: {input_jsonl}")
        return 2
    if not base_index_json.exists():
        _warn(f"base_index_json not found: {base_index_json}")
        return 2

    try:
        base_index = json.loads(base_index_json.read_text(encoding="utf-8"))
    except Exception as e:
        _warn(f"failed to parse base_index_json: {type(e).__name__}: {e}")
        return 2
    if not isinstance(base_index, dict):
        _warn(f"expected a JSON object: {base_index_json}")
        return 2

    tag = str(args.tag or "").strip()
    if not tag:
        _warn("empty --tag")
        return 2

    budget = int(args.budget)
    if budget <= 0:
        _warn("--budget must be > 0")
        return 2

    max_statements = int(args.max_statements)
    if max_statements < 0:
        _warn("--max_statements must be >= 0")
        return 2

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    baseline_guess: Optional[str] = None
    for obj in _read_jsonl(input_jsonl):
        gid = _group_problem_id(obj)
        if not gid:
            continue
        grouped.setdefault(gid, []).append(obj)
        if gid not in order:
            order.append(gid)
        if baseline_guess is None:
            b = _source_baseline(obj)
            baseline_guess = b if b else None

    if not grouped:
        _warn("no valid rows in input_jsonl")
        return 2

    baseline = str(base_index.get("baseline") or baseline_guess or "").strip()
    if not baseline:
        _warn("failed to infer baseline")
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    index_out.parent.mkdir(parents=True, exist_ok=True)

    problems_index: List[ProblemInfo] = []
    out_lines: List[str] = []

    for gid in order:
        items = list(grouped.get(gid) or [])
        items.sort(key=lambda x: (_rank(x), str(x.get("problem_id") or "")))
        if max_statements > 0 and len(items) > max_statements:
            items = items[:max_statements]

        allocs = _allocations_round_robin(budget=budget, m=len(items))
        kept: List[Tuple[int, Dict[str, Any]]] = [(i, it) for i, it in enumerate(items) if allocs[i] > 0]
        if not kept:
            problems_index.append(ProblemInfo(problem_id=gid, baseline=baseline, statements=0, budget=budget, allocations=[]))
            continue

        problems_index.append(
            ProblemInfo(
                problem_id=gid,
                baseline=baseline,
                statements=len(kept),
                budget=budget,
                allocations=[allocs[i] for i, _ in kept],
            )
        )

        for s_idx, (i, it) in enumerate(kept):
            origin_id = f"{gid}__{baseline}__{tag}__{_format_s_idx(s_idx)}"
            row = dict(it)
            row["problem_id"] = origin_id
            row["origin_problem_id"] = origin_id
            row["n"] = int(allocs[i])
            row["rr_budget"] = int(budget)
            row["rr_statements_total"] = int(len(kept))
            row["rr_statement_index"] = int(s_idx)
            out_lines.append(json.dumps(row, ensure_ascii=False))

    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    index_obj = {
        "input_jsonl": str(input_jsonl),
        "base_index_json": str(base_index_json),
        "budget": int(budget),
        "tag": tag,
        "baseline": baseline,
        "problems": [
            {
                "problem_id": p.problem_id,
                "baseline": p.baseline,
                "statements": int(p.statements),
                "budget": int(p.budget),
                "allocations": list(p.allocations),
            }
            for p in problems_index
        ],
    }
    index_out.write_text(json.dumps(index_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote: {out_path} ({len(out_lines)} rows)")
    print(f"Wrote: {index_out} ({len(problems_index)} problems)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

