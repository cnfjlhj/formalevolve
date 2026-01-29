#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.dont_write_bytecode = True


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


_G_SUFFIX_PAT = re.compile(r"_g\d+$")


def _strip_g_suffix(name: str) -> str:
    return _G_SUFFIX_PAT.sub("", name or "")


def _parse_attempt_id(attempt_id: str) -> Tuple[str, Optional[str], Optional[int]]:
    base = _strip_g_suffix(attempt_id)
    parts = base.split("__")
    group_id = parts[0] if parts and parts[0] else base
    baseline = parts[1] if len(parts) >= 2 and parts[1] else None

    k_idx: Optional[int] = None
    for p in parts[2:]:
        if p.startswith("k") and p[1:].isdigit():
            k_idx = int(p[1:])
            break
    return group_id, baseline, k_idx


_THEOREM_START_RE = re.compile(r"(?m)^\s*(?:noncomputable\s+)?(?:theorem|lemma)\b")


def _theorem_or_lemma_blocks(code: str) -> List[str]:
    s = code or ""
    ms = list(_THEOREM_START_RE.finditer(s))
    if not ms:
        return []
    blocks: List[str] = []
    for i, m in enumerate(ms):
        start = int(m.start())
        end = int(ms[i + 1].start()) if i + 1 < len(ms) else len(s)
        blk = (s[start:end] or "").strip()
        if blk:
            blocks.append(blk)
    return blocks


def _has_theorem_or_lemma_block_without_sorry(code: str) -> bool:
    blocks = _theorem_or_lemma_blocks(code)
    if not blocks:
        return False
    for blk in blocks:
        if "sorry" not in blk:
            return True
    return False


@dataclass(frozen=True)
class AttemptResult:
    attempt_id: str
    group_id: str
    baseline: Optional[str]
    k_idx: Optional[int]
    ok: bool
    pass_ok: bool
    complete_ok: bool
    theorem_complete_ok: bool


def _load_compilation_results(path: Path, *, field: str) -> List[AttemptResult]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list: {path}")

    out: List[AttemptResult] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name") or r.get("problem_id") or "").strip()
        if not name:
            continue
        comp = r.get("compilation_result") or {}
        if not isinstance(comp, dict):
            comp = {}
        pass_ok = bool(comp.get("pass", False))
        complete_ok = bool(comp.get("complete", False))
        code = str(r.get("code") or "")
        theorem_complete_ok = bool(pass_ok and _has_theorem_or_lemma_block_without_sorry(code))
        ok = bool(comp.get(field, False))
        group_id, baseline, k_idx = _parse_attempt_id(name)
        out.append(
            AttemptResult(
                attempt_id=name,
                group_id=group_id,
                baseline=baseline,
                k_idx=k_idx,
                ok=ok,
                pass_ok=pass_ok,
                complete_ok=complete_ok,
                theorem_complete_ok=theorem_complete_ok,
            )
        )
    return out


def _load_input_index(input_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for line in input_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        pid = str(obj.get("problem_id") or "").strip()
        if not pid:
            continue
        out[pid] = obj
    return out


def _load_selection_index(index_json: Path) -> Dict[str, Any]:
    raw = json.loads(index_json.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid index JSON (expected object): {index_json}")
    problems = raw.get("problems")
    if problems is not None and not isinstance(problems, list):
        raise ValueError(f"Invalid index JSON (problems): {index_json}")
    return raw


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Summarize Goedel-Prover-V2 compilation results into statement-portfolio proof_hit@K.\n"
            "Grouping rule: group_id = attempt_id.split('__',1)[0] after stripping trailing _g\\d+.\n"
        )
    )
    ap.add_argument("--compilation_json", type=str, required=True, help="Path to code_compilation_repl*.json")
    ap.add_argument(
        "--field",
        type=str,
        default="complete",
        choices=["complete", "pass"],
        help="Success field inside compilation_result (default: complete).",
    )
    ap.add_argument(
        "--input_jsonl",
        type=str,
        default="",
        help="Optional dataset JSONL (from build_goedel_prover_dataset.py) to attach metadata.",
    )
    ap.add_argument(
        "--index_json",
        type=str,
        default="",
        help=(
            "Optional selection index JSON (from build_goedel_prover_dataset.py --index_out) to keep the "
            "denominator consistent when some problems have zero selected statements."
        ),
    )
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for summary.json and per_problem.json")
    args = ap.parse_args()

    comp_path = Path(args.compilation_json).expanduser().resolve()
    if not comp_path.exists():
        _warn(f"file not found: {comp_path}")
        return 2

    attempts = _load_compilation_results(comp_path, field=str(args.field))
    if not attempts:
        _warn(f"no attempts parsed from: {comp_path}")
        # If an index_json is provided, we can still produce a stable-denominator
        # summary treating all problems as unsolved with 0 attempts.
        if not args.index_json:
            return 2
        idx = Path(args.index_json).expanduser().resolve()
        if not idx.exists():
            _warn(f"--index_json not found: {idx}")
            return 2

        selection_index = _load_selection_index(idx)
        baseline = selection_index.get("baseline")
        baseline = str(baseline) if baseline is not None else None
        probs = selection_index.get("problems") or []
        expected: List[Tuple[str, Optional[str]]] = []
        for p in probs:
            if not isinstance(p, dict):
                continue
            pid = str(p.get("problem_id") or "").strip()
            if not pid:
                continue
            expected.append((pid, baseline))
        expected_groups = list(dict.fromkeys(expected))

        per_problem: List[Dict[str, Any]] = []
        for (group_id, base) in expected_groups:
            per_problem.append(
                {
                    "group_id": group_id,
                    "baseline": base,
                    "num_attempts": 0,
                    "solved": False,
                    "solved_pass": False,
                    "solved_complete": False,
                    "solved_theorem_complete": False,
                    "attempts": [],
                }
            )

        per_baseline: Dict[str, Dict[str, Any]] = {}
        by_base: Dict[Optional[str], List[Dict[str, Any]]] = defaultdict(list)
        for r in per_problem:
            by_base[r.get("baseline")].append(r)
        for base, rows in sorted(by_base.items(), key=lambda x: str(x[0] or "")):
            total = len(rows)
            per_baseline[str(base or "")] = {
                "problems": total,
                "solved": 0,
                "solved_ratio": 0.0,
                "solved_pass": 0,
                "solved_pass_ratio": 0.0,
                "solved_complete": 0,
                "solved_complete_ratio": 0.0,
                "solved_theorem_complete": 0,
                "solved_theorem_complete_ratio": 0.0,
            }

        summary = {
            "compilation_json": str(comp_path),
            "field": str(args.field),
            "denominator": {
                "mode": "index_json",
                "index_json": str(idx),
                "expected_problems": len(expected_groups),
                "baseline": baseline,
            },
            "overall": {
                "problems": len(per_problem),
                "solved": 0,
                "solved_ratio": 0.0,
                "solved_pass": 0,
                "solved_pass_ratio": 0.0,
                "solved_complete": 0,
                "solved_complete_ratio": 0.0,
                "solved_theorem_complete": 0,
                "solved_theorem_complete_ratio": 0.0,
            },
            "by_baseline": per_baseline,
        }

        out_dir = Path(args.out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "per_problem.json").write_text(
            json.dumps(per_problem, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Wrote: {out_dir / 'summary.json'}")
        print(f"Wrote: {out_dir / 'per_problem.json'}")
        return 0

    meta_by_attempt: Dict[str, Dict[str, Any]] = {}
    if args.input_jsonl:
        inp = Path(args.input_jsonl).expanduser().resolve()
        if inp.exists():
            meta_by_attempt = _load_input_index(inp)
        else:
            _warn(f"--input_jsonl not found: {inp}")

    grouped: Dict[Tuple[str, Optional[str]], List[AttemptResult]] = defaultdict(list)
    for a in attempts:
        grouped[(a.group_id, a.baseline)].append(a)

    expected_groups: Optional[List[Tuple[str, Optional[str]]]] = None
    selection_index: Optional[Dict[str, Any]] = None
    if args.index_json:
        idx = Path(args.index_json).expanduser().resolve()
        if not idx.exists():
            _warn(f"--index_json not found: {idx}")
        else:
            selection_index = _load_selection_index(idx)
            baseline = selection_index.get("baseline")
            baseline = str(baseline) if baseline is not None else None
            probs = selection_index.get("problems") or []
            expected: List[Tuple[str, Optional[str]]] = []
            for p in probs:
                if not isinstance(p, dict):
                    continue
                pid = str(p.get("problem_id") or "").strip()
                if not pid:
                    continue
                expected.append((pid, baseline))
            expected_groups = expected

    per_problem: List[Dict[str, Any]] = []
    keys: List[Tuple[str, Optional[str]]]
    if expected_groups is not None:
        keys = list(dict.fromkeys(expected_groups))
    else:
        keys = sorted(grouped.keys(), key=lambda x: (x[1] or "", x[0]))

    for (group_id, baseline) in keys:
        rows = grouped.get((group_id, baseline), [])
        rows_sorted = sorted(rows, key=lambda r: (r.k_idx if r.k_idx is not None else 1_000_000, r.attempt_id))
        solved = any(r.ok for r in rows_sorted)
        solved_pass = any(r.pass_ok for r in rows_sorted)
        solved_complete = any(r.complete_ok for r in rows_sorted)
        solved_theorem_complete = any(r.theorem_complete_ok for r in rows_sorted)
        per_problem.append(
            {
                "group_id": group_id,
                "baseline": baseline,
                "num_attempts": len(rows_sorted),
                "solved": bool(solved),
                "solved_pass": bool(solved_pass),
                "solved_complete": bool(solved_complete),
                "solved_theorem_complete": bool(solved_theorem_complete),
                "attempts": [
                    {
                        "attempt_id": r.attempt_id,
                        "ok": bool(r.ok),
                        "pass_ok": bool(r.pass_ok),
                        "complete_ok": bool(r.complete_ok),
                        "theorem_complete_ok": bool(r.theorem_complete_ok),
                        "k_idx": r.k_idx,
                        "meta": meta_by_attempt.get(_strip_g_suffix(r.attempt_id)),
                    }
                    for r in rows_sorted
                ],
            }
        )

    overall_total = len(per_problem)
    overall_solved = sum(1 for r in per_problem if r.get("solved"))
    overall_solved_pass = sum(1 for r in per_problem if r.get("solved_pass"))
    overall_solved_complete = sum(1 for r in per_problem if r.get("solved_complete"))
    overall_solved_theorem_complete = sum(1 for r in per_problem if r.get("solved_theorem_complete"))

    per_baseline: Dict[str, Dict[str, Any]] = {}
    by_base: Dict[Optional[str], List[Dict[str, Any]]] = defaultdict(list)
    for r in per_problem:
        by_base[r.get("baseline")].append(r)
    for base, rows in sorted(by_base.items(), key=lambda x: str(x[0] or "")):
        total = len(rows)
        solved = sum(1 for r in rows if r.get("solved"))
        solved_pass = sum(1 for r in rows if r.get("solved_pass"))
        solved_complete = sum(1 for r in rows if r.get("solved_complete"))
        solved_theorem_complete = sum(1 for r in rows if r.get("solved_theorem_complete"))
        per_baseline[str(base or "")] = {
            "problems": total,
            "solved": solved,
            "solved_ratio": (float(solved) / float(total)) if total > 0 else 0.0,
            "solved_pass": solved_pass,
            "solved_pass_ratio": (float(solved_pass) / float(total)) if total > 0 else 0.0,
            "solved_complete": solved_complete,
            "solved_complete_ratio": (float(solved_complete) / float(total)) if total > 0 else 0.0,
            "solved_theorem_complete": solved_theorem_complete,
            "solved_theorem_complete_ratio": (float(solved_theorem_complete) / float(total)) if total > 0 else 0.0,
        }

    summary = {
        "compilation_json": str(comp_path),
        "field": str(args.field),
        "denominator": (
            {
                "mode": "index_json",
                "index_json": str(Path(args.index_json).expanduser().resolve()),
                "expected_problems": len(expected_groups or []),
                "baseline": (selection_index or {}).get("baseline") if selection_index else None,
            }
            if expected_groups is not None and selection_index is not None
            else {"mode": "observed_only"}
        ),
        "overall": {
            "problems": overall_total,
            "solved": overall_solved,
            "solved_ratio": (float(overall_solved) / float(overall_total)) if overall_total > 0 else 0.0,
            "solved_pass": overall_solved_pass,
            "solved_pass_ratio": (float(overall_solved_pass) / float(overall_total)) if overall_total > 0 else 0.0,
            "solved_complete": overall_solved_complete,
            "solved_complete_ratio": (float(overall_solved_complete) / float(overall_total)) if overall_total > 0 else 0.0,
            "solved_theorem_complete": overall_solved_theorem_complete,
            "solved_theorem_complete_ratio": (float(overall_solved_theorem_complete) / float(overall_total))
            if overall_total > 0
            else 0.0,
        },
        "by_baseline": per_baseline,
    }

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "per_problem.json").write_text(
        json.dumps(per_problem, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote: {out_dir / 'summary.json'}")
    print(f"Wrote: {out_dir / 'per_problem.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
