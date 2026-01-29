#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.dont_write_bytecode = True


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


_G_SUFFIX_RE = re.compile(r"_g\d+$")


def _strip_g_suffix(attempt_id: str) -> str:
    return _G_SUFFIX_RE.sub("", attempt_id or "")


def _parse_group_id(attempt_id: str) -> str:
    base = _strip_g_suffix(attempt_id)
    parts = base.split("__")
    return parts[0] if parts and parts[0] else base


_LEAN_BLOCK_COMMENT_RE = re.compile(r"/-[\s\S]*?-/", re.MULTILINE)
_LEAN_LINE_COMMENT_RE = re.compile(r"(?m)^\s*--.*$")
_SORRY_TOKEN_RE = re.compile(r"\bsorry\b")


def _remove_comments_lean(text: str) -> str:
    out = _LEAN_BLOCK_COMMENT_RE.sub("", text or "")
    out = _LEAN_LINE_COMMENT_RE.sub("", out)
    return out


def _formal_statement_chunks(formal_statement: str) -> List[str]:
    tmpl = (_remove_comments_lean(formal_statement) or "").strip()
    if not tmpl:
        return []

    parts: List[str] = []
    for chunk in tmpl.split("\n\n"):
        if not chunk.strip():
            continue
        # Match CombiBench's splitting behavior around theorem boundaries.
        chunk = chunk.replace("\nnoncomputable theorem", "\n\nnoncomputable theorem")
        chunk = chunk.replace("\ntheorem", "\n\ntheorem")
        chunk = chunk.replace("\nlemma", "\n\nlemma")
        parts.append(chunk)

    filtered: List[str] = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        if s.startswith(("import", "set_option", "open")):
            continue
        s2 = _SORRY_TOKEN_RE.sub("", s).strip()
        if s2:
            filtered.append(s2)
    return filtered


_THEOREM_OR_LEMMA_NAME_RE = re.compile(r"(?m)^\s*(?:noncomputable\s+)?(?:theorem|lemma)\s+([A-Za-z0-9_']+)\b")


def _extract_theorem_name(formal_statement: str) -> Optional[str]:
    m = _THEOREM_OR_LEMMA_NAME_RE.search(formal_statement or "")
    if not m:
        return None
    return m.group(1)


def _contains_all_chunks(*, code: str, chunks: List[str]) -> bool:
    if not chunks:
        return False
    out = _remove_comments_lean(code or "").strip()
    if not out:
        return False
    return all(c in out for c in chunks)


def _target_theorem_has_sorry(*, code: str, theorem_name: str) -> bool:
    if not theorem_name:
        return True
    out = code or ""
    m = re.search(
        rf"(?m)^\s*(?:noncomputable\s+)?(?:theorem|lemma)\s+{re.escape(theorem_name)}\b",
        out,
    )
    if not m:
        return True
    return "sorry" in (out[m.start() :] or "")


@dataclass(frozen=True)
class Attempt:
    group_id: str
    pass_ok: bool
    complete_ok: bool
    stmt_ok: bool
    target_theorem_no_sorry: bool


def _load_selection_index(path: Path) -> Tuple[Dict[str, Any], Dict[str, List[List[str]]], Dict[str, Optional[str]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid selection_index (expected object): {path}")

    chunks_by_problem: Dict[str, List[List[str]]] = {}
    theorem_by_problem: Dict[str, Optional[str]] = {}

    for p in raw.get("problems") or []:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("problem_id") or "").strip()
        if not pid:
            continue
        selected = p.get("selected") or []
        statements: List[str] = []
        if isinstance(selected, list):
            for s in selected:
                if not isinstance(s, dict):
                    continue
                st = s.get("statement")
                if isinstance(st, str) and st.strip():
                    statements.append(st)

        chunks_list = [_formal_statement_chunks(st) for st in statements]
        chunks_list = [c for c in chunks_list if c]
        chunks_by_problem[pid] = chunks_list

        thm = None
        for st in statements:
            thm = _extract_theorem_name(st)
            if thm:
                break
        theorem_by_problem[pid] = thm

    return raw, chunks_by_problem, theorem_by_problem


def _load_attempts(
    *, compilation_json: Path, chunks_by_problem: Dict[str, List[List[str]]], theorem_by_problem: Dict[str, Optional[str]]
) -> List[Attempt]:
    raw = json.loads(compilation_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Invalid compilation JSON (expected list): {compilation_json}")

    out: List[Attempt] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name") or r.get("problem_id") or "").strip()
        if not name:
            continue
        gid = _parse_group_id(name)
        comp = r.get("compilation_result") or {}
        if not isinstance(comp, dict):
            comp = {}

        pass_ok = bool(comp.get("pass", False))
        complete_ok = bool(comp.get("complete", False))
        code = str(r.get("code") or "")

        stmt_ok = False
        for chunks in chunks_by_problem.get(gid, []):
            if _contains_all_chunks(code=code, chunks=chunks):
                stmt_ok = True
                break

        thm = theorem_by_problem.get(gid) or ""
        target_theorem_no_sorry = bool(stmt_ok and pass_ok and (not _target_theorem_has_sorry(code=code, theorem_name=thm)))

        out.append(
            Attempt(
                group_id=gid,
                pass_ok=pass_ok,
                complete_ok=complete_ok,
                stmt_ok=stmt_ok,
                target_theorem_no_sorry=target_theorem_no_sorry,
            )
        )
    return out


def _summarize_attempts(attempts: List[Attempt], *, expected_problems: Iterable[str]) -> Dict[str, Any]:
    by_problem: Dict[str, List[Attempt]] = {}
    for a in attempts:
        by_problem.setdefault(a.group_id, []).append(a)

    rows: List[Dict[str, Any]] = []
    max_attempts = 0
    for pid in expected_problems:
        pid = str(pid)
        arr = by_problem.get(pid, [])
        max_attempts = max(max_attempts, len(arr))
        rows.append(
            {
                "problem_id": pid,
                "attempts": len(arr),
                "pass@16": any(x.pass_ok for x in arr),
                "complete@16": any(x.complete_ok for x in arr),
                "strict_stmt_pass@16": any(x.pass_ok and x.stmt_ok for x in arr),
                "strict_stmt_complete@16": any(x.complete_ok and x.stmt_ok for x in arr),
                "strict_stmt_theorem_complete@16": any(x.target_theorem_no_sorry for x in arr),
            }
        )

    def _count(key: str) -> int:
        return sum(1 for r in rows if r.get(key))

    total = len(rows)
    summary = {
        "problems": total,
        "attempts_total": len(attempts),
        "attempts_per_problem_max": max_attempts,
        "pass@16": _count("pass@16"),
        "complete@16": _count("complete@16"),
        "strict_stmt_pass@16": _count("strict_stmt_pass@16"),
        "strict_stmt_complete@16": _count("strict_stmt_complete@16"),
        "strict_stmt_theorem_complete@16": _count("strict_stmt_theorem_complete@16"),
    }
    for k in list(summary.keys()):
        if k.endswith("@16"):
            summary[k + "_ratio"] = (float(summary[k]) / float(total)) if total else 0.0

    return {"summary": summary, "per_problem": rows}


def _preset_runs() -> List[Tuple[str, Path]]:
    # Relative to examples/autoformalization_v1/experiments/
    return [
        (
            "proofnet50_ground_truth",
            Path("_analysis_tmp/proving_proofnet_fullname_noearly_20260108_143957/proofnet_k1_b16"),
        ),
        (
            "combibench100_gt_with_solution",
            Path("_analysis_tmp/proving_overnight_20260107_224027/gt_with_solution_k1_b16/k01_n16_b16"),
        ),
        (
            "combibench100_gt_without_solution",
            Path("_analysis_tmp/proving_overnight_20260107_224027/gt_without_solution_k1_b16/k01_n16_b16"),
        ),
    ]


def _parse_run_arg(s: str) -> Tuple[str, Path]:
    if "=" not in s:
        raise ValueError("Expected --run label=path")
    label, path = s.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError("Invalid --run label=path (empty label/path)")
    return label, Path(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize ProofNet/CombiBench proving runs into a CSV/JSON table.")
    ap.add_argument(
        "--preset",
        type=str,
        default="",
        choices=["", "ground_truth"],
        help="Built-in run set (use --preset ground_truth for this repo's existing runs).",
    )
    ap.add_argument("--run", action="append", default=[], help="Run spec: label=path (repeatable).")
    ap.add_argument(
        "--out_csv",
        type=str,
        default="",
        help="Output CSV path (default: experiments/_analysis_tmp/proving_ground_truth_summary_<ts>.csv).",
    )
    ap.add_argument(
        "--out_json",
        type=str,
        default="",
        help="Output JSON path (default: experiments/_analysis_tmp/proving_ground_truth_summary_<ts>.json).",
    )
    ap.add_argument(
        "--out_simple_csv",
        type=str,
        default="",
        help=(
            "Optional simplified CSV path with canonical columns for reporting.\n"
            "- ProofNet: pass@16, complete@16, strict_stmt_complete@16 (no-sorry theorem); strict_anscheck_complete@16 is NA\n"
            "- CombiBench: pass@16, complete@16, strict_stmt_complete@16 (=theorem_complete), "
            "and (for without_solution only) strict_anscheck_complete@16.\n"
            "Default: experiments/_analysis_tmp/proving_ground_truth_summary_<ts>_simple.csv."
        ),
    )
    args = ap.parse_args()

    runs: List[Tuple[str, Path]] = []
    if args.preset == "ground_truth":
        runs.extend(_preset_runs())
    for r in args.run:
        runs.append(_parse_run_arg(r))
    if not runs:
        _warn("No runs specified (use --preset ground_truth or --run label=path).")
        return 2

    script_dir = Path(__file__).resolve().parent
    out_dir_default = script_dir / "_analysis_tmp"
    out_dir_default.mkdir(parents=True, exist_ok=True)
    stamp = _now_stamp()
    out_csv = Path(args.out_csv).expanduser() if args.out_csv else (out_dir_default / f"proving_ground_truth_summary_{stamp}.csv")
    out_json = Path(args.out_json).expanduser() if args.out_json else (out_dir_default / f"proving_ground_truth_summary_{stamp}.json")
    out_simple_csv = (
        Path(args.out_simple_csv).expanduser()
        if args.out_simple_csv
        else (out_dir_default / f"proving_ground_truth_summary_{stamp}_simple.csv")
    )

    all_runs_out: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    simple_rows: List[Dict[str, Any]] = []

    for label, rel_path in runs:
        run_root = (script_dir / rel_path).resolve() if not rel_path.is_absolute() else rel_path.resolve()
        if not run_root.exists():
            _warn(f"missing run root: {label} -> {run_root}")
            continue

        sel_path = run_root / "selection_index.json"
        comp_path = run_root / "goedel_out" / "code_compilation_repl_full16.json"
        comp_ans_path = run_root / "goedel_out" / "code_compilation_repl_full16_anscheck.json"

        if not sel_path.exists():
            _warn(f"{label}: missing selection_index.json: {sel_path}")
            continue
        if not comp_path.exists():
            _warn(f"{label}: missing compilation JSON: {comp_path}")
            continue

        selection_meta, chunks_by_problem, theorem_by_problem = _load_selection_index(sel_path)
        expected = [str(p.get("problem_id")) for p in (selection_meta.get("problems") or []) if isinstance(p, dict) and p.get("problem_id")]

        attempts = _load_attempts(compilation_json=comp_path, chunks_by_problem=chunks_by_problem, theorem_by_problem=theorem_by_problem)
        base = _summarize_attempts(attempts, expected_problems=expected)

        ans = None
        if comp_ans_path.exists():
            attempts_ans = _load_attempts(
                compilation_json=comp_ans_path, chunks_by_problem=chunks_by_problem, theorem_by_problem=theorem_by_problem
            )
            ans = _summarize_attempts(attempts_ans, expected_problems=expected)

        run_out = {
            "label": label,
            "root": str(run_root),
            "selection_index": str(sel_path),
            "compilation_json": str(comp_path),
            "compilation_json_anscheck": str(comp_ans_path) if comp_ans_path.exists() else None,
            "selection_meta": {
                "baseline": selection_meta.get("baseline"),
                "solution_mode": selection_meta.get("solution_mode"),
                "source": selection_meta.get("source"),
                "mode": selection_meta.get("mode"),
            },
            "base": base,
            "anscheck": ans,
        }
        all_runs_out.append(run_out)

        mode = str((selection_meta.get("mode") or "")).strip()
        solution_mode = str((selection_meta.get("solution_mode") or "")).strip()
        dataset = "proofnet" if mode == "benchmark_proofnet" else "combibench"

        row = {
            "label": label,
            "problems": base["summary"]["problems"],
            "attempts_total": base["summary"]["attempts_total"],
            "attempts_per_problem_max": base["summary"]["attempts_per_problem_max"],
            "pass@16": base["summary"]["pass@16"],
            "pass@16_ratio": base["summary"]["pass@16_ratio"],
            "complete@16": base["summary"]["complete@16"],
            "complete@16_ratio": base["summary"]["complete@16_ratio"],
            "strict_stmt_pass@16": base["summary"]["strict_stmt_pass@16"],
            "strict_stmt_pass@16_ratio": base["summary"]["strict_stmt_pass@16_ratio"],
            "strict_stmt_complete@16": base["summary"]["strict_stmt_complete@16"],
            "strict_stmt_complete@16_ratio": base["summary"]["strict_stmt_complete@16_ratio"],
            "strict_stmt_theorem_complete@16": base["summary"]["strict_stmt_theorem_complete@16"],
            "strict_stmt_theorem_complete@16_ratio": base["summary"]["strict_stmt_theorem_complete@16_ratio"],
            "anscheck_pass@16": "" if ans is None else ans["summary"]["pass@16"],
            "anscheck_pass@16_ratio": "" if ans is None else ans["summary"]["pass@16_ratio"],
            "anscheck_complete@16": "" if ans is None else ans["summary"]["complete@16"],
            "anscheck_complete@16_ratio": "" if ans is None else ans["summary"]["complete@16_ratio"],
            "anscheck_strict_stmt_pass@16": "" if ans is None else ans["summary"]["strict_stmt_pass@16"],
            "anscheck_strict_stmt_pass@16_ratio": "" if ans is None else ans["summary"]["strict_stmt_pass@16_ratio"],
            "anscheck_strict_stmt_complete@16": "" if ans is None else ans["summary"]["strict_stmt_complete@16"],
            "anscheck_strict_stmt_complete@16_ratio": "" if ans is None else ans["summary"]["strict_stmt_complete@16_ratio"],
        }
        csv_rows.append(row)

        # NOTE: This matches the user's bookkeeping:
        # - complete@16 := compilation_result.complete (no statement-alignment requirement)
        # - strict_complete@16 := no-sorry theorem (same as solved_theorem_complete)
        strict_stmt_complete = base["summary"]["strict_stmt_theorem_complete@16"]
        strict_anscheck_complete = (
            ans["summary"]["strict_stmt_theorem_complete@16"]
            if (dataset == "combibench" and solution_mode == "without_solution" and ans is not None)
            else "NA"
        )

        simple_rows.append(
            {
                "label": label,
                "dataset": dataset,
                "solution_mode": solution_mode,
                "problems": base["summary"]["problems"],
                "pass@16": base["summary"]["pass@16"],
                "complete@16": base["summary"]["complete@16"],
                "strict_stmt_complete@16": strict_stmt_complete,
                "strict_anscheck_complete@16": strict_anscheck_complete,
            }
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"), "runs": all_runs_out}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if csv_rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        if simple_rows:
            out_simple_csv.parent.mkdir(parents=True, exist_ok=True)
            with out_simple_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(simple_rows[0].keys()))
                w.writeheader()
                w.writerows(simple_rows)
    else:
        _warn("No rows written (no valid runs found).")

    print(f"Wrote JSON: {out_json}")
    if csv_rows:
        print(f"Wrote CSV:  {out_csv}")
        if simple_rows:
            print(f"Wrote CSV:  {out_simple_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
