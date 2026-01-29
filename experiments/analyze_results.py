#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from collections import Counter
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
class DiversitySig:
    good: int
    unique_sig: int
    dupmax_sig: int


@dataclass(frozen=True)
class TopSig:
    count: int
    sig: str


@dataclass(frozen=True)
class ProgramRow:
    code: str
    public_metrics: Dict[str, Any]
    combined_score: float
    metadata: Dict[str, Any]


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


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


def _diversity_unique_sig(codes: Iterable[str]) -> Tuple[DiversitySig, Optional[TopSig]]:
    sigs = [_decl_signature(c) for c in codes]
    sigs = [s for s in sigs if s]
    if not sigs:
        return DiversitySig(good=0, unique_sig=0, dupmax_sig=0), None
    cnt = Counter(sigs)
    top_sig, top_count = cnt.most_common(1)[0]
    return (
        DiversitySig(good=len(sigs), unique_sig=len(cnt), dupmax_sig=int(top_count)),
        TopSig(count=int(top_count), sig=str(top_sig)),
    )


def _read_total_calls(run_dir: Path, *, fallback: int = 0) -> int:
    term = _read_json(run_dir / "termination_log.json") or {}
    for key in ("total_budget_calls", "raw_llm_api_calls", "total_llm_calls"):
        if key not in term:
            continue
        try:
            return int(term.get(key, 0) or 0)
        except Exception:
            continue
    return int(fallback)


def _read_program_rows(run_dir: Path) -> List[Tuple[str, Dict[str, Any]]]:
    db_path = run_dir / "evolution_db.sqlite"
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT code, public_metrics, combined_score, metadata FROM programs")
        rows = cur.fetchall()
        conn.close()
    except Exception:
        return []

    out: List[ProgramRow] = []
    for code, pm_raw, combined_score, metadata_raw in rows:
        if pm_raw is None:
            continue
        try:
            pm = json.loads(pm_raw) if isinstance(pm_raw, str) else dict(pm_raw)
        except Exception:
            continue
        try:
            md = json.loads(metadata_raw) if isinstance(metadata_raw, str) else dict(metadata_raw or {})
        except Exception:
            md = {}
        try:
            cs = float(combined_score or 0.0)
        except Exception:
            cs = 0.0
        out.append(
            ProgramRow(
                code=str(code or ""),
                public_metrics=pm,
                combined_score=cs,
                metadata=md if isinstance(md, dict) else {},
            )
        )
    return out


def _is_ok(pm: Dict[str, Any], key: str) -> bool:
    try:
        return int(pm.get(key, 0) or 0) == 1
    except Exception:
        return False


def _fmt_ratio(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def _truncate(s: str, n: int = 140) -> str:
    s = (s or "").strip().replace("\n", " ")
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)].rstrip() + "…"


def _is_run_root(path: Path) -> bool:
    return path.is_dir() and (path / "manifest.json").exists() and (path / "runs").exists()


def find_run_roots(paths: List[Path]) -> List[Path]:
    roots: List[Path] = []
    seen: set[Path] = set()
    for p in paths:
        p = p.expanduser().resolve()
        if p.is_file():
            p = p.parent
        if _is_run_root(p) and p not in seen:
            roots.append(p)
            seen.add(p)
            continue
        if not p.exists():
            _warn(f"path not found: {p}")
            continue
        for m in p.rglob("manifest.json"):
            cand = m.parent
            if _is_run_root(cand) and cand not in seen:
                roots.append(cand)
                seen.add(cand)
    roots.sort(key=lambda x: str(x))
    return roots


def analyze_run_root(run_root: Path) -> Dict[str, Any]:
    manifest = _read_json(run_root / "manifest.json") or {}
    baselines = manifest.get("baselines")
    if not isinstance(baselines, list) or not baselines:
        # Fallback for non-baseline manifests (e.g., evolve-only runs) or legacy
        # manifests that didn't record `baselines`: infer from `runs/` folder names.
        runs_dir = run_root / "runs"
        if runs_dir.is_dir():
            baselines = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()])
        else:
            baselines = []
    if not baselines:
        raise ValueError(f"Invalid manifest (no baselines/runs found): {run_root / 'manifest.json'}")

    problems = manifest.get("problems") or []
    if not isinstance(problems, list):
        raise ValueError(f"Invalid manifest (problems): {run_root / 'manifest.json'}")

    num_problems = int(manifest.get("num_problems", len(problems)) or len(problems))
    budget_calls = int(manifest.get("budget_calls", 0) or 0)

    per_problem: List[Dict[str, Any]] = []
    case_table: List[Dict[str, Any]] = []
    baseline_agg: Dict[str, Dict[str, Any]] = {}

    for b in baselines:
        baseline_agg[str(b)] = {
            "compile": {
                "hit": 0,
                "total_ok": 0,
                "total_calls": 0,
                "density_sum": 0.0,
                "unique_sig_sum": 0,
                "dupmax_sig_max": 0,
            },
            "semantic": {
                "hit": 0,
                "total_ok": 0,
                "total_calls": 0,
                "density_sum": 0.0,
                "unique_sig_sum": 0,
                "dupmax_sig_max": 0,
            },
        }

    for p in problems:
        if not isinstance(p, dict):
            continue
        row: Dict[str, Any] = {
            "problem_id": p.get("problem_id"),
            "problem_name": p.get("problem_name"),
        }
        case_row: Dict[str, Any] = {
            "problem_id": p.get("problem_id"),
            "problem_name": p.get("problem_name"),
        }
        runs = p.get("runs") or {}
        if not isinstance(runs, dict):
            runs = {}

        for b in baselines:
            run_dir_raw = runs.get(b)
            run_dir = Path(str(run_dir_raw)) if run_dir_raw else None
            if not run_dir or not run_dir.exists():
                _warn(f"missing run_dir for baseline={b} problem={p.get('problem_id')}: {run_dir_raw}")
                calls = budget_calls
                programs_rows: List[ProgramRow] = []
            else:
                calls = _read_total_calls(run_dir, fallback=budget_calls)
                programs_rows = _read_program_rows(run_dir)

            programs = len(programs_rows)
            compile_codes = [r.code for r in programs_rows if _is_ok(r.public_metrics, "compile_ok")]
            semantic_codes = [r.code for r in programs_rows if _is_ok(r.public_metrics, "semantic_ok")]

            compile_ok = len(compile_codes)
            semantic_ok = len(semantic_codes)

            compile_density = (float(compile_ok) / float(calls)) if calls > 0 else 0.0
            semantic_density = (float(semantic_ok) / float(calls)) if calls > 0 else 0.0

            c_div, c_top = _diversity_unique_sig(compile_codes)
            s_div, s_top = _diversity_unique_sig(semantic_codes)

            row[f"{b}_calls"] = int(calls)
            row[f"{b}_programs"] = int(programs)

            row[f"{b}_compile_ok"] = int(compile_ok)
            row[f"{b}_compile_density"] = float(compile_density)
            row[f"{b}_compile_unique_sig"] = int(c_div.unique_sig)
            row[f"{b}_compile_dupmax_sig"] = int(c_div.dupmax_sig)
            row[f"{b}_compile_top_sig"] = {"count": c_top.count, "sig": c_top.sig} if c_top else None

            row[f"{b}_semantic_ok"] = int(semantic_ok)
            row[f"{b}_semantic_density"] = float(semantic_density)
            row[f"{b}_semantic_unique_sig"] = int(s_div.unique_sig)
            row[f"{b}_semantic_dupmax_sig"] = int(s_div.dupmax_sig)
            row[f"{b}_semantic_top_sig"] = {"count": s_top.count, "sig": s_top.sig} if s_top else None

            # Best program for per-problem case study. IMPORTANT:
            # - Includes semantic repair outputs, because they are recorded as
            #   separate programs in evolution_db.sqlite (same table).
            best_prog: Optional[ProgramRow] = None
            if programs_rows:
                best_prog = max(
                    programs_rows,
                    key=lambda r: (
                        float(r.combined_score),
                        int(_is_ok(r.public_metrics, "semantic_ok")),
                        int(_is_ok(r.public_metrics, "compile_ok")),
                        -len(_decl_signature(r.code) or ""),
                        -len(r.code or ""),
                    ),
                )
            if best_prog is None:
                case_row[str(b)] = None
            else:
                stmt = _extract_first_decl(_clean_code(best_prog.code))
                case_row[str(b)] = {
                    "combined_score": float(best_prog.combined_score),
                    "compile_ok": 1 if _is_ok(best_prog.public_metrics, "compile_ok") else 0,
                    "semantic_ok": 1 if _is_ok(best_prog.public_metrics, "semantic_ok") else 0,
                    "beq_ok": 1 if _is_ok(best_prog.public_metrics, "beq_ok") else 0,
                    "statement": stmt,
                    "signature": _decl_signature(best_prog.code),
                    "metadata": best_prog.metadata,
                }

            agg = baseline_agg[str(b)]
            agg["compile"]["total_ok"] += int(compile_ok)
            agg["compile"]["total_calls"] += int(calls)
            agg["compile"]["density_sum"] += float(compile_density)
            agg["compile"]["unique_sig_sum"] += int(c_div.unique_sig)
            agg["compile"]["dupmax_sig_max"] = max(int(agg["compile"]["dupmax_sig_max"]), int(c_div.dupmax_sig))
            agg["compile"]["hit"] += 1 if compile_ok > 0 else 0

            agg["semantic"]["total_ok"] += int(semantic_ok)
            agg["semantic"]["total_calls"] += int(calls)
            agg["semantic"]["density_sum"] += float(semantic_density)
            agg["semantic"]["unique_sig_sum"] += int(s_div.unique_sig)
            agg["semantic"]["dupmax_sig_max"] = max(int(agg["semantic"]["dupmax_sig_max"]), int(s_div.dupmax_sig))
            agg["semantic"]["hit"] += 1 if semantic_ok > 0 else 0

        per_problem.append(row)
        case_table.append(case_row)

    baseline_summary: Dict[str, Any] = {}
    for b in baselines:
        agg = baseline_agg[str(b)]
        baseline_summary[str(b)] = {}
        for metric in ["compile", "semantic"]:
            hit = int(agg[metric]["hit"])
            total_ok = int(agg[metric]["total_ok"])
            total_calls = int(agg[metric]["total_calls"])
            density_total = (float(total_ok) / float(total_calls)) if total_calls > 0 else 0.0
            density_mean = (float(agg[metric]["density_sum"]) / float(num_problems)) if num_problems > 0 else 0.0
            hit_rate = (float(hit) / float(num_problems)) if num_problems > 0 else 0.0
            unique_sig_sum = int(agg[metric]["unique_sig_sum"])
            unique_sig_density_total = (float(unique_sig_sum) / float(total_calls)) if total_calls > 0 else 0.0
            baseline_summary[str(b)][metric] = {
                "hit": hit,
                "hit_rate": hit_rate,
                "total_ok": total_ok,
                "total_calls": total_calls,
                "density_total": density_total,
                "density_mean": density_mean,
                "unique_sig_sum": unique_sig_sum,
                "unique_sig_density_total": unique_sig_density_total,
                "dupmax_sig_max": int(agg[metric]["dupmax_sig_max"]),
            }

        c_ok = baseline_summary[str(b)]["compile"]["total_ok"]
        s_ok = baseline_summary[str(b)]["semantic"]["total_ok"]
        baseline_summary[str(b)]["semantic_per_compile_total"] = (float(s_ok) / float(c_ok)) if c_ok > 0 else 0.0

    manifest_preview = {
        k: manifest.get(k)
        for k in [
            "created_at",
            "git_commit",
            "dataset_path",
            "num_problems",
            "budget_calls",
            "concurrency",
            "seed",
            "run",
            "baselines",
            "llm_mode",
            "openai_llm_base_url",
            "llm_models",
            "use_semantic",
            "use_cycle_consistency",
            "cycle_api_base_url",
            "cycle_model_name",
            "criticlean_url",
            "criticlean_model",
        ]
        if k in manifest
    }
    if not isinstance(manifest_preview.get("baselines"), list) or not manifest_preview.get("baselines"):
        manifest_preview["baselines"] = list(baselines)

    return {
        "run_root": str(run_root),
        "manifest": manifest_preview,
        "baseline_summary": baseline_summary,
        "per_problem": per_problem,
        "case_table": case_table,
    }


def _print_run_report(rep: Dict[str, Any], *, per_problem: bool, show_top: bool) -> None:
    run_root = rep.get("run_root", "")
    manifest = rep.get("manifest") or {}
    baselines = list((manifest.get("baselines") or []))
    num_problems = manifest.get("num_problems")
    budget_calls = manifest.get("budget_calls")
    created_at = manifest.get("created_at")

    print(f"\n== Run: {run_root} ==")
    print(f"created_at={created_at} problems={num_problems} budget_calls={budget_calls} baselines={baselines}")

    hdr = [
        "baseline",
        "c_hit",
        "c_density",
        "c_dupmax",
        "c_uniqSig",
        "s_hit",
        "s_density",
        "s_dupmax",
        "s_uniqSig",
        "s/c",
    ]

    rows: List[List[str]] = []
    for b in baselines:
        s = (rep.get("baseline_summary") or {}).get(b) or {}
        c = s.get("compile") or {}
        sem = s.get("semantic") or {}
        rows.append(
            [
                str(b),
                f"{c.get('hit', 0)}/{num_problems}",
                _fmt_ratio(float(c.get("density_total", 0.0))),
                str(c.get("dupmax_sig_max", 0)),
                str(c.get("unique_sig_sum", 0)),
                f"{sem.get('hit', 0)}/{num_problems}",
                _fmt_ratio(float(sem.get("density_total", 0.0))),
                str(sem.get("dupmax_sig_max", 0)),
                str(sem.get("unique_sig_sum", 0)),
                _fmt_ratio(float(s.get("semantic_per_compile_total", 0.0))),
            ]
        )

    col_w = [max(len(hdr[i]), *(len(r[i]) for r in rows)) for i in range(len(hdr))]
    print("  ".join(hdr[i].ljust(col_w[i]) for i in range(len(hdr))))
    print("  ".join("-" * col_w[i] for i in range(len(hdr))))
    for r in rows:
        print("  ".join(r[i].ljust(col_w[i]) for i in range(len(hdr))))

    if not per_problem:
        return

    per = rep.get("per_problem") or []
    print("\n-- Per-problem (semantic) --")
    for b in baselines:
        print(f"[{b}]")
        for p in per:
            name = p.get("problem_name")
            calls = int(p.get(f"{b}_calls", 0) or 0)
            s_ok = int(p.get(f"{b}_semantic_ok", 0) or 0)
            s_den = float(p.get(f"{b}_semantic_density", 0.0) or 0.0)
            s_uniq = int(p.get(f"{b}_semantic_unique_sig", 0) or 0)
            s_dup = int(p.get(f"{b}_semantic_dupmax_sig", 0) or 0)
            print(f"  {name}: ok={s_ok}/{calls} dens={_fmt_ratio(s_den)} uniqSig={s_uniq} dupmax={s_dup}")
            if show_top and s_dup > 0:
                top = p.get(f"{b}_semantic_top_sig")
                if isinstance(top, dict) and top.get("sig"):
                    print(f"    topSig×{top.get('count')}: {_truncate(str(top.get('sig')))}")


def write_report(outputs: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_report.json").write_text(
        json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Optional: per-problem best-case table for quick case study.
    # `case_table` lives under each run report entry.
    runs = outputs.get("runs")
    if isinstance(runs, list) and len(runs) == 1:
        case_table = (runs[0] or {}).get("case_table")
        if isinstance(case_table, list) and case_table:
            (out_dir / "case_table.json").write_text(
                json.dumps(case_table, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    rows: List[Dict[str, Any]] = []
    for rep in outputs.get("runs", []):
        manifest = rep.get("manifest") or {}
        run_root = rep.get("run_root")
        baselines = list(manifest.get("baselines") or [])
        for b in baselines:
            s = (rep.get("baseline_summary") or {}).get(b) or {}
            c = s.get("compile") or {}
            sem = s.get("semantic") or {}
            rows.append(
                {
                    "run_root": run_root,
                    "created_at": manifest.get("created_at"),
                    "num_problems": manifest.get("num_problems"),
                    "budget_calls": manifest.get("budget_calls"),
                    "baseline": b,
                    "compile_hit": c.get("hit"),
                    "compile_hit_rate": c.get("hit_rate"),
                    "compile_density_total": c.get("density_total"),
                    "compile_dupmax_sig_max": c.get("dupmax_sig_max"),
                    "compile_unique_sig_sum": c.get("unique_sig_sum"),
                    "semantic_hit": sem.get("hit"),
                    "semantic_hit_rate": sem.get("hit_rate"),
                    "semantic_density_total": sem.get("density_total"),
                    "semantic_dupmax_sig_max": sem.get("dupmax_sig_max"),
                    "semantic_unique_sig_sum": sem.get("unique_sig_sum"),
                    "semantic_per_compile_total": s.get("semantic_per_compile_total"),
                }
            )

    csv_path = out_dir / "analysis_report.csv"
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze autoformalization experiment runs (hit/density + diversity via unique_sig).\n"
            "Input can be a run directory (contains manifest.json) or an experiment directory."
        )
    )
    ap.add_argument("paths", nargs="+", help="Run root(s) or parent experiment dir(s).")
    ap.add_argument("--per_problem", action="store_true", help="Print per-problem semantic metrics.")
    ap.add_argument("--show_top", action="store_true", help="With --per_problem, show most frequent signature.")
    ap.add_argument("--write", type=str, default="", help="Write analysis_report.{json,csv} into this directory.")
    args = ap.parse_args()

    run_roots = find_run_roots([Path(p) for p in args.paths])
    if not run_roots:
        _warn("No run roots found (expected directories containing manifest.json + runs/)")
        return 2

    reports: List[Dict[str, Any]] = []
    for rr in run_roots:
        try:
            rep = analyze_run_root(rr)
        except Exception as e:
            _warn(f"Failed to analyze {rr}: {e}")
            continue
        reports.append(rep)
        _print_run_report(rep, per_problem=bool(args.per_problem), show_top=bool(args.show_top))

    outputs = {"runs": reports}
    if args.write:
        write_report(outputs, Path(args.write))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
