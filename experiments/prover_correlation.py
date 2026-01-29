                      
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_total_calls(run_dir: Path, *, fallback: int) -> int:
    term = _read_json(run_dir / "termination_log.json") or {}
    if str(term.get("no_llm", "")).strip().lower() in {"1", "true", "yes", "y", "on"}:
        return 0
    for key in ("total_budget_calls", "raw_llm_api_calls", "total_llm_calls"):
        if key not in term:
            continue
        try:
            v = int(term.get(key, 0) or 0)
        except Exception:
            continue
        return v if v > 0 else int(fallback)
    return int(fallback)


def _read_program_metrics(run_dir: Path) -> List[Dict[str, Any]]:
    db_path = run_dir / "evolution_db.sqlite"
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT public_metrics FROM programs")
        rows = cur.fetchall()
        conn.close()
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for (pm_raw,) in rows:
        if pm_raw is None:
            continue
        try:
            pm = json.loads(pm_raw) if isinstance(pm_raw, str) else dict(pm_raw)
        except Exception:
            continue
        if isinstance(pm, dict):
            out.append(pm)
    return out


def _count_ok(pms: List[Dict[str, Any]], key: str) -> int:
    n = 0
    for pm in pms:
        try:
            if int(pm.get(key, 0) or 0) == 1:
                n += 1
        except Exception:
            continue
    return n


@dataclass(frozen=True)
class DensityRow:
    compile_density: float
    semantic_density: float


def load_densities(run_root: Path, baseline: str) -> Dict[str, DensityRow]:
    manifest = _read_json(run_root / "manifest.json") or {}
    problems = manifest.get("problems") or []
    if not isinstance(problems, list):
        raise ValueError(f"Invalid manifest: {run_root / 'manifest.json'}")
    budget_calls = int(manifest.get("budget_calls", 0) or 0)
    out: Dict[str, DensityRow] = {}
    for p in problems:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("problem_id") or "").strip()
        runs = p.get("runs") or {}
        if not pid or not isinstance(runs, dict):
            continue
        run_dir_raw = runs.get(baseline)
        if not run_dir_raw:
            continue
        run_dir = Path(str(run_dir_raw)).expanduser().resolve()
        if not run_dir.exists():
            continue
        calls = _read_total_calls(run_dir, fallback=budget_calls)
        pms = _read_program_metrics(run_dir)
        c_ok = _count_ok(pms, "compile_ok")
        s_ok = _count_ok(pms, "semantic_ok")
        c_den = (float(c_ok) / float(calls)) if calls > 0 else 0.0
        s_den = (float(s_ok) / float(calls)) if calls > 0 else 0.0
        out[pid] = DensityRow(compile_density=float(c_den), semantic_density=float(s_den))
    return out


def pearson_r(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or not x:
        return 0.0
    n = len(x)
    mx = sum(x) / float(n)
    my = sum(y) / float(n)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0 or vy <= 0:
        return 0.0
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return float(cov) / math.sqrt(float(vx) * float(vy))


def perm_p_pearson(x: List[float], y01: List[int], *, n_perm: int, seed: int) -> float:
    rng = random.Random(int(seed))
    y = list(y01)
    obs = abs(pearson_r(x, [float(v) for v in y]))
    ge = 1
    tot = 1
    for _ in range(int(n_perm)):
        rng.shuffle(y)
        r = abs(pearson_r(x, [float(v) for v in y]))
        tot += 1
        if r >= obs - 1e-15:
            ge += 1
    return float(ge) / float(tot)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Analyze correlation between per-problem densities and prover success."
    )
    ap.add_argument("--prover_report", required=True, help="prover_eval.py output JSON")
    ap.add_argument("--run_root", required=True, help="Run root used for densities (contains manifest.json).")
    ap.add_argument("--baseline", default="", help="Baseline/method name inside run_root (default: infer if unique).")
    ap.add_argument("--metric", choices=["semantic_density", "compile_density"], default="semantic_density")
    ap.add_argument("--perm", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    report_path = Path(str(args.prover_report)).expanduser().resolve()
    rep = _read_json(report_path) or {}
    probs = rep.get("problems") or []
    if not isinstance(probs, list):
        raise SystemExit(f"Invalid prover_report: {report_path}")
    success: Dict[str, int] = {}
    for p in probs:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("problem_id") or "").strip()
        if not pid:
            continue
        success[pid] = 1 if bool(p.get("success")) else 0

    run_root = Path(str(args.run_root)).expanduser().resolve()
    manifest = _read_json(run_root / "manifest.json") or {}
    baselines = manifest.get("baselines")
    if isinstance(baselines, list) and baselines:
        baselines = [str(b) for b in baselines]
    else:
        runs_dir = run_root / "runs"
        baselines = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()]) if runs_dir.is_dir() else []
    baseline = str(args.baseline or "").strip()
    if not baseline:
        if len(baselines) != 1:
            raise SystemExit(f"Need --baseline (run_root has {baselines})")
        baseline = baselines[0]
    if baseline not in baselines:
        raise SystemExit(f"Unknown baseline {baseline!r} (run_root has {baselines})")

    dens = load_densities(run_root, baseline)
    common = sorted(set(success.keys()) & set(dens.keys()))
    if not common:
        raise SystemExit("No overlapping problem_ids between prover_report and run_root densities")

    y01 = [int(success[pid]) for pid in common]
    if str(args.metric) == "semantic_density":
        x = [float(dens[pid].semantic_density) for pid in common]
    else:
        x = [float(dens[pid].compile_density) for pid in common]

    r = pearson_r(x, [float(v) for v in y01])
    p = perm_p_pearson(x, y01, n_perm=int(args.perm), seed=int(args.seed))

    solved = sum(y01)
    n = len(common)
    x_solved = [xv for xv, yv in zip(x, y01) if yv == 1]
    x_unsolved = [xv for xv, yv in zip(x, y01) if yv == 0]
    mean_solved = (sum(x_solved) / float(len(x_solved))) if x_solved else 0.0
    mean_unsolved = (sum(x_unsolved) / float(len(x_unsolved))) if x_unsolved else 0.0

    print(f"[Correlation] metric={args.metric} baseline={baseline} n={n} solved={solved} rate={solved/float(n):.3f}")
    print(f"  mean(metric|solved)= {mean_solved:.5f}")
    print(f"  mean(metric|unsolved)= {mean_unsolved:.5f}")
    print(f"  pearson_r= {r:.4f}  perm_p= {p:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
