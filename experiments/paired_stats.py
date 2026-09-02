#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


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


def _count_ok(program_metrics: Iterable[Dict[str, Any]], key: str) -> int:
    n = 0
    for pm in program_metrics:
        try:
            if int(pm.get(key, 0) or 0) == 1:
                n += 1
        except Exception:
            continue
    return n


def _infer_baselines(manifest: Dict[str, Any], run_root: Path) -> List[str]:
    baselines = manifest.get("baselines")
    if isinstance(baselines, list) and baselines:
        return [str(b) for b in baselines]
    runs_dir = run_root / "runs"
    if runs_dir.is_dir():
        return sorted([p.name for p in runs_dir.iterdir() if p.is_dir()])
    return []


@dataclass(frozen=True)
class ProblemMetrics:
    calls: int
    compile_ok: int
    semantic_ok: int

    @property
    def compile_hit(self) -> int:
        return 1 if self.compile_ok > 0 else 0

    @property
    def semantic_hit(self) -> int:
        return 1 if self.semantic_ok > 0 else 0

    @property
    def compile_density(self) -> float:
        return (float(self.compile_ok) / float(self.calls)) if self.calls > 0 else 0.0

    @property
    def semantic_density(self) -> float:
        return (float(self.semantic_ok) / float(self.calls)) if self.calls > 0 else 0.0


@dataclass
class RunData:
    run_root: Path
    baseline: str
    dataset_path: str
    num_problems: int
    budget_calls: int
    seed: Optional[int]
    per_problem: Dict[str, ProblemMetrics]

    def totals(self) -> Dict[str, Any]:
        probs = list(self.per_problem.values())
        total_calls = sum(p.calls for p in probs)
        c_ok = sum(p.compile_ok for p in probs)
        s_ok = sum(p.semantic_ok for p in probs)
        c_hit = sum(p.compile_hit for p in probs)
        s_hit = sum(p.semantic_hit for p in probs)
        n = len(probs)
        return {
            "n": n,
            "compile_hit": c_hit,
            "compile_hit_rate": (float(c_hit) / float(n)) if n > 0 else 0.0,
            "compile_density_total": (float(c_ok) / float(total_calls)) if total_calls > 0 else 0.0,
            "semantic_hit": s_hit,
            "semantic_hit_rate": (float(s_hit) / float(n)) if n > 0 else 0.0,
            "semantic_density_total": (float(s_ok) / float(total_calls)) if total_calls > 0 else 0.0,
            "semantic_per_compile_total": (float(s_ok) / float(c_ok)) if c_ok > 0 else 0.0,
            "total_calls": total_calls,
            "total_compile_ok": c_ok,
            "total_semantic_ok": s_ok,
        }


def _parse_run_spec(spec: str) -> Tuple[Path, Optional[str]]:
    raw = str(spec or "").strip()
    if not raw:
        raise ValueError("Empty run spec")
    if "::" in raw:
        root_s, baseline = raw.split("::", 1)
        return Path(root_s).expanduser().resolve(), (baseline.strip() or None)
    return Path(raw).expanduser().resolve(), None


def load_run(spec: str) -> RunData:
    run_root, baseline = _parse_run_spec(spec)
    manifest_path = run_root / "manifest.json"
    manifest = _read_json(manifest_path) or {}
    baselines = _infer_baselines(manifest, run_root)
    if baseline is None:
        if len(baselines) != 1:
            raise ValueError(
                f"run_root has {len(baselines)} baselines; please pass baseline explicitly: {run_root}::BASELINE"
            )
        baseline = baselines[0]
    if baseline not in baselines:
        raise ValueError(f"Unknown baseline {baseline!r} for run_root={run_root} (have {baselines})")

    problems = manifest.get("problems") or []
    if not isinstance(problems, list):
        raise ValueError(f"Invalid manifest problems: {manifest_path}")
    num_problems = int(manifest.get("num_problems", len(problems)) or len(problems))
    budget_calls = int(manifest.get("budget_calls", 0) or 0)
    dataset_path = str(manifest.get("dataset_path", "") or "")
    seed = None
    try:
        seed = int(manifest.get("seed")) if manifest.get("seed") is not None else None
    except Exception:
        seed = None

    per_problem: Dict[str, ProblemMetrics] = {}
    for p in problems:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("problem_id") or "").strip()
        runs = p.get("runs") or {}
        if not pid or not isinstance(runs, dict):
            continue
        run_dir_raw = runs.get(baseline)
        if not run_dir_raw:
            _warn(f"Missing run dir: baseline={baseline} problem_id={pid}")
            continue
        run_dir = Path(str(run_dir_raw)).expanduser().resolve()
        if not run_dir.exists():
            _warn(f"Run dir not found: baseline={baseline} problem_id={pid} run_dir={run_dir}")
            continue

        calls = _read_total_calls(run_dir, fallback=budget_calls if budget_calls > 0 else 0)
        program_metrics = _read_program_metrics(run_dir)
        c_ok = _count_ok(program_metrics, "compile_ok")
        s_ok = _count_ok(program_metrics, "semantic_ok")
        per_problem[pid] = ProblemMetrics(calls=int(calls), compile_ok=int(c_ok), semantic_ok=int(s_ok))

    return RunData(
        run_root=run_root,
        baseline=str(baseline),
        dataset_path=dataset_path,
        num_problems=int(num_problems),
        budget_calls=int(budget_calls),
        seed=seed,
        per_problem=per_problem,
    )


def _binom_cdf(k: int, n: int, p: float) -> float:
    if n <= 0:
        return 1.0
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    s = 0.0
    for i in range(0, k + 1):
        s += math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i))
    return float(s)


def mcnemar_exact_p(a_hits: List[int], b_hits: List[int]) -> Tuple[int, int, float]:
    if len(a_hits) != len(b_hits):
        raise ValueError("mcnemar: length mismatch")
    n10 = 0  # a=1, b=0
    n01 = 0  # a=0, b=1
    for a, b in zip(a_hits, b_hits):
        if a == 1 and b == 0:
            n10 += 1
        elif a == 0 and b == 1:
            n01 += 1
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    p = 2.0 * _binom_cdf(k, n, 0.5)
    return n10, n01, float(min(1.0, p))


def bootstrap_ci_mean_diff(
    diffs: List[float], *, n_boot: int = 10_000, seed: int = 0
) -> Tuple[float, float]:
    if not diffs:
        return 0.0, 0.0
    rng = random.Random(int(seed))
    n = len(diffs)
    means: List[float] = []
    for _ in range(int(n_boot)):
        s = 0.0
        for _j in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / float(n))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means))]
    return float(lo), float(hi)


def paired_permutation_p_mean(
    diffs: List[float], *, n_perm: int = 20_000, seed: int = 0
) -> float:
    if not diffs:
        return 1.0
    rng = random.Random(int(seed))
    n = len(diffs)
    obs = sum(diffs) / float(n)
    obs_abs = abs(obs)
    ge = 1  # add-one smoothing
    tot = 1
    for _ in range(int(n_perm)):
        s = 0.0
        for d in diffs:
            s += d if (rng.getrandbits(1) == 1) else -d
        m = s / float(n)
        tot += 1
        if abs(m) >= obs_abs - 1e-15:
            ge += 1
    return float(ge) / float(tot)


def _fmt(x: float, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    except Exception:
        return "0"


def compare_runs(
    a: RunData,
    b: RunData,
    *,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> Dict[str, Any]:
    common = sorted(set(a.per_problem.keys()) & set(b.per_problem.keys()))
    if not common:
        raise ValueError("No overlapping problems between runs")

    a_c_hit = [a.per_problem[pid].compile_hit for pid in common]
    b_c_hit = [b.per_problem[pid].compile_hit for pid in common]
    a_s_hit = [a.per_problem[pid].semantic_hit for pid in common]
    b_s_hit = [b.per_problem[pid].semantic_hit for pid in common]

    a_c_den = [a.per_problem[pid].compile_density for pid in common]
    b_c_den = [b.per_problem[pid].compile_density for pid in common]
    a_s_den = [a.per_problem[pid].semantic_density for pid in common]
    b_s_den = [b.per_problem[pid].semantic_density for pid in common]

    diffs = {
        "compile_hit": [float(x - y) for x, y in zip(a_c_hit, b_c_hit)],
        "semantic_hit": [float(x - y) for x, y in zip(a_s_hit, b_s_hit)],
        "compile_density": [float(x - y) for x, y in zip(a_c_den, b_c_den)],
        "semantic_density": [float(x - y) for x, y in zip(a_s_den, b_s_den)],
    }

    # Hit: McNemar exact.
    c_n10, c_n01, c_p = mcnemar_exact_p(a_c_hit, b_c_hit)
    s_n10, s_n01, s_p = mcnemar_exact_p(a_s_hit, b_s_hit)

    # Density: paired permutation + bootstrap CI for mean diff.
    c_den_mean = sum(diffs["compile_density"]) / float(len(common))
    s_den_mean = sum(diffs["semantic_density"]) / float(len(common))
    c_den_ci = bootstrap_ci_mean_diff(diffs["compile_density"], n_boot=n_boot, seed=seed)
    s_den_ci = bootstrap_ci_mean_diff(diffs["semantic_density"], n_boot=n_boot, seed=seed + 1)
    c_den_p = paired_permutation_p_mean(diffs["compile_density"], n_perm=n_perm, seed=seed)
    s_den_p = paired_permutation_p_mean(diffs["semantic_density"], n_perm=n_perm, seed=seed + 1)

    return {
        "n_common": len(common),
        "problems": common,
        "hit": {
            "compile": {
                "a_rate": sum(a_c_hit) / float(len(common)),
                "b_rate": sum(b_c_hit) / float(len(common)),
                "diff_rate": (sum(a_c_hit) - sum(b_c_hit)) / float(len(common)),
                "n10_a1_b0": c_n10,
                "n01_a0_b1": c_n01,
                "mcnemar_p": c_p,
            },
            "semantic": {
                "a_rate": sum(a_s_hit) / float(len(common)),
                "b_rate": sum(b_s_hit) / float(len(common)),
                "diff_rate": (sum(a_s_hit) - sum(b_s_hit)) / float(len(common)),
                "n10_a1_b0": s_n10,
                "n01_a0_b1": s_n01,
                "mcnemar_p": s_p,
            },
        },
        "density": {
            "compile": {
                "a_mean": sum(a_c_den) / float(len(common)),
                "b_mean": sum(b_c_den) / float(len(common)),
                "diff_mean": c_den_mean,
                "diff_mean_ci": {"lo": c_den_ci[0], "hi": c_den_ci[1]},
                "perm_p": c_den_p,
            },
            "semantic": {
                "a_mean": sum(a_s_den) / float(len(common)),
                "b_mean": sum(b_s_den) / float(len(common)),
                "diff_mean": s_den_mean,
                "diff_mean_ci": {"lo": s_den_ci[0], "hi": s_den_ci[1]},
                "perm_p": s_den_p,
            },
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Paired comparison for autoformalization runs (hit/density).\n"
            "Run spec format: RUN_ROOT or RUN_ROOT::BASELINE.\n"
            "Example: experiments/combibench.../run_xxx::ours"
        )
    )
    ap.add_argument("--a", required=True, help="Reference run spec (e.g., ours).")
    ap.add_argument(
        "--b",
        action="append",
        default=[],
        help="Comparison run spec(s). Repeatable.",
    )
    ap.add_argument("--bootstrap", type=int, default=10_000, help="Bootstrap samples for CI (default 10000).")
    ap.add_argument("--perm", type=int, default=20_000, help="Random sign-flip permutations (default 20000).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default 0).")
    ap.add_argument("--write", type=str, default="", help="Write JSON report to this path.")
    args = ap.parse_args()

    a = load_run(str(args.a))
    b_specs = list(args.b or [])
    if not b_specs:
        raise SystemExit("Need at least one --b RUN_SPEC")

    out: Dict[str, Any] = {
        "a": {"run_root": str(a.run_root), "baseline": a.baseline, "totals": a.totals()},
        "comparisons": [],
    }

    print(f"[A] {a.baseline} @ {a.run_root}")
    a_tot = a.totals()
    print(
        "  "
        + f"compile_hit={a_tot['compile_hit']}/{a_tot['n']} ({_fmt(a_tot['compile_hit_rate'],3)}) "
        + f"compile_density_total={_fmt(a_tot['compile_density_total'],4)} "
        + f"semantic_hit={a_tot['semantic_hit']}/{a_tot['n']} ({_fmt(a_tot['semantic_hit_rate'],3)}) "
        + f"semantic_density_total={_fmt(a_tot['semantic_density_total'],4)}"
    )

    for b_spec in b_specs:
        b = load_run(str(b_spec))
        rep = compare_runs(a, b, n_boot=int(args.bootstrap), n_perm=int(args.perm), seed=int(args.seed))
        b_tot = b.totals()
        print(f"\n[B] {b.baseline} @ {b.run_root}")
        print(
            "  "
            + f"compile_hit={b_tot['compile_hit']}/{b_tot['n']} ({_fmt(b_tot['compile_hit_rate'],3)}) "
            + f"compile_density_total={_fmt(b_tot['compile_density_total'],4)} "
            + f"semantic_hit={b_tot['semantic_hit']}/{b_tot['n']} ({_fmt(b_tot['semantic_hit_rate'],3)}) "
            + f"semantic_density_total={_fmt(b_tot['semantic_density_total'],4)}"
        )

        print("\n[Paired] A - B")
        h = rep["hit"]
        d = rep["density"]
        print(
            "  "
            + f"compile_hit Δrate={_fmt(h['compile']['diff_rate'],3)} "
            + f"(n10={h['compile']['n10_a1_b0']} n01={h['compile']['n01_a0_b1']} p={_fmt(h['compile']['mcnemar_p'],4)})"
        )
        print(
            "  "
            + f"semantic_hit Δrate={_fmt(h['semantic']['diff_rate'],3)} "
            + f"(n10={h['semantic']['n10_a1_b0']} n01={h['semantic']['n01_a0_b1']} p={_fmt(h['semantic']['mcnemar_p'],4)})"
        )
        print(
            "  "
            + f"compile_density Δmean={_fmt(d['compile']['diff_mean'],5)} "
            + f"CI[{_fmt(d['compile']['diff_mean_ci']['lo'],5)},{_fmt(d['compile']['diff_mean_ci']['hi'],5)}] "
            + f"perm_p={_fmt(d['compile']['perm_p'],4)}"
        )
        print(
            "  "
            + f"semantic_density Δmean={_fmt(d['semantic']['diff_mean'],5)} "
            + f"CI[{_fmt(d['semantic']['diff_mean_ci']['lo'],5)},{_fmt(d['semantic']['diff_mean_ci']['hi'],5)}] "
            + f"perm_p={_fmt(d['semantic']['perm_p'],4)}"
        )

        out["comparisons"].append(
            {
                "b": {"run_root": str(b.run_root), "baseline": b.baseline, "totals": b_tot},
                "paired": rep,
            }
        )

    if str(args.write).strip():
        out_path = Path(str(args.write)).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[Write] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
