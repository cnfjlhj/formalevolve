#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _wait_for(path: Path, *, poll_s: float) -> None:
    while not path.exists():
        time.sleep(float(poll_s))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Wait for CombiBench round-1 runs (sample/strong_compile/strong_semantic + ours-after-strong_compile)\n"
            "then generate analysis reports.\n"
        )
    )
    ap.add_argument(
        "--strong_compile_run_root",
        type=str,
        default=(
            "experiments/proofnet20_baselines_calls100_round1/"
            "run_20251231_012647__combibench__n100__calls100__seed0__conc20__runstrong_compile__repT0p7__semT0p7__critnpu"
        ),
    )
    ap.add_argument(
        "--strong_semantic_run_root",
        type=str,
        default=(
            "experiments/proofnet20_baselines_calls100_round1/"
            "run_20251231_012647__combibench__n100__calls100__seed0__conc12__runstrong_semantic__repT0p7__semT0p7__critnpu"
        ),
    )
    ap.add_argument(
        "--sample_run_root",
        type=str,
        default=(
            "experiments/proofnet20_baselines_calls100_round1/"
            "run_20251231_012647__combibench__n100__calls100__seed0__conc12__runsample__repT0p7__semT0p7__critnpu"
        ),
    )
    ap.add_argument(
        "--evolve_schedule_state",
        type=str,
        default="experiments/combibench100_evolve_calls100_round1/schedule_state_after_strong_compile.json",
        help="Path to schedule_state_after_strong_compile.json to discover ours run root.",
    )
    ap.add_argument("--poll_s", type=float, default=60.0)
    ap.add_argument("--out_dir", type=str, default="", help="Where to write analysis outputs (default: auto).")
    args = ap.parse_args()

    workdir = Path(__file__).resolve().parents[1]

    strong_compile = (workdir / str(args.strong_compile_run_root)).resolve()
    strong_semantic = (workdir / str(args.strong_semantic_run_root)).resolve()
    sample = (workdir / str(args.sample_run_root)).resolve()
    schedule_state = (workdir / str(args.evolve_schedule_state)).resolve()

    print(f"[{_now()}] Waiting strong_compile summary: {strong_compile / 'summary.json'}")
    _wait_for(strong_compile / "summary.json", poll_s=float(args.poll_s))
    print(f"[{_now()}] strong_compile done")

    print(f"[{_now()}] Waiting strong_semantic summary: {strong_semantic / 'summary.json'}")
    _wait_for(strong_semantic / "summary.json", poll_s=float(args.poll_s))
    print(f"[{_now()}] strong_semantic done")

    print(f"[{_now()}] Waiting sample summary: {sample / 'summary.json'}")
    _wait_for(sample / "summary.json", poll_s=float(args.poll_s))
    print(f"[{_now()}] sample done")

    print(f"[{_now()}] Waiting ours schedule state: {schedule_state}")
    _wait_for(schedule_state, poll_s=float(args.poll_s))
    ours_root: Optional[Path] = None
    while ours_root is None:
        st = _load_json(schedule_state) or {}
        out = str(st.get("evolve_out_root") or "").strip()
        if out:
            p = Path(out)
            ours_root = p if p.is_absolute() else (workdir / p).resolve()
            break
        time.sleep(float(args.poll_s))
    assert ours_root is not None
    print(f"[{_now()}] Detected ours run_root: {ours_root}")

    print(f"[{_now()}] Waiting ours summary: {ours_root / 'summary.json'}")
    _wait_for(ours_root / "summary.json", poll_s=float(args.poll_s))
    print(f"[{_now()}] ours done")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(str(args.out_dir)).expanduser().resolve() if str(args.out_dir).strip() else (ours_root.parent / f"analysis_round1_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    analyze_cmd = [
        str(workdir / "experiments" / "analyze_results.py"),
        str(strong_compile),
        str(strong_semantic),
        str(sample),
        str(ours_root),
        "--write",
        str(out_dir),
    ]
    paired_cmd = [
        str(workdir / "experiments" / "paired_stats.py"),
        "--a",
        f"{ours_root}::ours",
        "--b",
        f"{sample}::sample",
        "--b",
        f"{strong_compile}::strong_compile",
        "--b",
        f"{strong_semantic}::strong_semantic",
        "--write",
        str(out_dir / "paired_stats.json"),
    ]

    print(f"[{_now()}] Running analyze_results -> {out_dir}")
    print(f"[{_now()}] $ python {' '.join(analyze_cmd)}")
    rc1 = subprocess.call([sys.executable, *analyze_cmd], cwd=str(workdir))
    if rc1 != 0:
        print(f"[{_now()}] analyze_results failed (rc={rc1})")
        return int(rc1)

    print(f"[{_now()}] Running paired_stats -> {out_dir / 'paired_stats.json'}")
    print(f"[{_now()}] $ python {' '.join(paired_cmd)}")
    rc2 = subprocess.call([sys.executable, *paired_cmd], cwd=str(workdir))
    if rc2 != 0:
        print(f"[{_now()}] paired_stats failed (rc={rc2})")
        return int(rc2)

    print(f"[{_now()}] Done. Reports in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
