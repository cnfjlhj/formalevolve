                      
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _compress_ranges(xs: List[int]) -> List[Tuple[int, int]]:
    if not xs:
        return []
    xs = sorted(set(int(x) for x in xs))
    out: List[Tuple[int, int]] = []
    start = prev = xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append((start, prev))
        start = prev = x
    out.append((start, prev))
    return out


def _iter_problem_run_dirs(manifest: Dict[str, Any], baseline: str) -> Iterable[Tuple[str, Path, Path]]:
    for p in manifest.get("problems", []) or []:
        problem_id = str(p.get("problem_id") or "")
        input_dir = Path(str(p.get("input_dir") or "")).resolve()
        runs = p.get("runs") or {}
        run_dir = runs.get(baseline)
        if not problem_id or not run_dir:
            continue
        yield problem_id, Path(str(run_dir)).resolve(), input_dir


def _missing_sample_indices(results_dir: Path, *, num_samples: int) -> List[int]:
    base = results_dir / "baseline_batchN"
    if not base.exists():
        return list(range(num_samples))
    missing: List[int] = []
    for i in range(int(num_samples)):
        m = base / f"sample_{i}" / "results" / "metrics.json"
        if not m.exists():
            missing.append(i)
            continue
        try:
            if m.stat().st_size <= 0:
                missing.append(i)
        except Exception:
            missing.append(i)
    return missing


@dataclass(frozen=True)
class RerunTask:
    problem_id: str
    results_dir: Path
    config_path: Path
    missing: List[int]


def _build_run_evo_cmd(
    *,
    run_evo_py: Path,
    manifest: Dict[str, Any],
    task: RerunTask,
    max_llm_calls: int,
) -> List[str]:
    budget_calls = int(manifest.get("budget_calls") or 0)
    if budget_calls <= 0:
        raise ValueError(f"Invalid budget_calls in manifest: {manifest.get('budget_calls')!r}")
    max_llm_calls = int(max_llm_calls)
    if max_llm_calls <= 0:
        raise ValueError(f"Invalid max_llm_calls: {max_llm_calls!r}")

    cmd: List[str] = [
        sys.executable,
        os.fspath(run_evo_py),
        "--baseline_mode",
        "batchN",
        "--seed",
        str(int(manifest.get("seed") or 0)),
        "--llm_mode",
        str(manifest.get("llm_mode") or "auto"),
        "--openai_llm_base_url",
        str(manifest.get("openai_llm_base_url") or ""),
        "--llm_models",
        str(manifest.get("llm_models") or ""),
        "--cycle_api_base_url",
        str(manifest.get("cycle_api_base_url") or ""),
        "--cycle_model_name",
        str(manifest.get("cycle_model_name") or ""),
        "--results_dir",
        os.fspath(task.results_dir),
        "--max_llm_calls",
        str(int(max_llm_calls)),
        "--num_init_candidates_gen0",
        str(budget_calls),
        "--max_repair_attempts",
        str(int(manifest.get("max_repair_attempts") or 2)),
        "--max_repair_attempts_gen0",
        str(int(manifest.get("max_repair_attempts_gen0") or 5)),
        "--config",
        os.fspath(task.config_path),
    ]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fill missing baseline_batchN samples by re-running only completed problems (no clobber)."
    )
    ap.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Baseline launcher out_root (must contain manifest.json and status.json).",
    )
    ap.add_argument(
        "--baseline",
        type=str,
        default="",
        help="Baseline label to fix (default: all baselines in manifest).",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Max parallel reruns (default 1; keep low to avoid interfering with ongoing runs).",
    )
    ap.add_argument(
        "--watch",
        action="store_true",
        help="Continuously rescan finished tasks and fill holes until launcher finishes (default: enabled when --run).",
    )
    ap.add_argument(
        "--poll_s",
        type=float,
        default=30.0,
        help="Polling interval seconds for --watch (default 30).",
    )
    ap.add_argument(
        "--run",
        action="store_true",
        help="Actually execute reruns (default: dry-run summary only).",
    )
    args = ap.parse_args()

    if args.run and not args.watch:
                                                                         
        args.watch = True

    out_root = Path(args.out_root).resolve()
    manifest_path = out_root / "manifest.json"
    status_path = out_root / "status.json"
    manifest = _read_json(manifest_path) or {}
    if not manifest:
        raise SystemExit(f"manifest.json not found or invalid: {manifest_path}")

    baselines = [str(b).strip() for b in (manifest.get("baselines") or []) if str(b).strip()]
    if args.baseline:
        baselines = [str(args.baseline).strip()]
    if not baselines:
        raise SystemExit("No baselines found in manifest.json")

    budget_calls = int(manifest.get("budget_calls") or 0)
    if budget_calls <= 0:
        raise SystemExit(f"Invalid manifest budget_calls: {manifest.get('budget_calls')!r}")

    run_evo_py = Path(__file__).resolve().parents[1] / "run_evo.py"
    if not run_evo_py.exists():
        raise SystemExit(f"run_evo.py not found: {run_evo_py}")

    def _scan_tasks(status: Dict[str, Any]) -> List[RerunTask]:
        finished_dirs: set[Path] = set()
        for e in status.get("finished", []) or []:
            rd = e.get("results_dir")
            if rd:
                finished_dirs.add(Path(str(rd)).resolve())

        tasks: List[RerunTask] = []
        for baseline in baselines:
            for problem_id, results_dir, input_dir in _iter_problem_run_dirs(manifest, baseline):
                if finished_dirs and results_dir not in finished_dirs:
                    continue
                config_path = input_dir / "problem_config.json"
                missing = _missing_sample_indices(results_dir, num_samples=budget_calls)
                if not missing:
                    continue
                tasks.append(
                    RerunTask(
                        problem_id=problem_id,
                        results_dir=results_dir,
                        config_path=config_path,
                        missing=missing,
                    )
                )
        return tasks

    print(f"[Init] out_root={out_root}")
    print(f"[Init] baselines={baselines} budget_calls={budget_calls} watch={bool(args.watch)} poll_s={float(args.poll_s)}")

    if not args.run:
        status = _read_json(status_path) or {}
        tasks = _scan_tasks(status)
        if not tasks:
            print("[OK] No missing samples detected (within finished tasks).")
            return 0
        print(f"[Scan] tasks_to_fix={len(tasks)}")
        for t in tasks[:20]:
            ranges = _compress_ranges(t.missing)
            print(f"  - {t.problem_id} missing={len(t.missing)} ranges={ranges}")
        if len(tasks) > 20:
            print(f"  ... ({len(tasks) - 20} more)")
        print("[DryRun] Pass --run to execute reruns.")
        return 0

                                                                   
    baseline_defs = manifest.get("baseline_definitions") or {}
    sem_enabled = False
    disable_repair = False
    for b in baselines:
        spec = baseline_defs.get(b) or {}
        sem_enabled = bool(spec.get("enable_semantic_repair"))
        disable_repair = bool(spec.get("disable_repair"))
        break

                                                                     
                                                                                       
    max_repair_attempts_gen0 = int(manifest.get("max_repair_attempts_gen0") or 0)
    max_semantic_repair_attempts = int(manifest.get("max_semantic_repair_attempts") or 0)
    multiplier = 1
    if not disable_repair:
        multiplier += max(0, max_repair_attempts_gen0)
    if sem_enabled:
        multiplier += max(0, max_semantic_repair_attempts)
    max_llm_calls_eff = int(budget_calls) * int(multiplier)
    print(
        f"[Budget] max_llm_calls_eff={max_llm_calls_eff} "
        f"(budget_calls={budget_calls} * multiplier={multiplier}; "
        f"disable_repair={disable_repair} sem_enabled={sem_enabled})"
    )

    base_env = dict(os.environ)
    base_env["AUTOFORMAL_ENABLE_SEMANTIC_REPAIR"] = "1" if sem_enabled else "0"
    if sem_enabled:
        base_env["AUTOFORMAL_SEMANTIC_REPAIR_MAX_ATTEMPTS"] = str(int(manifest.get("max_semantic_repair_attempts") or 0))
        base_env["AUTOFORMAL_SEMANTIC_REPAIR_MAX_ATTEMPTS_GEN0"] = str(
            int(manifest.get("max_semantic_repair_attempts") or 0)
        )
        base_env["AUTOFORMAL_SEMANTIC_REPAIR_TEMPERATURE"] = str(float(manifest.get("semantic_repair_temperature") or 0.0))

                             
    max_workers = max(1, int(args.concurrency))
    failures = 0

    def _start_one(t: RerunTask) -> subprocess.Popen[str]:
        cmd = _build_run_evo_cmd(
            run_evo_py=run_evo_py,
            manifest=manifest,
            task=t,
            max_llm_calls=max_llm_calls_eff,
        )
        if disable_repair:
            cmd.append("--disable_repair")
                                                                                              
        cmd.extend(["--max_parallel_jobs", "1"])
        log_path = t.results_dir / "fill_batchN.log"
        err_path = t.results_dir / "fill_batchN.err"
        log_f = open(log_path, "a", encoding="utf-8")
        err_f = open(err_path, "a", encoding="utf-8")
        print(f"[Run] {t.problem_id} -> {t.results_dir} (missing={len(t.missing)})")
        return subprocess.Popen(cmd, env=base_env, stdout=log_f, stderr=err_f, text=True)

    while True:
        status = _read_json(status_path) or {}
        tasks = _scan_tasks(status)
        tasks.sort(key=lambda t: (-len(t.missing), t.problem_id))
        if tasks:
            print(f"[Scan] tasks_to_fix={len(tasks)} (finished={len(status.get('finished',[]) or [])}, running={len(status.get('running',[]) or [])}, pending={status.get('pending')})")
            for t in tasks[:5]:
                ranges = _compress_ranges(t.missing)
                print(f"  - {t.problem_id} missing={len(t.missing)} ranges={ranges}")
            if len(tasks) > 5:
                print(f"  ... ({len(tasks) - 5} more)")
        else:
            pending = int(status.get("pending", 0) or 0)
            running_n = len(status.get("running", []) or [])
            if not args.watch or (pending == 0 and running_n == 0):
                print("[OK] No missing samples detected (within finished tasks).")
                break
            time.sleep(max(0.5, float(args.poll_s)))
            continue

        running: List[subprocess.Popen[str]] = []
                                                                                         
        queue = list(tasks[:max_workers]) if args.watch else list(tasks)

        while queue or running:
            while queue and len(running) < max_workers:
                running.append(_start_one(queue.pop(0)))

            still: List[subprocess.Popen[str]] = []
            for p in running:
                rc = p.poll()
                if rc is None:
                    still.append(p)
                    continue
                if rc != 0:
                    failures += 1
            running = still
            time.sleep(0.5)

        if not args.watch:
            break

    if failures:
        print(f"[Done] failures={failures} (see per-problem fill_batchN.err)")
        return 2
    print("[Done] All reruns finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
