#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class CmdResult:
    cmd: List[str]
    returncode: int
    elapsed_s: float


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _run(cmd: List[str], *, env: Optional[dict] = None) -> CmdResult:
    t0 = time.time()
    p = subprocess.run(cmd, env=env, text=True)
    return CmdResult(cmd=cmd, returncode=int(p.returncode), elapsed_s=float(time.time() - t0))


def _maybe_analyze(run_root: Path) -> None:
    analyze = Path(__file__).resolve().parent / "analyze_results.py"
    if not analyze.exists():
        return
    subprocess.run([sys.executable, str(analyze), str(run_root), "--write", str(run_root)], check=False)


def _build_baseline_cmd(
    *,
    launcher: Path,
    run_root: Path,
    run_choice: str,
    num_problems: int,
    budget_calls: int,
    concurrency: int,
    seed: int,
    dataset_path: Path,
    llm_mode: str,
    openai_llm_base_url: str,
    llm_models: str,
    lean_server_url: str,
    compile_timeout: int,
    criticlean_url: str,
    criticlean_model: str,
) -> List[str]:
    return [
        sys.executable,
        str(launcher),
        "--num_problems",
        str(num_problems),
        "--budget_calls",
        str(budget_calls),
        "--concurrency",
        str(concurrency),
        "--seed",
        str(seed),
        "--run",
        str(run_choice),
        "--dataset_path",
        str(dataset_path),
        "--out_root",
        str(run_root),
        "--llm_mode",
        str(llm_mode),
        "--openai_llm_base_url",
        str(openai_llm_base_url),
        "--llm_models",
        str(llm_models),
        "--lean_server_url",
        str(lean_server_url),
        "--compile_timeout",
        str(compile_timeout),
        "--use_semantic",
        "--no_cycle_consistency",
        "--criticlean_url",
        str(criticlean_url),
        "--criticlean_model",
        str(criticlean_model),
    ]


def _build_evolution_cmd(
    *,
    launcher: Path,
    run_root: Path,
    num_problems: int,
    budget_calls: int,
    concurrency: int,
    seed: int,
    dataset_path: Path,
    llm_mode: str,
    openai_llm_base_url: str,
    llm_models: str,
    lean_server_url: str,
    compile_timeout: int,
    criticlean_url: str,
    criticlean_model: str,
    enable_semantic_repair: bool,
    max_repair_attempts: int,
    max_repair_attempts_gen0: int,
    repair_temperature: float,
    semantic_repair_max_attempts: int,
    semantic_repair_max_attempts_gen0: int,
    semantic_repair_temperature: float,
    temperatures: str,
    num_generations: int,
    num_init_candidates_gen0: int,
    max_patch_attempts: int,
    tautology_guard_level: int,
) -> List[str]:
    cmd = [
        sys.executable,
        str(launcher),
        "--num_problems",
        str(num_problems),
        "--budget_calls",
        str(budget_calls),
        "--concurrency",
        str(concurrency),
        "--seed",
        str(seed),
        "--dataset_path",
        str(dataset_path),
        "--out_root",
        str(run_root),
        "--llm_mode",
        str(llm_mode),
        "--openai_llm_base_url",
        str(openai_llm_base_url),
        "--llm_models",
        str(llm_models),
        "--lean_server_url",
        str(lean_server_url),
        "--compile_timeout",
        str(compile_timeout),
        "--use_semantic",
        "--no_cycle_consistency",
        "--criticlean_url",
        str(criticlean_url),
        "--criticlean_model",
        str(criticlean_model),
        "--num_generations",
        str(num_generations),
        "--num_init_candidates_gen0",
        str(num_init_candidates_gen0),
        "--max_repair_attempts",
        str(max_repair_attempts),
        "--max_repair_attempts_gen0",
        str(max_repair_attempts_gen0),
        "--repair_temperature",
        str(repair_temperature),
        "--temperatures",
        str(temperatures),
        "--max_patch_attempts",
        str(max_patch_attempts),
        "--tautology_guard_level",
        str(tautology_guard_level),
        "--score_base",
        "100",
        "--score_cycle_weight",
        "0",
        "--score_semantic_bonus",
        "100",
        "--score_beq_bonus",
        "0",
    ]
    if enable_semantic_repair:
        cmd.extend(
            [
                "--enable_semantic_repair",
                "--semantic_repair_max_attempts",
                str(semantic_repair_max_attempts),
                "--semantic_repair_max_attempts_gen0",
                str(semantic_repair_max_attempts_gen0),
                "--semantic_repair_temperature",
                str(semantic_repair_temperature),
            ]
        )
    return cmd


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Run CombiBench suite and/or sweep concurrency.")
    ap.add_argument("--mode", choices=["suite", "sweep"], default="suite")
    ap.add_argument("--tag", type=str, default="run", help="Prefix for run directory names.")

    ap.add_argument("--num_problems", type=int, default=100)
    ap.add_argument("--budget_calls", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--concurrency_list", type=str, default="4,8,12,16")

    ap.add_argument("--dataset_path", type=str, default="benchmark/combibench_lean4.15.0.jsonl")
    ap.add_argument("--out_base", type=str, default="experiments/combibench100_call100")

    ap.add_argument("--llm_mode", choices=["auto", "real", "mock", "replay"], default="real")
    ap.add_argument("--openai_llm_base_url", type=str, default="http://127.0.0.1:8009/v1")
    ap.add_argument("--llm_models", type=str, default="Kimina-Autoformalizer-7B")

    ap.add_argument("--lean_server_url", type=str, default=os.environ.get("LEAN_SERVER_URL", "http://127.0.0.1:8002/api/check"))
    ap.add_argument("--compile_timeout", type=int, default=600)

    ap.add_argument("--criticlean_url", type=str, default=os.environ.get("CRITIC_LEAN_URL", "http://127.0.0.1:6082/v1/chat/completions"))
    ap.add_argument("--criticlean_model", type=str, default=os.environ.get("CRITIC_LEAN_MODEL", "criticlean-qwen3-14b"))

    ap.add_argument("--run_baselines", action="store_true")
    ap.add_argument("--run_evolution", action="store_true")
    ap.add_argument("--no_analyze", action="store_true", help="Skip analyze_results.py --write.")

    ap.add_argument("--enable_semantic_repair", action="store_true", help="Evolution: enable semantic repair.")
    ap.add_argument("--max_repair_attempts", type=int, default=2)
    ap.add_argument("--max_repair_attempts_gen0", type=int, default=2)
    ap.add_argument("--repair_temperature", type=float, default=0.7)
    ap.add_argument("--semantic_repair_max_attempts", type=int, default=2)
    ap.add_argument("--semantic_repair_max_attempts_gen0", type=int, default=2)
    ap.add_argument("--semantic_repair_temperature", type=float, default=0.7)
    ap.add_argument("--temperatures", type=str, default="0,0.5,1.0")
    ap.add_argument("--num_generations", type=int, default=400)
    ap.add_argument("--num_init_candidates_gen0", type=int, default=3)
    ap.add_argument("--max_patch_attempts", type=int, default=1)
    ap.add_argument("--tautology_guard_level", type=int, default=1)

    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    baseline_launcher = here / "proofnet20_baselines_calls100_round1" / "launcher.py"
    evo_launcher = here / "combibench100_evolve_calls100_round1" / "launcher.py"
    if not baseline_launcher.exists():
        print(f"[ERROR] baseline launcher not found: {baseline_launcher}", file=sys.stderr)
        return 2
    if not evo_launcher.exists():
        print(f"[ERROR] evolution launcher not found: {evo_launcher}", file=sys.stderr)
        return 2

    out_base = Path(args.out_base).resolve()
    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        print(f"[ERROR] dataset not found: {dataset_path}", file=sys.stderr)
        return 2

    run_baselines = bool(args.run_baselines or (args.mode == "suite" and not args.run_evolution))
    run_evolution = bool(args.run_evolution or (args.mode == "suite"))

    def run_suite_once(conc: int, *, tag: str) -> List[CmdResult]:
        ts = _ts()
        results: List[CmdResult] = []

        if run_baselines:
            # sample
            sample_root = out_base / "sample100" / f"{tag}_{ts}__combibench__n{args.num_problems}__calls{args.budget_calls}__seed{args.seed}__conc{conc}"
            cmd = _build_baseline_cmd(
                launcher=baseline_launcher,
                run_root=sample_root,
                run_choice="sample",
                num_problems=args.num_problems,
                budget_calls=args.budget_calls,
                concurrency=conc,
                seed=args.seed,
                dataset_path=dataset_path,
                llm_mode=args.llm_mode,
                openai_llm_base_url=args.openai_llm_base_url,
                llm_models=args.llm_models,
                lean_server_url=args.lean_server_url,
                compile_timeout=args.compile_timeout,
                criticlean_url=args.criticlean_url,
                criticlean_model=args.criticlean_model,
            )
            results.append(_run(cmd))
            if not args.no_analyze:
                _maybe_analyze(sample_root)

            # strong_compile
            sc_root = out_base / "compile_repair100" / f"{tag}_{ts}__combibench__n{args.num_problems}__calls{args.budget_calls}__seed{args.seed}__conc{conc}"
            cmd = _build_baseline_cmd(
                launcher=baseline_launcher,
                run_root=sc_root,
                run_choice="strong_compile",
                num_problems=args.num_problems,
                budget_calls=args.budget_calls,
                concurrency=conc,
                seed=args.seed,
                dataset_path=dataset_path,
                llm_mode=args.llm_mode,
                openai_llm_base_url=args.openai_llm_base_url,
                llm_models=args.llm_models,
                lean_server_url=args.lean_server_url,
                compile_timeout=args.compile_timeout,
                criticlean_url=args.criticlean_url,
                criticlean_model=args.criticlean_model,
            )
            results.append(_run(cmd))
            if not args.no_analyze:
                _maybe_analyze(sc_root)

            # strong_semantic
            ss_root = out_base / "semantic_repair100" / f"{tag}_{ts}__combibench__n{args.num_problems}__calls{args.budget_calls}__seed{args.seed}__conc{conc}"
            cmd = _build_baseline_cmd(
                launcher=baseline_launcher,
                run_root=ss_root,
                run_choice="strong_semantic",
                num_problems=args.num_problems,
                budget_calls=args.budget_calls,
                concurrency=conc,
                seed=args.seed,
                dataset_path=dataset_path,
                llm_mode=args.llm_mode,
                openai_llm_base_url=args.openai_llm_base_url,
                llm_models=args.llm_models,
                lean_server_url=args.lean_server_url,
                compile_timeout=args.compile_timeout,
                criticlean_url=args.criticlean_url,
                criticlean_model=args.criticlean_model,
            )
            results.append(_run(cmd))
            if not args.no_analyze:
                _maybe_analyze(ss_root)

        if run_evolution:
            evo_root = out_base / "evolution100" / f"{tag}_{ts}__combibench__n{args.num_problems}__calls{args.budget_calls}__seed{args.seed}__conc{conc}"
            cmd = _build_evolution_cmd(
                launcher=evo_launcher,
                run_root=evo_root,
                num_problems=args.num_problems,
                budget_calls=args.budget_calls,
                concurrency=conc,
                seed=args.seed,
                dataset_path=dataset_path,
                llm_mode=args.llm_mode,
                openai_llm_base_url=args.openai_llm_base_url,
                llm_models=args.llm_models,
                lean_server_url=args.lean_server_url,
                compile_timeout=args.compile_timeout,
                criticlean_url=args.criticlean_url,
                criticlean_model=args.criticlean_model,
                enable_semantic_repair=bool(args.enable_semantic_repair),
                max_repair_attempts=int(args.max_repair_attempts),
                max_repair_attempts_gen0=int(args.max_repair_attempts_gen0),
                repair_temperature=float(args.repair_temperature),
                semantic_repair_max_attempts=int(args.semantic_repair_max_attempts),
                semantic_repair_max_attempts_gen0=int(args.semantic_repair_max_attempts_gen0),
                semantic_repair_temperature=float(args.semantic_repair_temperature),
                temperatures=str(args.temperatures),
                num_generations=int(args.num_generations),
                num_init_candidates_gen0=int(args.num_init_candidates_gen0),
                max_patch_attempts=int(args.max_patch_attempts),
                tautology_guard_level=int(args.tautology_guard_level),
            )
            results.append(_run(cmd))
            if not args.no_analyze:
                _maybe_analyze(evo_root)

        return results

    if args.mode == "suite":
        results = run_suite_once(int(args.concurrency), tag=str(args.tag))
        for r in results:
            print(f"[done] rc={r.returncode} elapsed_s={r.elapsed_s:.1f} cmd={' '.join(r.cmd)}")
        bad = [r for r in results if r.returncode != 0]
        return 0 if not bad else 1

    # Sweep mode: run only evolution by default.
    concs = _parse_int_list(args.concurrency_list)
    if not concs:
        print("[ERROR] empty --concurrency_list", file=sys.stderr)
        return 2

    # In sweep mode, default to evolution-only unless explicitly requested otherwise.
    if not args.run_baselines and not args.run_evolution:
        run_evolution = True
        run_baselines = False

    sweep_tag = str(args.tag).strip() or "sweep"
    sweep_base = out_base / "_concurrency_sweep" / f"{sweep_tag}_{_ts()}__n{args.num_problems}__calls{args.budget_calls}__seed{args.seed}"
    sweep_base.mkdir(parents=True, exist_ok=True)
    # In sweep mode, isolate outputs under the sweep directory to avoid mixing
    # with the main experiment tree.
    out_base = sweep_base
    print(f"[sweep] out_base={out_base}")

    summary_lines: List[str] = []
    for conc in concs:
        res = run_suite_once(int(conc), tag=f"{sweep_tag}_conc{conc}")
        ok = all(r.returncode == 0 for r in res)
        elapsed = sum(r.elapsed_s for r in res)
        summary_lines.append(f"conc={conc}\tok={int(ok)}\telapsed_s={elapsed:.1f}")
        print(summary_lines[-1])

    (sweep_base / "sweep_summary.tsv").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"[sweep] wrote {sweep_base / 'sweep_summary.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
