#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class ProblemItem:
    index: int
    problem_name: str
    full_name: str
    informal: str
    header: str
    ground_truth: str


def _read_jsonl_first_n(path: Path, n: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_benchmark(dataset_path: Path, n: int) -> List[ProblemItem]:
    rows = _read_jsonl_first_n(dataset_path, n)
    problems: List[ProblemItem] = []
    for i, obj in enumerate(rows):
        problem_name = str(obj.get("problem_name") or f"problem_{i:04d}").strip()
        full_name = str(obj.get("full_name") or problem_name).strip()
        informal = str(obj.get("informal_stmt") or "").strip()
        header = str(obj.get("header") or "import Mathlib").strip()
        ground_truth = str(obj.get("formal_stmt") or "").strip()
        problems.append(
            ProblemItem(
                index=i,
                problem_name=problem_name,
                full_name=full_name,
                informal=informal,
                header=header,
                ground_truth=ground_truth,
            )
        )
    return problems


def _dataset_path(cwd: Path, dataset: str, dataset_path_override: str) -> Path:
    if dataset_path_override:
        p = Path(dataset_path_override).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--dataset_path not found: {p}")
        return p

    bench = cwd / "benchmark"
    if dataset == "proofnet_test":
        p = bench / "proofnet_lean4.15.0_test.jsonl"
    elif dataset == "proofnet_full":
        p = bench / "proofnet_lean4.15.0.jsonl"
    elif dataset == "combibench":
        p = bench / "combibench_lean4.15.0.jsonl"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if not p.exists():
        raise FileNotFoundError(f"Bundled dataset not found: {p}")
    return p


def _write_problem_inputs(
    out_dir: Path,
    problem: ProblemItem,
    *,
    compile_timeout: int,
    lean_server_url: str,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "problem_config.json"
    init_path = out_dir / "initial.lean"

    cfg: Dict[str, Any] = {
        "informal": problem.informal,
        "header": problem.header,
        "ground_truth": problem.ground_truth,
        # Seedbank building is compile-only by default.
        "use_beq": False,
        "use_semantic": False,
        "use_cycle_consistency": False,
        "compile_timeout": int(compile_timeout),
        "lean_server_url": str(lean_server_url),
        # No seedbank when building one.
        "init_programs_dir": "",
        # Run metadata (helps local audits; do not commit outputs).
        "baseline_mode": "ours",
        "llm_mode": "auto",
        "no_llm": False,
    }
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    init_stmt = f"theorem seed0_placeholder_{problem.problem_name} : True := by sorry"
    init_text = "\n".join(["-- EVOLVE-BLOCK-START", init_stmt, "-- EVOLVE-BLOCK-END", ""])
    init_path.write_text(init_text, encoding="utf-8")

    return {"problem_config": cfg_path, "initial_lean": init_path}


def _safe_problem_dirname(problem: ProblemItem) -> str:
    raw = f"{problem.index:04d}_{problem.problem_name}"
    return "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in raw)


def _copy_seedbank_gen0(run_root: Path, seedbank_problem_root: Path, seeds: int) -> None:
    src_gen0 = run_root / "gen_0"
    if not src_gen0.exists():
        raise FileNotFoundError(f"Missing Gen0 folder in run output: {src_gen0}")

    dst_gen0 = seedbank_problem_root / "gen_0"
    dst_gen0.mkdir(parents=True, exist_ok=True)

    copied = 0
    for i in range(int(seeds)):
        src_seed = src_gen0 / f"seed_{i}"
        src_main = src_seed / "main.lean"
        if not src_main.exists():
            raise FileNotFoundError(f"Missing seed file: {src_main}")

        dst_seed = dst_gen0 / f"seed_{i}"
        (dst_seed / "results").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_main, dst_seed / "main.lean")
        # Optional: allow fast reuse via AUTOFORMAL_REUSE_INIT_EVAL=1.
        for cand in [
            src_seed / "results" / "metrics.json",
            src_seed / "metrics.json",
        ]:
            if cand.exists():
                shutil.copy2(cand, dst_seed / "results" / "metrics.json")
                break
        copied += 1

    (seedbank_problem_root / "seedbank_info.json").write_text(
        json.dumps(
            {
                "seeds": copied,
                "layout": "gen_0/seed_i/main.lean",
                "note": "Local artifact; do not commit to an anonymous repo.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _env_from_csv(flag_value: str) -> List[str]:
    return [x.strip() for x in str(flag_value or "").split(",") if x.strip()]


def _run_one_problem(
    *,
    cwd: Path,
    run_evo_py: Path,
    problem: ProblemItem,
    run_root: Path,
    cfg_path: Path,
    init_path: Path,
    llm_mode: str,
    openai_llm_base_url: str,
    llm_models: str,
    seed: int,
    seeds_per_problem: int,
    max_llm_calls: int,
    max_repair_attempts_gen0: int,
    max_parallel_jobs: int,
    repair_openai_llm_base_url: str,
    repair_llm_models: str,
) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "build_seedbank.log"
    err_path = run_root / "build_seedbank.err"

    cmd = [
        sys.executable,
        os.fspath(run_evo_py),
        "--baseline_mode",
        "ours",
        "--seed",
        str(int(seed)),
        "--llm_mode",
        str(llm_mode),
        "--openai_llm_base_url",
        str(openai_llm_base_url),
        "--llm_models",
        str(llm_models),
        "--config",
        os.fspath(cfg_path),
        "--init_program",
        os.fspath(init_path),
        "--results_dir",
        os.fspath(run_root),
        # Gen0-only: num_generations=1 triggers Gen0 then stops.
        "--num_generations",
        "1",
        "--max_llm_calls",
        str(int(max_llm_calls)),
        "--max_parallel_jobs",
        str(int(max_parallel_jobs)),
        "--num_init_candidates_gen0",
        str(int(seeds_per_problem)),
        "--max_repair_attempts_gen0",
        str(int(max_repair_attempts_gen0)),
        # Prevent meta recommendation noise during seedbank generation.
        "--meta_rec_interval",
        "0",
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Explicitly disable semantic/cycle scoring during seedbank build.
    env["AUTOFORMAL_DISABLE_SEMANTIC"] = "1"
    env["CYCLE_API_BASE_URL"] = ""
    # Optional repair override: keep generation model separate from repair model.
    if str(repair_openai_llm_base_url).strip() and str(repair_llm_models).strip():
        env["AUTOFORMAL_REPAIR_OPENAI_LLM_BASE_URL"] = str(repair_openai_llm_base_url).strip()
        env["AUTOFORMAL_REPAIR_LLM_MODELS"] = str(repair_llm_models).strip()

    with log_path.open("w", encoding="utf-8") as out, err_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=out, stderr=err, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"run_evo failed for problem={problem.problem_name} (index={problem.index}). "
            f"See: {log_path} / {err_path}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build per-problem Gen0 seedbanks by running run_evo.py for 1 generation (Gen0-only)."
    )
    ap.add_argument("--dataset", type=str, default="proofnet_test", choices=["proofnet_test", "proofnet_full", "combibench"])
    ap.add_argument("--dataset_path", type=str, default="", help="Optional override JSONL path (otherwise uses bundled benchmark/*)")
    ap.add_argument("--num_problems", type=int, default=5, help="How many problems (from the start of JSONL)")
    ap.add_argument("--seeds_per_problem", type=int, default=16, help="How many Gen0 seeds to export per problem")
    ap.add_argument("--out_root", type=str, default="", help="Seedbank root (only gen_0/seed_i is exported). Default: /tmp/formalevolve_seedbank_*")
    ap.add_argument("--work_root", type=str, default="", help="Work root for running run_evo.py. Default: <out_root>/_build_runs")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed (increments by problem index)")
    ap.add_argument("--llm_mode", type=str, default="auto", choices=["auto", "real", "mock", "replay"])
    ap.add_argument("--openai_llm_base_url", type=str, default=os.environ.get("OPENAI_LLM_BASE_URL", "").strip())
    ap.add_argument("--llm_models", type=str, default=os.environ.get("AUTOFORMAL_LLM_MODELS", "Kimina-Autoformalizer-7B"))
    ap.add_argument("--repair_openai_llm_base_url", type=str, default=os.environ.get("AUTOFORMAL_REPAIR_OPENAI_LLM_BASE_URL", "").strip())
    ap.add_argument("--repair_llm_models", type=str, default=os.environ.get("AUTOFORMAL_REPAIR_LLM_MODELS", "").strip())
    ap.add_argument("--max_llm_calls", type=int, default=200, help="Budget for seedbank build (includes Gen0 + possible repairs)")
    ap.add_argument("--max_repair_attempts_gen0", type=int, default=2)
    ap.add_argument("--max_parallel_jobs", type=int, default=1, help="Per-problem evaluator parallelism (keep small)")
    ap.add_argument("--compile_timeout", type=int, default=60)
    ap.add_argument("--lean_server_url", type=str, default="local", help="Lean server URL or 'local' (lean-interact)")

    args = ap.parse_args(argv)

    cwd = Path(__file__).resolve().parents[1]
    run_evo_py = cwd / "run_evo.py"
    if not run_evo_py.exists():
        raise FileNotFoundError(f"run_evo.py not found at: {run_evo_py}")

    dataset_path = _dataset_path(cwd, str(args.dataset), str(args.dataset_path))
    problems = _load_benchmark(dataset_path, int(args.num_problems))
    if not problems:
        raise RuntimeError("No problems loaded (empty dataset slice?)")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else Path(f"/tmp/formalevolve_seedbank_{args.dataset}_n{len(problems)}_{ts}")
    )
    work_root = Path(args.work_root).expanduser().resolve() if args.work_root else (out_root / "_build_runs")
    out_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    (out_root / "seedbank_manifest.json").write_text(
        json.dumps(
            {
                "created_at": ts,
                "dataset": str(args.dataset),
                "num_problems": len(problems),
                "seeds_per_problem": int(args.seeds_per_problem),
                "llm_mode": str(args.llm_mode),
                "llm_models": _env_from_csv(str(args.llm_models)),
                "repair_llm_models": _env_from_csv(str(args.repair_llm_models)),
                "compile_timeout": int(args.compile_timeout),
                "lean_server_url": str(args.lean_server_url),
                "note": "Local artifact; do not commit to an anonymous repo.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    for p in problems:
        prob_dir = _safe_problem_dirname(p)
        run_root = work_root / prob_dir
        inputs_dir = run_root / "inputs"
        paths = _write_problem_inputs(
            inputs_dir,
            p,
            compile_timeout=int(args.compile_timeout),
            lean_server_url=str(args.lean_server_url),
        )
        _run_one_problem(
            cwd=cwd,
            run_evo_py=run_evo_py,
            problem=p,
            run_root=run_root,
            cfg_path=paths["problem_config"],
            init_path=paths["initial_lean"],
            llm_mode=str(args.llm_mode),
            openai_llm_base_url=str(args.openai_llm_base_url),
            llm_models=str(args.llm_models),
            seed=int(args.seed) + int(p.index),
            seeds_per_problem=int(args.seeds_per_problem),
            max_llm_calls=int(args.max_llm_calls),
            max_repair_attempts_gen0=int(args.max_repair_attempts_gen0),
            max_parallel_jobs=int(args.max_parallel_jobs),
            repair_openai_llm_base_url=str(args.repair_openai_llm_base_url),
            repair_llm_models=str(args.repair_llm_models),
        )

        seedbank_problem_root = out_root / prob_dir
        _copy_seedbank_gen0(
            run_root=run_root,
            seedbank_problem_root=seedbank_problem_root,
            seeds=int(args.seeds_per_problem),
        )
        print(f"[OK] seedbank exported: {seedbank_problem_root}")

    print(f"[DONE] Seedbank root: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

