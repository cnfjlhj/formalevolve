#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def load_benchmark(dataset_path: Path, n: int) -> List[ProblemItem]:
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


def _resolve_seedbank_dir(init_programs_root: Path, problem_name: str, problem_index: int) -> Optional[Path]:
    """Resolve a per-problem seedbank directory under a shared root.

    Supported layouts:
    - <root>/<problem_name>/seed_0/...
    - <root>/<problem_name>/gen_0/seed_0/...
    - <root>/*_<problem_name>/gen_0/seed_0/...  (e.g., `0000_exercise_xxx`)
    """
    if not str(problem_name or "").strip():
        return None

    root = init_programs_root
    direct = root / str(problem_name).strip()
    if direct.exists() and direct.is_dir():
        return direct

    # Prefer an explicit index-prefixed directory when available.
    # ProofNet-style seedbanks are commonly laid out as: 0000_<problem_name>, 0001_<problem_name>, ...
    try:
        idx = int(problem_index)
    except Exception:
        idx = -1
    if idx >= 0:
        indexed = root / f"{idx:04d}_{problem_name}"
        if indexed.exists() and indexed.is_dir():
            return indexed

    matches = sorted([p for p in root.glob(f"*_{problem_name}") if p.is_dir()])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # If multiple exist, fall back to the index-prefixed one when possible.
        if idx >= 0:
            indexed = root / f"{idx:04d}_{problem_name}"
            if indexed in matches:
                return indexed
        raise ValueError(
            f"Ambiguous seedbank dir for problem_name={problem_name} (index={problem_index}): "
            + ", ".join(str(p) for p in matches[:10])
        )

    return None


def _normalize_http_base_url(url: str) -> str:
    s = str(url or "").strip()
    if not s:
        return ""
    if not (s.startswith("http://") or s.startswith("https://")):
        s = "http://" + s
    return s.rstrip("/")


def _http_get_json(url: str, timeout_s: float = 2.0) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "formalevolve-run_dataset_pilot/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        data = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(data)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object from {url}, got {type(obj).__name__}")
    return obj


def _probe_single_openai_model_id(base_url: str, timeout_s: float = 2.0) -> str:
    """Probe an OpenAI-compatible server and return the single model id.

    This keeps the CLI minimal: when the server hosts exactly one model (common for vLLM),
    users can specify only `--criticlean_base_url` and omit `--criticlean_model`.
    """
    base = _normalize_http_base_url(base_url)
    if not base:
        raise ValueError("Empty base_url for model probe")
    models_url = base + "/v1/models"
    obj = _http_get_json(models_url, timeout_s=timeout_s)
    data = obj.get("data")
    if not isinstance(data, list):
        raise ValueError(f"Unexpected /v1/models payload from {models_url}: missing 'data' list")
    ids: List[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "").strip()
        if mid:
            ids.append(mid)
    ids = sorted(set(ids))
    if len(ids) == 1:
        return ids[0]
    if not ids:
        raise ValueError(f"No model ids found from {models_url}")
    raise ValueError(
        f"Multiple models found from {models_url}; please pass --criticlean_model. "
        f"Models={ids}"
    )


def _criticlean_chat_url(base_url: str) -> str:
    base = _normalize_http_base_url(base_url)
    if not base:
        return ""
    return base + "/v1/chat/completions"


def write_problem_inputs(
    out_dir: Path,
    problem: ProblemItem,
    *,
    init_programs_dir: str,
    cycle_api_base_url: str,
    cycle_model_name: str,
    lean_server_url: str,
    compile_timeout: int,
    use_beq: bool,
    use_semantic: bool,
    use_cycle_consistency: bool,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "problem_config.json"
    init_path = out_dir / "initial.lean"

    config: Dict[str, Any] = {
        "informal": problem.informal,
        "header": problem.header,
        "ground_truth": problem.ground_truth,
        "use_beq": bool(use_beq),
        "use_semantic": bool(use_semantic),
        "use_cycle_consistency": bool(use_cycle_consistency),
        # Optional seedbank for Gen0 bootstrapping. The runner expects this to
        # point to a seedbank directory containing either:
        # - seed_i/main.lean, or
        # - gen_0/seed_i/main.lean
        "init_programs_dir": str(init_programs_dir or "").strip(),
        "cycle_api_base_url": cycle_api_base_url,
        "cycle_model_name": cycle_model_name,
        "informalize_prompt_template": "Informalize: {formal_statement}",
        "cycle_softmax_temperature": 3.5,
        "cycle_temperature": 0.0,
        "cycle_max_tokens": 1024,
        "cycle_normalize_decl_name": True,
        "cycle_normalized_decl_name": "my_theorem",
        "semantic_normalize_decl_name": True,
        "semantic_normalized_decl_name": "my_theorem",
        "beq_normalize_decl_name": True,
        "beq_candidate_decl_name": "my_cand",
        "beq_ground_truth_decl_name": "my_gt",
        "compile_timeout": int(compile_timeout),
        "lean_server_url": str(lean_server_url),
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    init_stmt = f"theorem seed0_placeholder_{problem.problem_name} : True := by sorry"
    init_text = "\n".join(
        [
            "-- EVOLVE-BLOCK-START",
            init_stmt,
            "-- EVOLVE-BLOCK-END",
            "",
        ]
    )
    init_path.write_text(init_text, encoding="utf-8")

    return {"problem_config": config_path, "initial_lean": init_path}


def _sanitize_tmux_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in s)[:60]


def run_one(
    *,
    cwd: Path,
    run_evo_py: Path,
    baseline_mode: str,
    seed: int,
    llm_mode: str,
    openai_llm_base_url: str,
    llm_models: str,
    cycle_api_base_url: str,
    cycle_model_name: str,
    max_llm_calls: int,
    num_generations: int,
    max_parallel_jobs: int,
    meta_rec_interval: int,
    stagnation_generations: int,
    max_soft_resets: int,
    config_path: Path,
    init_program_path: Path,
    results_dir: Path,
) -> int:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run_evo.log"
    err_path = results_dir / "run_evo.err"

    cmd = [
        sys.executable,
        str(run_evo_py),
        "--baseline_mode",
        str(baseline_mode),
        "--seed",
        str(seed),
        "--llm_mode",
        str(llm_mode),
        "--openai_llm_base_url",
        str(openai_llm_base_url),
        "--llm_models",
        str(llm_models),
        "--cycle_api_base_url",
        str(cycle_api_base_url),
        "--cycle_model_name",
        str(cycle_model_name),
        "--meta_rec_interval",
        str(meta_rec_interval),
        "--stagnation_generations",
        str(stagnation_generations),
        "--max_soft_resets",
        str(max_soft_resets),
        "--max_parallel_jobs",
        str(max_parallel_jobs),
        "--config",
        str(config_path),
        "--init_program",
        str(init_program_path),
        "--results_dir",
        str(results_dir),
        "--num_generations",
        str(num_generations),
        "--max_llm_calls",
        str(max_llm_calls),
    ]

    env = os.environ.copy()
    # Keep logs deterministic-ish and readable.
    env.setdefault("PYTHONUNBUFFERED", "1")

    with log_path.open("w", encoding="utf-8") as out, err_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=out,
            stderr=err,
            text=True,
        )
        return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="proofnet_test",
        choices=["proofnet_test", "proofnet_full", "combibench"],
        help="Which bundled benchmark JSONL to run (default: proofnet_test).",
    )
    parser.add_argument("--num_problems", type=int, default=10)
    parser.add_argument("--max_llm_calls", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline_mode", type=str, default="ours", choices=["ours", "batchN", "repairloop1"])

    parser.add_argument("--llm_mode", type=str, default="auto", choices=["auto", "real", "mock", "replay"])
    parser.add_argument("--openai_llm_base_url", type=str, default=os.environ.get("OPENAI_LLM_BASE_URL", ""))
    parser.add_argument("--llm_models", type=str, default=os.environ.get("AUTOFORMAL_LLM_MODELS", "Kimina-Autoformalizer-7B"))
    parser.add_argument(
        "--patch_openai_llm_base_url",
        type=str,
        default=os.environ.get("AUTOFORMAL_PATCH_OPENAI_LLM_BASE_URL", ""),
        help="Optional OpenAI-compatible base URL for patch/edit proposals. Empty reuses OPENAI_LLM_BASE_URL.",
    )
    parser.add_argument(
        "--patch_llm_models",
        type=str,
        default=os.environ.get("AUTOFORMAL_PATCH_LLM_MODELS", ""),
        help="Comma-separated model names for patch/edit proposals. Empty reuses --llm_models.",
    )
    parser.add_argument("--cycle_api_base_url", type=str, default=os.environ.get("CYCLE_API_BASE_URL", ""))
    parser.add_argument("--cycle_model_name", type=str, default=os.environ.get("CYCLE_MODEL_NAME", "Qwen2.5-32B-Instruct"))

    # CriticLean semantic judge (used when --use_semantic is enabled).
    parser.add_argument(
        "--criticlean_base_url",
        type=str,
        default=os.environ.get("CRITIC_LEAN_BASE_URL", ""),
        help="CriticLean OpenAI-compatible base URL (e.g., http://<host>:<port>). "
             "When set and --use_semantic is enabled, the runner uses <base>/v1/chat/completions.",
    )
    parser.add_argument(
        "--criticlean_model",
        type=str,
        default=os.environ.get("CRITIC_LEAN_MODEL", ""),
        help="CriticLean model id (from <base>/v1/models). "
             "If empty and the server hosts a single model, it is auto-detected.",
    )

    parser.add_argument(
        "--lean_server_url",
        type=str,
        default="local",
        help="Lean server URL (http(s)://...) or 'local' to force local lean-interact compilation (default: local).",
    )
    parser.add_argument("--compile_timeout", type=int, default=60)

    parser.add_argument("--use_beq", action="store_true")
    parser.add_argument("--use_semantic", action="store_true")
    # Default ON (paper/system default). Provide an explicit disable flag for ablations.
    parser.add_argument(
        "--use_cycle_consistency",
        dest="use_cycle_consistency",
        action="store_true",
        default=False,
        help="Enable cycle-consistency scoring (default: disabled; not used in the main protocol).",
    )
    parser.add_argument(
        "--no_cycle_consistency",
        dest="use_cycle_consistency",
        action="store_false",
        help="Disable cycle-consistency scoring (default).",
    )

    parser.add_argument("--num_generations", type=int, default=400)
    parser.add_argument("--max_parallel_jobs", type=int, default=1)
    parser.add_argument("--meta_rec_interval", type=int, default=0)
    parser.add_argument(
        "--stagnation_generations",
        type=int,
        default=5,
        help="Generations without improvement before soft reset (passed to run_evo.py)",
    )
    parser.add_argument(
        "--max_soft_resets",
        type=int,
        default=3,
        help="Max number of soft resets before stopping (passed to run_evo.py)",
    )
    parser.add_argument(
        "--parent_selection_strategy",
        type=str,
        default="weighted",
        choices=["cycle_softmax", "best_of_n", "weighted", "power_law", "beam_search"],
        help="Parent selection strategy (passed to run_evo.py)",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Optional override path to a JSONL dataset. When empty, uses the bundled `benchmark/` files.",
    )
    parser.add_argument(
        "--init_programs_root",
        type=str,
        default="",
        help="Optional root directory containing per-problem seedbanks for Gen0 bootstrapping. "
             "Each problem resolves <root>/<problem_name> or <root>/*_<problem_name>.",
    )
    parser.add_argument(
        "--num_init_candidates_gen0",
        type=int,
        default=3,
        help="Number of Gen0 seeds to attempt (passed to run_evo.py). "
             "When a seedbank is provided, this controls how many seed_i programs are reused.",
    )
    parser.add_argument("--out_root", type=str, default="")

    args = parser.parse_args()

    cwd = Path(__file__).resolve().parents[1]
    run_evo_py = cwd / "run_evo.py"
    if not run_evo_py.exists():
        raise FileNotFoundError(f"run_evo.py not found at: {run_evo_py}")

    if args.dataset_path:
        dataset_path = Path(args.dataset_path).resolve()
    else:
        benchmark_dir = cwd / "benchmark"
        if args.dataset == "proofnet_test":
            dataset_path = benchmark_dir / "proofnet_lean4.15.0_test.jsonl"
        elif args.dataset == "proofnet_full":
            dataset_path = benchmark_dir / "proofnet_lean4.15.0.jsonl"
        else:
            dataset_path = benchmark_dir / "combibench_lean4.15.0.jsonl"

    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset_path not found: {dataset_path}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = (
        Path(args.out_root).resolve()
        if args.out_root
        else (cwd / f"results_pilot_{args.dataset}_calls{args.max_llm_calls}_{ts}")
    )
    inputs_root = out_root / "inputs"
    runs_root = out_root / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    inputs_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    problems = load_benchmark(dataset_path, int(args.num_problems))
    manifest = {
        "created_at": ts,
        "dataset_path": str(dataset_path),
        "num_problems": len(problems),
        "baseline_mode": args.baseline_mode,
        "seed": args.seed,
        "max_llm_calls": args.max_llm_calls,
        "num_generations": args.num_generations,
        "stagnation_generations": args.stagnation_generations,
        "max_soft_resets": args.max_soft_resets,
        "init_programs_root": str(Path(args.init_programs_root).resolve()) if args.init_programs_root else "",
        "num_init_candidates_gen0": int(args.num_init_candidates_gen0),
        "llm_mode": args.llm_mode,
        "openai_llm_base_url": args.openai_llm_base_url,
        "llm_models": args.llm_models,
        "patch_openai_llm_base_url": args.patch_openai_llm_base_url,
        "patch_llm_models": args.patch_llm_models,
        "cycle_api_base_url": args.cycle_api_base_url,
        "cycle_model_name": args.cycle_model_name,
        "criticlean_base_url": args.criticlean_base_url,
        "criticlean_model": args.criticlean_model,
        "use_beq": bool(args.use_beq),
        "use_semantic": bool(args.use_semantic),
        "use_cycle_consistency": bool(args.use_cycle_consistency),
        "max_parallel_jobs_per_problem": args.max_parallel_jobs,
        "meta_rec_interval": args.meta_rec_interval,
        "problems": [],
    }

    tasks = []
    init_programs_root = Path(args.init_programs_root).resolve() if args.init_programs_root else None

    # Resolve CriticLean env vars once (shared across all problems).
    criticlean_chat_url = ""
    criticlean_model = ""
    if bool(args.use_semantic):
        # Prefer explicit base_url, otherwise fall back to env CRITIC_LEAN_URL.
        if args.criticlean_base_url:
            criticlean_chat_url = _criticlean_chat_url(args.criticlean_base_url)
            if not criticlean_chat_url:
                raise ValueError("--criticlean_base_url is set but produced an empty chat URL")
        else:
            criticlean_chat_url = str(os.environ.get("CRITIC_LEAN_URL", "") or "").strip()

        criticlean_model = str(args.criticlean_model or os.environ.get("CRITIC_LEAN_MODEL", "") or "").strip()
        if criticlean_chat_url and not criticlean_model and args.criticlean_base_url:
            # Auto-detect when the server hosts a single model (common for vLLM).
            criticlean_model = _probe_single_openai_model_id(args.criticlean_base_url, timeout_s=3.0)

        if not criticlean_chat_url:
            raise ValueError(
                "--use_semantic requires CriticLean. Provide --criticlean_base_url or set CRITIC_LEAN_URL."
            )
        if not criticlean_model:
            raise ValueError(
                "--use_semantic requires CriticLean model id. Provide --criticlean_model or set CRITIC_LEAN_MODEL."
            )

        # Record the resolved critic endpoint/model for auditability.
        manifest["criticlean_url"] = criticlean_chat_url
        manifest["criticlean_model"] = criticlean_model

    for p in problems:
        p_input_dir = inputs_root / p.problem_name
        init_programs_dir = ""
        if init_programs_root is not None:
            resolved = _resolve_seedbank_dir(init_programs_root, p.problem_name, p.index)
            init_programs_dir = str(resolved) if resolved is not None else ""
        paths = write_problem_inputs(
            p_input_dir,
            p,
            init_programs_dir=init_programs_dir,
            cycle_api_base_url=args.cycle_api_base_url,
            cycle_model_name=args.cycle_model_name,
            lean_server_url=args.lean_server_url,
            compile_timeout=args.compile_timeout,
            use_beq=args.use_beq,
            use_semantic=args.use_semantic,
            use_cycle_consistency=args.use_cycle_consistency,
        )
        results_dir = runs_root / p.problem_name
        manifest["problems"].append(
            {
                "index": p.index,
                "problem_name": p.problem_name,
                "full_name": p.full_name,
                "input_dir": str(p_input_dir),
                "results_dir": str(results_dir),
            }
        )
        tasks.append((p, paths["problem_config"], paths["initial_lean"], results_dir))

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Simple concurrency controller (keeps the script dependency-free).
    running: List[Dict[str, Any]] = []
    pending = list(tasks)
    finished: List[Dict[str, Any]] = []

    def _spawn(task) -> Dict[str, Any]:
        p, cfg_path, init_path, res_dir = task
        res_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        log_path = res_dir / "launcher.log"
        log_path.write_text(
            "\n".join(
                [
                    f"problem_name={p.problem_name}",
                    f"full_name={p.full_name}",
                    f"seed={args.seed}",
                    f"max_llm_calls={args.max_llm_calls}",
                    f"stagnation_generations={args.stagnation_generations}",
                    f"max_soft_resets={args.max_soft_resets}",
                    f"generator_base_url={args.openai_llm_base_url}",
                    f"generator_models={args.llm_models}",
                    f"patch_base_url={args.patch_openai_llm_base_url}",
                    f"patch_models={args.patch_llm_models}",
                    f"cycle_base_url={args.cycle_api_base_url}",
                    f"cycle_model={args.cycle_model_name}",
                    f"criticlean_url={criticlean_chat_url if bool(args.use_semantic) else ''}",
                    f"criticlean_model={criticlean_model if bool(args.use_semantic) else ''}",
                    f"lean_server_url={args.lean_server_url}",
                    f"use_semantic={bool(args.use_semantic)}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        run_log = res_dir / "run_evo.log"
        run_err = res_dir / "run_evo.err"
        cmd = [
            os.fspath(run_evo_py),
            "--baseline_mode",
            str(args.baseline_mode),
            "--seed",
            str(args.seed),
            "--llm_mode",
            str(args.llm_mode),
            "--openai_llm_base_url",
            str(args.openai_llm_base_url),
            "--llm_models",
            str(args.llm_models),
            "--patch_openai_llm_base_url",
            str(args.patch_openai_llm_base_url),
            "--patch_llm_models",
            str(args.patch_llm_models),
            "--cycle_api_base_url",
            str(args.cycle_api_base_url),
            "--cycle_model_name",
            str(args.cycle_model_name),
            "--meta_rec_interval",
            str(args.meta_rec_interval),
            "--stagnation_generations",
            str(args.stagnation_generations),
            "--max_soft_resets",
            str(args.max_soft_resets),
            "--max_parallel_jobs",
            str(args.max_parallel_jobs),
            "--num_init_candidates_gen0",
            str(int(args.num_init_candidates_gen0)),
            "--config",
            os.fspath(cfg_path),
            "--init_program",
            os.fspath(init_path),
            "--results_dir",
            os.fspath(res_dir),
            "--num_generations",
            str(args.num_generations),
            "--max_llm_calls",
            str(args.max_llm_calls),
            "--parent_selection_strategy",
            str(args.parent_selection_strategy),
        ]
        proc = subprocess.Popen(
            [os.fspath(sys.executable), *cmd],
            cwd=str(cwd),
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                **(
                    {"CRITIC_LEAN_URL": criticlean_chat_url, "CRITIC_LEAN_MODEL": criticlean_model}
                    if bool(args.use_semantic)
                    else {}
                ),
            },
            stdout=run_log.open("w", encoding="utf-8"),
            stderr=run_err.open("w", encoding="utf-8"),
            text=True,
        )
        return {
            "problem_name": p.problem_name,
            "full_name": p.full_name,
            "results_dir": str(res_dir),
            "pid": proc.pid,
            "proc": proc,
            "start_time": start,
        }

    try:
        while pending or running:
            while pending and len(running) < int(args.concurrency):
                running.append(_spawn(pending.pop(0)))

            time.sleep(2.0)
            still_running = []
            for r in running:
                proc = r["proc"]
                code = proc.poll()
                if code is None:
                    still_running.append(r)
                    continue
                r["returncode"] = int(code)
                r["elapsed_s"] = round(time.time() - float(r["start_time"]), 3)
                finished.append(r)
            running = still_running

            # Periodic heartbeat summary.
            (out_root / "status.json").write_text(
                json.dumps(
                    {
                        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                        "pending": len(pending),
                        "running": [
                            {
                                "problem_name": r["problem_name"],
                                "pid": r["pid"],
                                "elapsed_s": round(time.time() - float(r["start_time"]), 3),
                            }
                            for r in running
                        ],
                        "finished": [
                            {k: v for k, v in r.items() if k not in {"proc"}}
                            for r in finished
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        summary = {
            "out_root": str(out_root),
            "finished": [
                {k: v for k, v in r.items() if k not in {"proc"}}
                for r in finished
            ],
        }
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0
    finally:
        # Best-effort cleanup if the launcher itself is interrupted.
        for r in running:
            try:
                r["proc"].terminate()
            except Exception:
                pass


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
