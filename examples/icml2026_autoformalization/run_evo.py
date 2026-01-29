#!/usr/bin/env python3
"""
FormalEvolve runner for Lean4 Autoformalization.

================================================================================
                               Design Overview
================================================================================

This file is the entrypoint script for the autoformalization pipeline. It:
1. parses CLI arguments
2. creates config objects
3. launches the evolutionary search

Responsibilities across the three main files
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────┐
│   run_evo.py    │ --> │ autoformal_runner.py │ --> │ evaluate.py │
│  (entry/config) │     │   (main evolution)   │     │ (evaluator) │
└─────────────────┘     └──────────────────────┘     └─────────────┘
      ↓                          ↓                         ↓
  parse args               coordinate search          score candidates
  create configs           manage repair queue        gated fitness function
  start run                termination strategy       Lean compile check

Why a separate entrypoint?
1. Separate config from logic: experiments can change config without touching core code.
2. Convenience: quickly switch knobs via CLI.
3. Clear dependency direction: the entrypoint imports core modules, not vice versa.

Spec alignment (Constraint G)
We intentionally avoid injecting dataset headers into prompts (to reduce prompt contamination).
The task prompt is built in `build_task_sys_msg()` and asks the model to output a complete Lean 4 file.

================================================================================
                           Spec Version: v1.0 (Final)
================================================================================

Usage:
    python run_evo.py --informal "Prove that for any odd n, 8 divides n^2-1"

Constraints:
- [G] No dataset header injection in prompts
"""

import argparse
import os
import sys
from pathlib import Path

# Load .env first
from dotenv import load_dotenv
BASE_DIR = Path(__file__).parent
env_path = BASE_DIR.parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# =============================================================================
# IMPORTANT: Patch Shinka prompts BEFORE importing shinka.core
# =============================================================================
# The original `shinka/prompts` are designed for generic code generation and are a poor fit
# for Lean 4 autoformalization. We monkey-patch them so the evolution loop uses Lean 4 prompts.
from prompts_lean4 import patch_all
patch_all()

from shinka.core import EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

# Use AutoformalizationRunner with repair_queue support
from autoformal_runner import AutoformalizationRunner, RepairConfig, TerminationConfig


# =============================================================================
# Default problem (for quick tests / demos)
# =============================================================================
#
# Why keep a default problem?
# 1. Quick sanity checks: avoid passing CLI flags every time
# 2. Example/demo: documents the expected input format
# 3. Regression checks: ensures the pipeline still runs end-to-end
#
# Why this particular default?
# - Simple but non-trivial: needs Lean knowledge to formalize correctly
# - Has a clear ground truth: convenient for BEq+ checks
# - Typical theorem shape: a divisibility claim
# =============================================================================

DEFAULT_INFORMAL = """
Prove that for any odd natural number n, 8 divides n^2 - 1.
"""

DEFAULT_HEADER = """import Mathlib

open Function Fintype Subgroup Ideal Polynomial Submodule Zsqrtd RingHom
open scoped BigOperators"""
# Header notes
# - import Mathlib: import the full Mathlib library (definitions + theorems)
# - open xxx: open namespaces so you can use definitions without prefixes
# - open scoped BigOperators: enable ∑ and ∏ notations

DEFAULT_GROUND_TRUTH = """
theorem odd_sq_sub_one_div_eight (n : ℕ) (hn : Odd n) : 8 ∣ n^2 - 1 := by sorry
"""
# Ground-truth notes
# - theorem xxx: theorem name (convention: descriptive, snake_case)
# - (n : ℕ): explicit parameter, a natural number n
# - (hn : Odd n): hypothesis, a proof that n is odd
# - 8 ∣ n^2 - 1: conclusion, 8 divides n²-1
# - := by sorry: proof placeholder (we only care about the statement)


# =============================================================================
# Task System Message（no dataset header）
# =============================================================================
#
# NOTE:
# - To avoid dataset-header contamination, we do not inject dataset headers into prompts.
# - The model still must output a *complete Lean file* (imports + theorem) so we can compile/evaluate it.
# =============================================================================

def build_task_sys_msg(informal: str, header: str) -> str:
    """
    Build the LLM task system message.

    NOTE:
    - The `header` parameter is kept only for backward compatibility (older call sites still pass it).
    - To avoid contamination, we do not inject dataset headers into the prompt.

    Prompt structure
    1. Role: Lean 4 expert
    2. Task: natural language → Lean 4 theorem statement
    3. Output protocol:
       - output a complete Lean file (imports + theorem)
       - end the theorem with `:= by sorry`

    Args:
        informal: natural-language mathematical statement
        header: Lean 4 header (imports/opens/options); currently unused

    Returns:
        Full system-message string
    """
    return f"""You are an expert in mathematics and Lean 4 (Mathlib).

Given a problem statement in natural language, write a COMPLETE Lean 4 file (imports + theorem)
that formalizes the mathematical content.

MANDATORY OUTPUT REQUIREMENTS:
- Output EXACTLY ONE Lean code block: ```lean ... ```.
- The code MUST start with the following two lines (in this order):
  import Mathlib
  import Aesop
- You MAY add additional `import ...` / `open ...` / `set_option ...` lines after that if needed.
- Do NOT include any comments.
- Include EXACTLY ONE `theorem` declaration.
- The theorem MUST end with `:= by sorry` (do not provide a proof).
- Use Lean 4 v4.15-compatible syntax and Mathlib definitions.

Natural language statement:
{informal.strip()}

Return ONLY the Lean code block."""
# =============================================================================
# Config construction
# =============================================================================
#
# Config hierarchy
# ShinkaEvolve uses multiple config objects to control different aspects:
#
# ┌─────────────────┐
# │  EvolutionConfig │ ← evolution algorithm params (gens, parallelism, LLM settings)
# └─────────────────┘
# ┌─────────────────┐
# │    JobConfig     │ ← job execution (evaluator entrypoint path)
# └─────────────────┘
# ┌─────────────────┐
# │  DatabaseConfig  │ ← archive DB config (size, sampling)
# └─────────────────┘
# ┌─────────────────┐
# │   RepairConfig   │ ← compile-repair config (attempts, temperature)
# └─────────────────┘
# ┌─────────────────┐
# │  ProblemConfig   │ ← problem description (for evaluator)
# └─────────────────┘
#
# Why so many configs?
# 1. Separation of concerns: each config controls one dimension
# 2. Reasonable defaults: most runs shouldn't need to tweak everything
# 3. Extensibility: adding new knobs doesn't break existing code
# =============================================================================

def create_configs(
    informal: str,
    header: str,
    ground_truth: str,
    use_beq: bool,
    results_dir: str,
    problem_overrides: dict | None = None,
    baseline_mode: str = "ours",
    seed: int | None = None,
    llm_mode: str = "auto",
    replay_path: str | None = None,
    mock_statements_path: str | None = None,
    no_llm: bool = False,
    embedding_model: str | None = None,
    code_embed_sim_threshold: float = 0.995,
    max_novelty_attempts: int = 3,
    novelty_llm_models: list[str] | None = None,
    novelty_llm_disabled: bool = False,
    num_init_candidates_gen0: int = 3,
    # Parent sampling: align with circle_packing_backup defaults (combined_score-driven).
    parent_selection_strategy: str = "weighted",
    num_archive_inspirations: int = 4,
    num_top_k_inspirations: int = 2,
    patch_types: list[str] | None = None,
    patch_type_probs: list[float] | None = None,
    num_generations: int = 400,
    max_parallel_jobs: int = 5,
    max_repair_attempts: int = 2,
    max_llm_calls: int | None = None,
    max_evals: int | None = None,
    max_time_seconds: float | None = None,
    stagnation_generations: int = 5,
    max_soft_resets: int = 3,
    init_program_path: str | None = None,
):
    """
    Create all config objects required by ShinkaEvolve.

    Config objects
    - evo_config: core evolution-algorithm knobs
    - job_config: how evaluation jobs are executed
    - db_config: archive database behavior
    - repair_config: compile-repair mechanism
    - problem_config: problem description (consumed by the evaluator)

    Args (high-level)
    - informal: natural-language statement to formalize
    - header: Lean 4 header (imports/opens/options)
    - ground_truth: optional reference statement (for BEq+)
    - use_beq: enable BEq+ equivalence check
    - results_dir: output directory for this run
    - num_generations: number of evolution generations
    - max_repair_attempts: max compile-repair attempts

    Returns:
        (evo_config, job_config, db_config, repair_config, problem_config) tuple
    """
    import json

    baseline_mode_str = str(baseline_mode or "ours").strip()
    baseline_mode_norm = baseline_mode_str.lower()
    is_baseline_mode = baseline_mode_norm != "ours"

    # =========================================================================
    # 1. Problem Config
    # =========================================================================
    # This config is written to disk so `evaluate.py` (a subprocess) can load it.
    #
    # Why write to a file?
    # `evaluate.py` is executed as a separate process; we cannot pass a rich object graph
    # via function args, so file-based communication is the simplest option.
    problem_config = {
        "informal": informal,      # natural-language statement
        "header": header,          # Lean4 header（import/open）
        "ground_truth": ground_truth,  # reference statement (for BEq+)
        "use_beq": use_beq,        # enable BEq+ check
        # Optional: a seed bank directory containing gen_0/seed_i/ programs.
        # When provided, AutoformalizationRunner will reuse these initial programs.
        "init_programs_dir": "",
        # --- experiment protocol (for auditability) ---
        "baseline_mode": baseline_mode_str,
        "seed": seed,
        # --- no-LLM / replay metadata (for audit & evaluator behavior) ---
        # llm_mode affects the generator/repair side; no_llm affects evaluator (cycle/critic).
        "llm_mode": str(llm_mode or "auto"),
        "no_llm": bool(no_llm),
        "replay_path": replay_path,
        "mock_statements_path": mock_statements_path,
        # Record the base URL used for generation (for post-run audits).
        "openai_llm_base_url": os.environ.get("OPENAI_LLM_BASE_URL", ""),
        # Default: disable semantic 0/1 judge (noisy)
        "use_semantic": False,
        # Default: disable cycle-consistency (noisy / service-dependent)
        "use_cycle_consistency": False,
        # Cycle-consistency model config (OpenAI-compatible)
        "cycle_api_base_url": os.environ.get("CYCLE_API_BASE_URL", "http://127.0.0.1:8090/v1"),
        "cycle_model_name": os.environ.get("CYCLE_MODEL_NAME", "Qwen2.5-32B-Instruct"),
        "informalize_prompt_template": os.environ.get("INFORMALIZE_PROMPT_TEMPLATE", "Informalize: {formal_statement}"),
        # Reduce evaluation variance from semantically irrelevant declaration naming noise.
        "cycle_normalize_decl_name": os.environ.get("AUTOFORMAL_CYCLE_NORMALIZE_DECL_NAME", "true").lower()
        in {"1", "true", "yes", "y", "on"},
        "cycle_normalized_decl_name": os.environ.get("AUTOFORMAL_CYCLE_NORMALIZED_DECL_NAME", "my_theorem"),
        # Future-proof: if semantic / BEq are enabled, normalize decl names too.
        "semantic_normalize_decl_name": os.environ.get("AUTOFORMAL_SEMANTIC_NORMALIZE_DECL_NAME", "true").lower()
        in {"1", "true", "yes", "y", "on"},
        "semantic_normalized_decl_name": os.environ.get("AUTOFORMAL_SEMANTIC_NORMALIZED_DECL_NAME", "my_theorem"),
        "beq_normalize_decl_name": os.environ.get("AUTOFORMAL_BEQ_NORMALIZE_DECL_NAME", "true").lower()
        in {"1", "true", "yes", "y", "on"},
        "beq_candidate_decl_name": os.environ.get("AUTOFORMAL_BEQ_CANDIDATE_DECL_NAME", "my_cand"),
        "beq_ground_truth_decl_name": os.environ.get("AUTOFORMAL_BEQ_GROUND_TRUTH_DECL_NAME", "my_gt"),
        "cycle_softmax_temperature": float(os.environ.get("SOFTMAX_TEMPERATURE", "3.5")),
        "cycle_temperature": float(os.environ.get("CYCLE_TEMPERATURE", "0.0")),
        "cycle_max_tokens": int(os.environ.get("CYCLE_MAX_TOKENS", "1024")),
        # --- scoring (for auditability; evaluator uses these weights) ---
        # combined_score = compile_ok * (base + cycle_w*cycle + semantic_bonus*semantic_ok + beq_bonus*beq_ok)
        # Defaults are chosen to ensure: beq > semantic > cycle (even if semantic is noisy).
        "score_base": float(os.environ.get("AUTOFORMAL_SCORE_BASE", "100")),
        "score_cycle_weight": float(os.environ.get("AUTOFORMAL_SCORE_CYCLE_WEIGHT", "50")),
        "score_semantic_bonus": float(os.environ.get("AUTOFORMAL_SCORE_SEMANTIC_BONUS", "100")),
        "score_beq_bonus": float(os.environ.get("AUTOFORMAL_SCORE_BEQ_BONUS", "200")),
        "compile_timeout": 600,     # compile timeout (seconds) - 10 minutes for harder theorems
        # Lean Server HTTP endpoint (Kimina Lean Server docs: http://localhost:8001/docs)
        "lean_server_url": os.environ.get(
            "LEAN_SERVER_URL", "local"
        ),
    }

    # Allow --config / external problem configs to override evaluator settings.
    if problem_overrides:
        allowed_override_keys = {
            "use_beq",
            "use_semantic",
            "use_cycle_consistency",
            "compile_timeout",
            "cycle_api_base_url",
            "cycle_model_name",
            "informalize_prompt_template",
            "cycle_normalize_decl_name",
            "cycle_normalized_decl_name",
            "semantic_normalize_decl_name",
            "semantic_normalized_decl_name",
            "beq_normalize_decl_name",
            "beq_candidate_decl_name",
            "beq_ground_truth_decl_name",
            "cycle_softmax_temperature",
            "cycle_temperature",
            "cycle_max_tokens",
            "score_base",
            "score_cycle_weight",
            "score_semantic_bonus",
            "score_beq_bonus",
            "lean_server_url",
            "no_llm",
            "init_programs_dir",
        }
        for k in allowed_override_keys:
            if k in problem_overrides:
                problem_config[k] = problem_overrides[k]

    # Write per-run problem config into the results directory to support parallel runs.
    run_root = Path(results_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    config_path = run_root / "problem_config.json"
    with open(config_path, "w") as f:
        json.dump(problem_config, f, indent=2, ensure_ascii=False)
    print(f"[Config] Problem config saved to {config_path}")

    # Optional legacy write for single-run workflows (NOT recommended for parallel runs).
    if os.environ.get("AUTOFORMAL_WRITE_GLOBAL_PROBLEM_CONFIG", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        legacy_path = BASE_DIR / "problem_config.json"
        with open(legacy_path, "w") as f:
            json.dump(problem_config, f, indent=2, ensure_ascii=False)
        print(f"[Config] (Legacy) Problem config also saved to {legacy_path}")

    # =========================================================================
    # 2. Repair Config
    # =========================================================================
    # Temperature protocol (repair):
    # - Compile-repair is prone to "no-op / echo" at temperature=0.0.
    # - Default to a higher fixed temperature to reduce duplicate attempts.
    repair_temperature = float(os.environ.get("AUTOFORMAL_REPAIR_TEMPERATURE", "0.7"))
    repair_config = RepairConfig(
        num_init_candidates_gen0=int(num_init_candidates_gen0),
        max_repair_attempts=max_repair_attempts,
        repair_temperature=repair_temperature,
        enabled=True,
    )

    # =========================================================================
    # 3. Job Config
    # =========================================================================
    # Path to the evaluator entrypoint. ShinkaEvolve calls:
    #   python evaluate.py --program_path xxx --results_dir xxx
    job_config = LocalJobConfig(
        eval_program_path=str(BASE_DIR / "evaluate.py"),
    )

    # =========================================================================
    # 4. Database Config
    # =========================================================================
    # Controls archive behavior.
    #
    # Key params
    # - num_islands: number of islands (parallel sub-populations exploring different directions)
    # - archive_size: global archive cap (shared across islands; not "per-island")
    # - elite_selection_ratio: elite selection ratio (for inspiration sampling)
    # - num_archive_inspirations: inspirations sampled from archive
    # - num_top_k_inspirations: inspirations sampled from top-k
    # - parent_selection_strategy: parent selection strategy
    #   - "weighted": fitness-weighted sampling (better candidates are more likely)
    #   - "uniform": uniform sampling
    # - parent_selection_lambda: temperature for weighted sampling
    #
    # Notes:
    # - Although there are multiple islands logically (`programs.island_idx`), the current archive
    #   implementation is a single shared table. `archive_size` caps the *global* number of retained
    #   correct programs, not per-island retention.
    # - Island imbalance can happen (some problems may even have empty archives for some islands).
    #
    # IMPORTANT (for ablations):
    # - Allow env overrides in ours-mode so we can run `num_islands=1 vs 2` without touching code.
    # - Baseline modes keep their semantics (no islands / archive disabled).
    def _cfg_int_env(name: str, default: int, *, min_value: int = 1) -> int:
        raw = str(os.environ.get(name, "")).strip()
        if not raw:
            return int(default)
        try:
            v = int(raw)
        except Exception:
            return int(default)
        return max(int(min_value), int(v))

    def _cfg_float_env(
        name: str, default: float, *, min_value: float | None = None
    ) -> float:
        raw = str(os.environ.get(name, "")).strip()
        if not raw:
            return float(default)
        try:
            v = float(raw)
        except Exception:
            return float(default)
        if min_value is not None:
            v = max(float(min_value), float(v))
        return float(v)

    def _cfg_bool_env(name: str, default: bool) -> bool:
        raw = str(os.environ.get(name, "")).strip().lower()
        if raw in {"1", "true", "yes", "y", "on"}:
            return True
        if raw in {"0", "false", "no", "n", "off"}:
            return False
        return bool(default)

    default_num_islands = 1 if is_baseline_mode else 2
    default_archive_size = 0 if is_baseline_mode else 40
    default_migration_interval = 10
    default_migration_rate = 0.1
    default_enforce_island_separation = True

    num_islands_eff = (
        default_num_islands
        if is_baseline_mode
        else _cfg_int_env("AUTOFORMAL_NUM_ISLANDS", default_num_islands, min_value=1)
    )
    archive_size_eff = (
        default_archive_size
        if is_baseline_mode
        else _cfg_int_env("AUTOFORMAL_ARCHIVE_SIZE", default_archive_size, min_value=0)
    )
    migration_interval_eff = _cfg_int_env(
        "AUTOFORMAL_MIGRATION_INTERVAL", default_migration_interval, min_value=1
    )
    migration_rate_eff = _cfg_float_env(
        "AUTOFORMAL_MIGRATION_RATE", default_migration_rate, min_value=0.0
    )
    enforce_island_separation_eff = _cfg_bool_env(
        "AUTOFORMAL_ENFORCE_ISLAND_SEPARATION", default_enforce_island_separation
    )
    db_config = DatabaseConfig(
        db_path="evolution_db.sqlite",
        # Baseline modes do not use islands for sampling; avoid initial-program auto-copy.
        num_islands=int(num_islands_eff),
        # Baseline modes do not use archive/parent sampling; force-disable archive updates.
        archive_size=int(archive_size_eff),
        elite_selection_ratio=0.3,
        num_archive_inspirations=int(num_archive_inspirations),
        num_top_k_inspirations=int(num_top_k_inspirations),
        migration_interval=int(migration_interval_eff),
        migration_rate=float(migration_rate_eff),
        island_elitism=True,
        enforce_island_separation=bool(enforce_island_separation_eff),
        parent_selection_strategy=str(parent_selection_strategy),
        cycle_softmax_temperature=float(os.environ.get("SOFTMAX_TEMPERATURE", "3.5")),
        parent_usage_penalty_alpha=float(
            os.environ.get("PARENT_USAGE_PENALTY_ALPHA", "0.05")
        ),
        parent_selection_lambda=_cfg_float_env("AUTOFORMAL_PARENT_SELECTION_LAMBDA", 10.0, min_value=0.0),
    )

    # =========================================================================
    # 5. Evolution Config
    # =========================================================================
    # This is the core config that controls the evolution algorithm.
    task_sys_msg = build_task_sys_msg(informal, header)
    meta_interval = int(os.environ.get("META_REC_INTERVAL", "10"))
    meta_rec_interval = meta_interval if meta_interval > 0 else None

    llm_models_list = [m.strip() for m in (os.environ.get("AUTOFORMAL_LLM_MODELS") or "").split(",") if m.strip()]
    # Normalize local path-like model names: vLLM/OpenAI-compatible servers typically register the directory path
    # without a trailing slash, so keep ids stable by stripping it.
    llm_models_list = [m.rstrip("/") for m in llm_models_list]
    if not llm_models_list:
        # Default: a reasonable placeholder name (users should override via --llm_models).
        # We intentionally avoid hard-coding any local filesystem paths here.
        llm_models_list = ["Kimina-Autoformalizer-7B"]

    # Disable meta-LLM in offline/no-LLM modes to keep the smoke pipeline deterministic.
    offline_mode = str(llm_mode or "").lower() in {"mock", "replay"}
    if offline_mode or no_llm:
        meta_rec_interval = None
    # Baseline modes should not use meta-LLM (keep budgets comparable).
    if is_baseline_mode:
        meta_rec_interval = None

    # Novelty filtering defaults:
    # - Off by default for Lean4 autoformalization (set embedding_model to enable).
    # - Novelty LLM is optional; if None, high-similarity samples are rejected by embedding threshold alone.
    embedding_model_eff = str(embedding_model or "").strip()
    if embedding_model_eff.lower() in {"", "none", "null"}:
        embedding_model_eff = ""
    if offline_mode or no_llm:
        # Keep offline/no-LLM modes deterministic and free of hidden dependencies by default.
        embedding_model_eff = ""
    novelty_llm_models_eff: list[str] | None = novelty_llm_models
    if embedding_model_eff == "":
        novelty_llm_models_eff = None
    if offline_mode or no_llm:
        novelty_llm_models_eff = None
    # Explicit opt-out: `--novelty_llm_models none` should disable novelty-LLM even in ours mode.
    if novelty_llm_disabled:
        novelty_llm_models_eff = None
    if (
        embedding_model_eff != ""
        and novelty_llm_models_eff is None
        and not (offline_mode or no_llm)
        and not novelty_llm_disabled
    ):
        # Default: use the generator models as the novelty judge LLM.
        # Baselines keep novelty-LLM disabled by default to avoid introducing an extra judge unless requested.
        novelty_llm_models_eff = None if is_baseline_mode else llm_models_list

    patch_types_eff = patch_types or ["full", "diff", "cross"]
    patch_type_probs_eff = patch_type_probs or [0.5, 0.3, 0.2]
    if len(patch_types_eff) != len(patch_type_probs_eff):
        raise ValueError(
            f"patch_types length {len(patch_types_eff)} != patch_type_probs length {len(patch_type_probs_eff)}"
        )
    if not patch_types_eff:
        raise ValueError("patch_types must be non-empty")
    s = float(sum(patch_type_probs_eff))
    if s <= 0:
        raise ValueError("patch_type_probs must sum to a positive value")
    patch_type_probs_eff = [float(p) / s for p in patch_type_probs_eff]

    # Temperature schedule (Shinka-aligned multi-temperature sampling).
    # Format: comma-separated floats, e.g. "0,0.5,1.0".
    temps_raw = str(os.environ.get("AUTOFORMAL_TEMPERATURES", "0,0.5,1.0")).strip()
    temperatures: list[float] = []
    for part in temps_raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            temperatures.append(float(part))
        except Exception:
            raise ValueError(f"Invalid float in AUTOFORMAL_TEMPERATURES: {part!r}")
    if not temperatures:
        raise ValueError("AUTOFORMAL_TEMPERATURES must contain at least one float value")

    def _cfg_int(name: str, default: int, *, min_value: int = 1) -> int:
        raw = str(os.environ.get(name, "")).strip()
        if not raw:
            return int(default)
        try:
            v = int(raw)
        except Exception:
            return int(default)
        return max(int(min_value), int(v))

    evo_config = EvolutionConfig(
        # --- Prompt config ---
        task_sys_msg=task_sys_msg,  # LLM system message

        # --- Patch-type config ---
        # Patch types:
        # - "full": rewrite from scratch
        # - "diff": local edit based on parent
        # - "cross": crossover (combine multiple parents)
        patch_types=patch_types_eff,
        patch_type_probs=patch_type_probs_eff,

        # --- Evolution parameters ---
        num_generations=num_generations,
        max_parallel_jobs=int(max_parallel_jobs),  # parallel jobs within a single problem run
        max_patch_resamples=_cfg_int("AUTOFORMAL_MAX_PATCH_RESAMPLES", 3),  # resamples if generation fails
        # Paper-aligned default: one patch attempt per step (can be overridden via env).
        max_patch_attempts=_cfg_int("AUTOFORMAL_MAX_PATCH_ATTEMPTS", 1),  # max attempts per patch

        # --- Job type ---
        job_type="local",           # local execution (vs. distributed)
        language="lean",            # code language (LLM output)

        # --- LLM config ---
        # Uses the framework's OpenAI(-compatible) model interface (see shinka/llm/models/pricing.py).
        llm_models=llm_models_list,
        # Evolution multi-temperature (sampled per call).
        llm_kwargs=dict(
            temperatures=temperatures,
            # Some OpenAI-compatible servers reject large `max_tokens` (e.g., 4096 may 400/500).
            # Keep a conservative default and allow override via env for different backends.
            max_tokens=int(os.environ.get("AUTOFORMAL_LLM_MAX_TOKENS", "2048")),
        ),

        # --- Advanced features ---
        # meta_rec_interval: meta-recommendation refresh interval (triggered by evaluated program count).
        # Enabled by default and kept short (Lean statements benefit from short, concrete rules).
        meta_rec_interval=meta_rec_interval,
        # IMPORTANT: if meta_rec_interval is disabled (None), also disable meta-LLM entirely.
        # Otherwise MetaSummarizer will accumulate all programs and try a huge "final summary"
        # that can exceed the model context window and stall the run at shutdown.
        meta_llm_models=None if (offline_mode or no_llm or meta_rec_interval is None) else llm_models_list,
        meta_llm_kwargs=dict(
            temperatures=[0.0],
            max_tokens=int(os.environ.get("AUTOFORMAL_META_LLM_MAX_TOKENS", os.environ.get("AUTOFORMAL_LLM_MAX_TOKENS", "2048"))),
        ),
        llm_dynamic_selection="ucb1",
        llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),

        # code_embed_sim_threshold: novelty threshold
        code_embed_sim_threshold=float(code_embed_sim_threshold),
        embedding_model=(embedding_model_eff or None),
        max_novelty_attempts=int(max_novelty_attempts),
        novelty_llm_models=novelty_llm_models_eff,
        novelty_llm_kwargs=dict(
            temperatures=[0.0],
            max_tokens=int(os.environ.get("AUTOFORMAL_NOVELTY_LLM_MAX_TOKENS", os.environ.get("AUTOFORMAL_LLM_MAX_TOKENS", "2048"))),
        ),

        # --- File paths ---
        # Baseline modes should generate independent samples (ignore seed0 file).
        init_program_path=(None if is_baseline_mode else (init_program_path or str(BASE_DIR / "initial.lean"))),
        results_dir=results_dir,
    )

    termination_config = TerminationConfig(
        max_llm_calls=max_llm_calls,
        max_evals=max_evals,
        max_time_seconds=max_time_seconds,
        stagnation_generations=stagnation_generations,
        max_soft_resets=max_soft_resets,
    )

    return evo_config, job_config, db_config, repair_config, termination_config, problem_config


# =============================================================================
# Main
# =============================================================================
#
# Execution flow
# 1. Parse CLI args
# 2. Print run info (for debugging/auditing)
# 3. Create configs
# 4. Create AutoformalizationRunner
# 5. Run evolution
# 6. Print stats
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run FormalEvolve for Lean4 autoformalization"
    )
    parser.add_argument(
        "--baseline_mode",
        type=str,
        default=os.environ.get("AUTOFORMAL_BASELINE_MODE", "ours"),
        choices=["ours", "batchN", "repairloop1"],
        help="Experiment protocol mode: ours (SPEC 3.3), batchN (N independent samples), repairloop1 (single repair trajectory)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for python/numpy (for auditability; LLM may still be nondeterministic)",
    )
    parser.add_argument(
        "--max_parallel_jobs",
        type=int,
        default=int(os.environ.get("AUTOFORMAL_MAX_PARALLEL_JOBS", "5")),
        help="Max parallel evaluation jobs within a single problem run (default 5). "
             "If you run many problems concurrently, keep this small to avoid overloading Lean/cycle servers.",
    )
    parser.add_argument(
        "--openai_llm_base_url",
        type=str,
        default=os.environ.get("OPENAI_LLM_BASE_URL", ""),
        help="Optional OpenAI-compatible base URL for LLM generation (e.g., http://127.0.0.1:8000/v1 for vLLM). "
             "Leave empty to use the OpenAI SDK default endpoint.",
    )
    parser.add_argument(
        "--openai_embed_base_url",
        type=str,
        default=os.environ.get("OPENAI_EMBED_BASE_URL", os.environ.get("OPENAI_BASE_URL", "")),
        help="OpenAI-compatible base URL for embeddings (leave empty to use OpenAI SDK default/env)",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=os.environ.get("AUTOFORMAL_EMBEDDING_MODEL", ""),
        help="Enable novelty filtering by providing an embedding model name. "
             "Supports OpenAI-compatible endpoints (use --openai_embed_base_url) "
             "or local HuggingFace encoders via 'hf:<model_or_path>'. Empty disables.",
    )
    parser.add_argument(
        "--code_embed_sim_threshold",
        type=float,
        default=float(os.environ.get("AUTOFORMAL_CODE_EMBED_SIM_THRESHOLD", "0.995")),
        help="Novelty filtering threshold: if max cosine similarity > threshold, trigger rejection sampling "
             "and (optionally) LLM novelty judge. Default 0.995 is conservative for short Lean statements.",
    )
    parser.add_argument(
        "--max_novelty_attempts",
        type=int,
        default=int(os.environ.get("AUTOFORMAL_MAX_NOVELTY_ATTEMPTS", "3")),
        help="Novelty rejection-sampling attempts per generation step (default 3).",
    )
    parser.add_argument(
        "--novelty_llm_models",
        type=str,
        default=os.environ.get("AUTOFORMAL_NOVELTY_LLM_MODELS", ""),
        help="Comma-separated model names for the LLM-as-novelty-judge. "
             "Empty uses the generator models (except baselines); set 'none' to disable.",
    )
    parser.add_argument(
        "--novelty_llm_base_url",
        type=str,
        default=os.environ.get("OPENAI_NOVELTY_LLM_BASE_URL", ""),
        help="OpenAI-compatible base URL for the novelty judge LLM (defaults to OPENAI_LLM_BASE_URL when empty). "
             "Example: http://127.0.0.1:8090/v1",
    )
    parser.add_argument(
        "--llm_models",
        type=str,
        default=os.environ.get("AUTOFORMAL_LLM_MODELS", "Kimina-Autoformalizer-7B"),
        help="Comma-separated model names for generation (override this for your provider/server).",
    )
    parser.add_argument(
        "--patch_openai_llm_base_url",
        type=str,
        default=os.environ.get("AUTOFORMAL_PATCH_OPENAI_LLM_BASE_URL", ""),
        help="Optional OpenAI-compatible base URL for PATCH/edit proposals. "
             "Leave empty to reuse OPENAI_LLM_BASE_URL / OpenAI SDK default.",
    )
    parser.add_argument(
        "--patch_llm_models",
        type=str,
        default=os.environ.get("AUTOFORMAL_PATCH_LLM_MODELS", ""),
        help="Comma-separated model names for PATCH/edit proposals (e.g., a patch-tuned model). "
             "Empty means reuse --llm_models.",
    )
    parser.add_argument(
        "--cycle_api_base_url",
        type=str,
        default=os.environ.get("CYCLE_API_BASE_URL", ""),
        help="Optional OpenAI-compatible base URL for cycle-consistency logprob (/completions). "
             "Leave empty to disable cycle-consistency unless set in problem_config.json.",
    )
    parser.add_argument(
        "--cycle_model_name",
        type=str,
        default=os.environ.get("CYCLE_MODEL_NAME", "Qwen2.5-32B-Instruct"),
        help="Model name for cycle-consistency scoring",
    )
    parser.add_argument(
        "--softmax_temperature",
        type=float,
        default=float(os.environ.get("SOFTMAX_TEMPERATURE", "3.5")),
        help="Softmax temperature used to convert log-probs to a smooth score (paper-aligned default 3.5)",
    )
    parser.add_argument(
        "--informal",
        type=str,
        default=DEFAULT_INFORMAL,
        help="Informal mathematical statement to formalize",
    )
    parser.add_argument(
        "--header",
        type=str,
        default=DEFAULT_HEADER,
        help="Lean4 header/imports",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=DEFAULT_GROUND_TRUTH,
        help="Ground truth Lean4 statement (for BEq+)",
    )
    parser.add_argument(
        "--use_beq",
        action="store_true",
        help="Enable BEq+ equivalence checking",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=400,
        help="Number of evolution generations",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_autoformal",
        help="Directory for results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--max_repair_attempts",
        type=int,
        default=2,
        help="Maximum compile repair attempts for gen>=1",
    )
    parser.add_argument(
        "--max_repair_attempts_gen0",
        type=int,
        default=2,
        help="Maximum compile repair attempts for generation 0 (bootstrap)",
    )
    parser.add_argument(
        "--num_init_candidates_gen0",
        type=int,
        default=int(os.environ.get("AUTOFORMAL_NUM_INIT_CANDIDATES_GEN0", "3")),
        help="Number of initial candidates at generation 0 (seed_0 uses init_program; others use LLM)",
    )
    parser.add_argument(
        "--disable_repair",
        action="store_true",
        help="Disable compile repair",
    )
    parser.add_argument(
        "--max_llm_calls",
        type=int,
        default=None,
        help="Budget: max LLM calls (hard stop)",
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=None,
        help="Budget: max evaluations (hard stop)",
    )
    parser.add_argument(
        "--max_time_seconds",
        type=float,
        default=None,
        help="Budget: max runtime in seconds (hard stop)",
    )
    parser.add_argument(
        "--stagnation_generations",
        type=int,
        default=5,
        help="Generations without improvement before soft reset",
    )
    parser.add_argument(
        "--max_soft_resets",
        type=int,
        default=3,
        help="Max number of soft resets before stopping",
    )
    parser.add_argument(
        "--meta_rec_interval",
        type=int,
        default=int(os.environ.get("META_REC_INTERVAL", "10")),
        help="Meta update interval (by evaluated programs); set 0 to disable",
    )
    parser.add_argument(
        "--parent_penalty_alpha",
        type=float,
        default=float(os.environ.get("PARENT_USAGE_PENALTY_ALPHA", "0.05")),
        help="Parent usage penalty alpha for cycle-softmax (0 disables)",
    )
    parser.add_argument(
        "--cross_k",
        type=int,
        default=int(os.environ.get("AUTOFORMAL_CROSS_K", "1")),
        help="Number of inspirations to include in cross prompts (default 1)",
    )
    parser.add_argument(
        "--cross_insp_penalty_alpha",
        type=float,
        default=float(os.environ.get("AUTOFORMAL_CROSS_INSP_PENALTY_ALPHA", "1.0")),
        help="DEPRECATED (ignored): cross inspirations are sampled uniformly; kept for backward compatibility.",
    )
    parser.add_argument(
        "--cross_insp_penalty_window",
        type=int,
        default=int(os.environ.get("AUTOFORMAL_CROSS_INSP_PENALTY_WINDOW", "50")),
        help="DEPRECATED (ignored): cross inspirations are sampled uniformly; kept for backward compatibility.",
    )
    parser.add_argument(
        "--cross_insp_temperature",
        type=float,
        default=float(os.environ.get("AUTOFORMAL_CROSS_INSP_TEMPERATURE", os.environ.get("SOFTMAX_TEMPERATURE", "3.5"))),
        help="DEPRECATED (ignored): cross inspirations are sampled uniformly; kept for backward compatibility.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to problem config JSON file (overrides --informal, --header, --ground_truth)",
    )
    parser.add_argument(
        "--init_program",
        type=str,
        default=None,
        help="Path to initial program file (overrides default initial.lean)",
    )
    parser.add_argument(
        "--parent_selection_strategy",
        type=str,
        default=os.environ.get("AUTOFORMAL_PARENT_SELECTION_STRATEGY", "weighted"),
        choices=["cycle_softmax", "best_of_n", "weighted", "power_law", "beam_search"],
        help="Parent selection strategy (baseline comparisons need best_of_n / weighted / etc.)",
    )
    parser.add_argument(
        "--num_archive_inspirations",
        type=int,
        default=int(os.environ.get("AUTOFORMAL_NUM_ARCHIVE_INSP", "4")),
        help="Number of archive inspirations to include in evolution prompts",
    )
    parser.add_argument(
        "--num_top_k_inspirations",
        type=int,
        default=int(os.environ.get("AUTOFORMAL_NUM_TOPK_INSP", "2")),
        help="Number of top-k inspirations to include in evolution prompts",
    )
    parser.add_argument(
        "--patch_types",
        type=str,
        default=os.environ.get("AUTOFORMAL_PATCH_TYPES", ""),
        help="Comma-separated patch types (subset of full,diff,cross). Empty uses defaults.",
    )
    parser.add_argument(
        "--patch_type_probs",
        type=str,
        default=os.environ.get("AUTOFORMAL_PATCH_TYPE_PROBS", ""),
        help="Comma-separated probabilities aligned with --patch_types. Empty uses defaults.",
    )
    parser.add_argument(
        "--llm_mode",
        type=str,
        default=os.environ.get("AUTOFORMAL_LLM_MODE", "auto"),
        choices=["auto", "real", "mock", "replay"],
        help="LLM mode: auto (probe+fallback), real, mock, replay",
    )
    parser.add_argument(
        "--replay_path",
        type=str,
        default=os.environ.get("AUTOFORMAL_REPLAY_PATH", None),
        help="Replay JSONL path (used when --llm_mode=replay)",
    )
    parser.add_argument(
        "--mock_statements_path",
        type=str,
        default=os.environ.get(
            "AUTOFORMAL_MOCK_STATEMENTS_PATH",
            str(BASE_DIR / "fixtures" / "mock_statements.json"),
        ),
        help="Mock statements path (.json list or text lines) for MockLLM",
    )
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="Disable ALL LLM usage (generator uses MockLLM; evaluator cycle/critic use stubs).",
    )

    args = parser.parse_args()

    # Seed RNGs early for auditability (does not guarantee deterministic LLM outputs).
    if args.seed is not None:
        import random
        import numpy as np

        random.seed(int(args.seed))
        np.random.seed(int(args.seed))

    # Configure OpenAI-compatible endpoints for this run (generation vs embeddings).
    if args.openai_llm_base_url:
        os.environ["OPENAI_LLM_BASE_URL"] = args.openai_llm_base_url
        os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY"))
    if args.openai_embed_base_url:
        os.environ["OPENAI_EMBED_BASE_URL"] = args.openai_embed_base_url
    if args.novelty_llm_base_url:
        os.environ["OPENAI_NOVELTY_LLM_BASE_URL"] = args.novelty_llm_base_url
    else:
        os.environ.pop("OPENAI_NOVELTY_LLM_BASE_URL", None)

    # Models list for EvolutionConfig is read from env inside create_configs().
    os.environ["AUTOFORMAL_LLM_MODELS"] = args.llm_models
    if args.patch_openai_llm_base_url:
        os.environ["AUTOFORMAL_PATCH_OPENAI_LLM_BASE_URL"] = args.patch_openai_llm_base_url
    else:
        os.environ.pop("AUTOFORMAL_PATCH_OPENAI_LLM_BASE_URL", None)
    if args.patch_llm_models:
        os.environ["AUTOFORMAL_PATCH_LLM_MODELS"] = args.patch_llm_models
    else:
        os.environ.pop("AUTOFORMAL_PATCH_LLM_MODELS", None)
    os.environ["CYCLE_API_BASE_URL"] = args.cycle_api_base_url
    os.environ["CYCLE_MODEL_NAME"] = args.cycle_model_name
    os.environ["SOFTMAX_TEMPERATURE"] = str(args.softmax_temperature)
    os.environ["META_REC_INTERVAL"] = str(args.meta_rec_interval)
    os.environ["PARENT_USAGE_PENALTY_ALPHA"] = str(args.parent_penalty_alpha)
    os.environ["AUTOFORMAL_CROSS_K"] = str(args.cross_k)
    os.environ["AUTOFORMAL_CROSS_INSP_PENALTY_ALPHA"] = str(args.cross_insp_penalty_alpha)
    os.environ["AUTOFORMAL_CROSS_INSP_PENALTY_WINDOW"] = str(args.cross_insp_penalty_window)
    os.environ["AUTOFORMAL_CROSS_INSP_TEMPERATURE"] = str(args.cross_insp_temperature)
    os.environ["AUTOFORMAL_PARENT_SELECTION_STRATEGY"] = str(args.parent_selection_strategy)
    os.environ["AUTOFORMAL_NUM_ARCHIVE_INSP"] = str(args.num_archive_inspirations)
    os.environ["AUTOFORMAL_NUM_TOPK_INSP"] = str(args.num_top_k_inspirations)
    os.environ["AUTOFORMAL_NUM_INIT_CANDIDATES_GEN0"] = str(args.num_init_candidates_gen0)
    os.environ["AUTOFORMAL_BASELINE_MODE"] = str(args.baseline_mode)
    if args.seed is not None:
        os.environ["AUTOFORMAL_SEED"] = str(int(args.seed))
    if args.patch_types:
        os.environ["AUTOFORMAL_PATCH_TYPES"] = str(args.patch_types)
    if args.patch_type_probs:
        os.environ["AUTOFORMAL_PATCH_TYPE_PROBS"] = str(args.patch_type_probs)

    # no-LLM / offline controls
    if args.no_llm:
        args.llm_mode = "mock"
        os.environ["AUTOFORMAL_NO_LLM"] = "1"
    else:
        os.environ.pop("AUTOFORMAL_NO_LLM", None)
    os.environ["AUTOFORMAL_LLM_MODE"] = str(args.llm_mode)
    if args.replay_path:
        os.environ["AUTOFORMAL_REPLAY_PATH"] = str(args.replay_path)
    if args.mock_statements_path:
        os.environ["AUTOFORMAL_MOCK_STATEMENTS_PATH"] = str(args.mock_statements_path)

    # If a config file is provided, load problem fields from it.
    config_data = None
    if args.config:
        import json
        with open(args.config, "r") as f:
            config_data = json.load(f)
        args.informal = config_data.get("informal", args.informal)
        args.header = config_data.get("header", args.header)
        args.ground_truth = config_data.get("ground_truth", args.ground_truth)
        args.use_beq = config_data.get("use_beq", args.use_beq)

    print("=" * 60)
    print("FormalEvolve - Lean4 Autoformalization (spec v1.2)")
    print("=" * 60)
    print(f"Informal: {args.informal[:100]}...")
    print(f"Baseline mode: {args.baseline_mode}")
    print(f"Seed: {args.seed}")
    print(f"Generations: {args.num_generations}")
    print(f"Results dir: {args.results_dir}")
    print(f"BEq+: {args.use_beq}")
    print(f"Repair: {'disabled' if args.disable_repair else f'enabled (max {args.max_repair_attempts} attempts)'}")
    print(f"Budgets: llm_calls={args.max_llm_calls}, evals={args.max_evals}, time={args.max_time_seconds}")
    print(f"Stagnation: {args.stagnation_generations} gens, max_soft_resets={args.max_soft_resets}")
    print("Meets constraints: [A] Novelty for compile_ok=1 only (enforced in evaluator/runner path)")
    print("Meets constraints: [B] Archive keeps compile_ok=1 only")
    print("Meets constraints: [C] Repair counts toward budget")
    print("Meets constraints: [D] Compile on full {header}+{body}")
    print("Meets constraints: [F] compile_ok=0 always ranks below compile_ok=1")
    print("Meets constraints: [G] No dataset header in prompts")
    print("=" * 60)

    novelty_llm_models_raw = str(args.novelty_llm_models or "").strip()
    novelty_llm_disabled = novelty_llm_models_raw.lower() in {"none", "null"}
    novelty_llm_models_parsed = (
        None
        if novelty_llm_models_raw.lower() in {"", "none", "null"}
        else [m.strip() for m in novelty_llm_models_raw.split(",") if m.strip()]
    )

    evo_config, job_config, db_config, repair_config, termination_config, problem_config = create_configs(
        informal=args.informal,
        header=args.header,
        ground_truth=args.ground_truth,
        use_beq=args.use_beq,
        results_dir=args.results_dir,
        problem_overrides=config_data,
        baseline_mode=str(args.baseline_mode),
        seed=args.seed,
        llm_mode=args.llm_mode,
        replay_path=args.replay_path,
        mock_statements_path=args.mock_statements_path,
        no_llm=bool(args.no_llm),
        embedding_model=(args.embedding_model or None),
        code_embed_sim_threshold=float(args.code_embed_sim_threshold),
        max_novelty_attempts=int(args.max_novelty_attempts),
        novelty_llm_models=novelty_llm_models_parsed,
        novelty_llm_disabled=novelty_llm_disabled,
        num_init_candidates_gen0=int(args.num_init_candidates_gen0),
        parent_selection_strategy=str(args.parent_selection_strategy),
        num_archive_inspirations=int(args.num_archive_inspirations),
        num_top_k_inspirations=int(args.num_top_k_inspirations),
        patch_types=[t.strip() for t in args.patch_types.split(",") if t.strip()] if args.patch_types else None,
        patch_type_probs=[float(x.strip()) for x in args.patch_type_probs.split(",") if x.strip()] if args.patch_type_probs else None,
        num_generations=args.num_generations,
        max_parallel_jobs=args.max_parallel_jobs,
        # evolution-level novelty settings
        max_repair_attempts=args.max_repair_attempts,
        max_llm_calls=args.max_llm_calls,
        max_evals=args.max_evals,
        max_time_seconds=args.max_time_seconds,
        stagnation_generations=args.stagnation_generations,
        max_soft_resets=args.max_soft_resets,
        init_program_path=args.init_program,
    )

    # Disable repair if requested
    if args.disable_repair:
        repair_config.enabled = False
    if args.max_repair_attempts_gen0 is not None:
        repair_config.max_repair_attempts_gen0 = int(args.max_repair_attempts_gen0)

    # Create and run evolution with AutoformalizationRunner
    runner = AutoformalizationRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        repair_config=repair_config,
        termination_config=termination_config,
        problem_config=problem_config,
        verbose=args.verbose,
    )

    runner.run()

    print("\n" + "=" * 60)
    print("Evolution complete!")
    print(f"Results saved to: {args.results_dir}")

    # Print repair statistics
    repair_stats = runner.get_repair_stats()
    print(f"Repair LLM calls: {repair_stats['total_repair_llm_calls']}")
    print(f"Repair evals: {repair_stats['total_repair_evals']}")
    print(f"Repair cost: ${repair_stats['total_repair_cost']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
