# FormalEvolve

FormalEvolve is a neuro-symbolic evolutionary search framework for generating **diverse** and **prover-effective** Lean 4 autoformalizations.

This repository is structured as:
- `shinka/`: the underlying evolutionary search engine (vendored)
- `autoformalization/`: Lean compilation + (optional) semantic judging utilities
- `examples/icml2026_autoformalization/`: the paper-aligned autoformalization pipeline (entrypoints, prompts, benchmarks)

## Reproducibility target (ICML-style)

This code release is designed for **reproducibility level B**:
- the pipeline runs end-to-end with a clear protocol and fixed budgets (e.g., `T=100` LLM calls),
- it aims to reproduce **trends** (e.g., CH@T / SH@T and uniformity metrics),
- it does **not** require releasing large historical run artifacts or exact, fully-matched numbers.

## Pipeline overview (paper-aligned)

The main workflow is:

1. **Pick a benchmark slice** (bundled JSONL files under `examples/icml2026_autoformalization/benchmark/`).
2. For each problem, run an **evolution loop** that proposes Lean 4 theorem statements.
3. Evaluate each candidate with:
   - local Lean compilation (`compile_ok`),
   - optional semantic 0/1 judging via **CriticLean** (`semantic_ok`) when enabled.
4. Log everything into a per-problem **SQLite DB** (`evolution_db.sqlite`) for strict post-hoc evaluation.

Recommended entrypoints:
- Multi-problem runner (paper-aligned): `examples/icml2026_autoformalization/scripts/run_dataset_pilot.py`
- Single-problem runner (debugging): `examples/icml2026_autoformalization/run_evo.py`

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Running the pipeline

### 1) Offline smoke test (no network)

This verifies the end-to-end wiring without depending on any external LLM service:

```bash
python examples/icml2026_autoformalization/run_evo.py \
  --llm_mode mock \
  --num_generations 1 \
  --max_llm_calls 5 \
  --max_parallel_jobs 1
```

### 2) Single-problem run (useful for prompt/debug)

```bash
python examples/icml2026_autoformalization/run_evo.py \
  --llm_mode auto \
  --openai_llm_base_url "http://<host>:<port>/v1" \
  --llm_models "<GENERATOR_MODEL_ID>" \
  --num_generations 200 \
  --max_llm_calls 100 \
  --results_dir /tmp/formalevolve_single_debug
```

Notes:
- When using an OpenAI-compatible local server (e.g., vLLM), also export `OPENAI_API_KEY=EMPTY`.
- `--num_generations` can be set larger than needed; the budget `--max_llm_calls` is the hard stop.

### 3) Pilot run on a bundled benchmark (paper-aligned multi-problem)

This runs `N` problems in parallel (each problem spawns its own `run_evo.py` subprocess) and writes:
- `manifest.json` (run configuration for auditability),
- `status.json` (heartbeat progress),
- `runs/<problem_id>/...` (per-problem outputs + SQLite DB).

Minimal example (semantic judging ON, cycle-consistency OFF by default):

```bash
python examples/icml2026_autoformalization/scripts/run_dataset_pilot.py \
  --dataset proofnet_test \
  --num_problems 5 \
  --baseline_mode ours \
  --llm_mode auto \
  --openai_llm_base_url "http://<host>:<port>/v1" \
  --llm_models "<GENERATOR_MODEL_ID>" \
  --max_llm_calls 100 \
  --concurrency 2 \
  --num_generations 400 \
  --compile_timeout 60 \
  --use_semantic \
  --no_cycle_consistency \
  --criticlean_base_url "<CRITICLEAN_BASE_URL>" \
  --criticlean_model "<CRITICLEAN_MODEL_ID>" \
  --out_root /tmp/formalevolve_pilot_proofnet_test
```

### Optional: use a dedicated patch/edit model

If you have a specialized patch model (e.g. Qwen3-Patch), you can route **edit proposals**
to that model while keeping Gen0 sampling on `--llm_models` (or seed bank):

```bash
python examples/icml2026_autoformalization/scripts/run_dataset_pilot.py \
  --dataset proofnet_test \
  --num_problems 5 \
  --baseline_mode ours \
  --llm_mode auto \
  --openai_llm_base_url "http://<host>:<port>/v1" \
  --llm_models "<GENERATOR_MODEL_ID>" \
  --max_llm_calls 100 \
  --patch_llm_models "Qwen3-Patch-Model-Name" \
  --patch_openai_llm_base_url "http://<host>:<port>/v1"
```

### Optional: launch a larger run in the background (nohup)

For long-running pilot runs (e.g., 20 problems at `T=100`) we provide a minimal `nohup` launcher:

```bash
OPENAI_LLM_BASE_URL="http://<host>:<port>/v1" \
OPENAI_API_KEY="EMPTY" \
AUTOFORMAL_LLM_MODELS="<GENERATOR_MODEL_ID>" \
CRITIC_LEAN_BASE_URL="http://<host>:<port>" \
bash examples/icml2026_autoformalization/scripts/nohup_run_demo.sh proofnet_test 20 100
```

This writes progress to:
- `<OUT_ROOT>/status.json` (heartbeat)
- `<OUT_ROOT>/nohup.out` / `<OUT_ROOT>/nohup.err`

### Optional: quick trend demo (ours vs batchN)

This runs the same slice twice and prints a strict summary diff:

```bash
OPENAI_LLM_BASE_URL="http://<host>:<port>/v1" \
OPENAI_API_KEY="EMPTY" \
AUTOFORMAL_LLM_MODELS="<GENERATOR_MODEL_ID>" \
CRITIC_LEAN_BASE_URL="http://<host>:<port>" \
bash examples/icml2026_autoformalization/scripts/run_trend_demo.sh proofnet_test 10 100
```

## Initial solutions (Gen0) / seedbanks

FormalEvolve supports two ways to bootstrap **generation 0**:

### A) Gen0 from the generator LLM (default)

If you do not provide a seedbank, Gen0 is created by sampling from `--llm_models`:
- Seed 0 uses `--init_program` (defaults to `examples/icml2026_autoformalization/initial.lean`).
- Seeds 1..K are sampled from the LLM (until at least one non-placeholder `compile_ok=1` seed exists, or the budget is exhausted).

Key knobs:
- `--num_init_candidates_gen0` (in both `run_evo.py` and `run_dataset_pilot.py`)
- `--max_repair_attempts_gen0` (in `run_evo.py`) if you want more aggressive compile repair during bootstrapping

### B) Gen0 from a seedbank (recommended for paper-style seeding)

If you have pre-generated initial solutions (e.g., from a seeder model), you can reuse them
without spending Gen0 LLM calls by providing `--init_programs_root` to `run_dataset_pilot.py`.

Supported directory layouts (per problem):

```
<INIT_PROGRAMS_ROOT>/
  <problem_name>/
    seed_0/main.lean
    seed_1/main.lean
    ...

<INIT_PROGRAMS_ROOT>/
  <problem_name>/
    gen_0/
      seed_0/main.lean
      seed_1/main.lean
      ...

<INIT_PROGRAMS_ROOT>/
  0000_<problem_name>/gen_0/seed_0/main.lean
  0001_<problem_name>/gen_0/seed_0/main.lean
  ...
```

Example usage:

```bash
python examples/icml2026_autoformalization/scripts/run_dataset_pilot.py \
  --dataset proofnet_test \
  --num_problems 20 \
  --baseline_mode ours \
  --llm_mode auto \
  --openai_llm_base_url "http://<host>:<port>/v1" \
  --llm_models "<GENERATOR_MODEL_ID>" \
  --max_llm_calls 100 \
  --use_semantic \
  --no_cycle_consistency \
  --criticlean_base_url "<CRITICLEAN_BASE_URL>" \
  --init_programs_root "<INIT_PROGRAMS_ROOT>" \
  --num_init_candidates_gen0 16
```

Budget accounting (optional, paper-aligned fairness):
- `AUTOFORMAL_SEEDBANK_DEBIT_CALLS=1` counts each reused seed as budget.
- `AUTOFORMAL_SEEDBANK_CALLS_PER_SEED=1` sets the per-seed debit (e.g., `1` call/seed).
- `AUTOFORMAL_REUSE_INIT_EVAL=1` reuses seedbank evaluation artifacts when available (faster).

## Patch mechanism configuration

The evolution loop proposes edits via a small set of patch styles (paper-aligned defaults):
- `full`: rewrite the entire Lean file
- `diff`: propose a unified diff patch
- `cross`: cross-inspiration edit using archived statements

How to tune:
- `run_evo.py` flags: `--patch_types`, `--patch_type_probs`
- env vars: `AUTOFORMAL_PATCH_TYPES`, `AUTOFORMAL_PATCH_TYPE_PROBS`

Example:

```bash
python examples/icml2026_autoformalization/run_evo.py \
  --llm_mode auto \
  --openai_llm_base_url "http://<host>:<port>/v1" \
  --llm_models "<GENERATOR_MODEL_ID>" \
  --patch_openai_llm_base_url "http://<host>:<port>/v1" \
  --patch_llm_models "<PATCH_MODEL_ID>" \
  --patch_types "full,diff,cross" \
  --patch_type_probs "0.5,0.3,0.2" \
  --max_llm_calls 100
```

## Evaluation (strict + paper metrics)

For runs produced by `run_dataset_pilot.py`, compute per-problem strict metrics from the SQLite DB:

```bash
python examples/icml2026_autoformalization/scripts/strict_eval_run.py --run_root <RUN_ROOT>
```

Then compute paper-aligned aggregate metrics (CH@T / SH@T / deduplicated SemOK totals and uniformity):

```bash
python examples/icml2026_autoformalization/scripts/paper_summary_run.py --run_root <RUN_ROOT>
```

## Where to tune parameters

If you only want the paper-aligned protocol, start with `run_dataset_pilot.py`:
- **Budget**: `--max_llm_calls`, `--num_generations`
- **Parallelism**: `--concurrency` (across problems), `--max_parallel_jobs` (within a problem)
- **Baselines**: `--baseline_mode` in `{ours,batchN,repairloop1}`
- **Semantic judge**: `--use_semantic` + `--criticlean_base_url/--criticlean_model`
- **Seedbank**: `--init_programs_root`, `--num_init_candidates_gen0`

For fine-grained knobs, use `run_evo.py` directly (single problem) and/or set env vars:
- **Parent sampling**: `--parent_selection_strategy`, `--num_archive_inspirations`, `--num_top_k_inspirations`
- **Patch styles**: `--patch_types`, `--patch_type_probs`
- **Repairs**: `--max_repair_attempts`, `--max_repair_attempts_gen0` (or `--disable_repair`)

## Supplementary zip (ICML-style code snapshot)

If you need a clean supplementary archive (no `.git/`, no outputs, no secrets), use:

```bash
bash submission/icml2026/make_supplementary_zip.sh /tmp/formalevolve_icml2026_supplementary.zip
```

## Optional external services

Some parts of the pipeline use external services; they are optional unless you enable them explicitly:

- **Lean server**: set `lean_server_url` in `problem_config.json` (otherwise the evaluator uses local `lean-interact`).
- **CriticLean semantic judging**: enable `--use_semantic` and configure CriticLean via `--criticlean_base_url/--criticlean_model` (or env `CRITIC_LEAN_URL`/`CRITIC_LEAN_MODEL`).

Cycle-consistency is supported as an ablation but is **disabled by default** and not required for the main protocol.
