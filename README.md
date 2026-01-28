# FormalEvolve

FormalEvolve is a neuro-symbolic evolutionary search framework for generating **diverse** and **prover-effective** Lean 4 autoformalizations.

This repository is structured as:
- `shinka/`: the underlying evolutionary search engine (vendored)
- `autoformalization/`: Lean compilation + (optional) semantic judging utilities
- `examples/icml2026_autoformalization/`: the paper-aligned autoformalization pipeline (entrypoints, prompts, benchmarks)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Smoke test (no network)

This runs end-to-end with a mock generator (useful to verify the pipeline wiring):

```bash
python examples/icml2026_autoformalization/run_evo.py --llm_mode mock --num_generations 10 --max_llm_calls 30
```

### Pilot run on a bundled benchmark

```bash
python examples/icml2026_autoformalization/scripts/run_dataset_pilot.py \
  --dataset proofnet_test \
  --num_problems 5 \
  --baseline_mode ours \
  --llm_mode auto \
  --max_llm_calls 100 \
  --use_semantic \
  --no_cycle_consistency \
  --criticlean_base_url "<CRITICLEAN_BASE_URL>" \
  --criticlean_model "<CRITICLEAN_MODEL_ID>"
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
  --max_llm_calls 100 \
  --patch_llm_models "Qwen3-Patch-Model-Name" \
  --patch_openai_llm_base_url "http://127.0.0.1:8009/v1"
```

## Evaluation (trend reproduction)

For runs produced by `run_dataset_pilot.py`, compute per-problem strict metrics from the SQLite DB:

```bash
python examples/icml2026_autoformalization/scripts/strict_eval_run.py --run_root <RUN_ROOT>
```

## Optional external services

Some parts of the pipeline use external services; they are optional unless you enable them explicitly:

- **Lean server**: set `lean_server_url` in `problem_config.json` (otherwise the evaluator uses local `lean-interact`).
- **CriticLean semantic judging**: enable `--use_semantic` and configure CriticLean via `--criticlean_base_url/--criticlean_model` (or env `CRITIC_LEAN_URL`/`CRITIC_LEAN_MODEL`).
