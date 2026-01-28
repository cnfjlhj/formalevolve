# ICML 2026 Supplementary Material (Code Snapshot)

This archive contains an **anonymized** code snapshot for reproducing the main *pipeline behavior / trends* reported in the accompanying ICML 2026 submission.

It is designed for **reproducibility level B**:
- End-to-end runnable pipeline with a clear protocol
- Reproduces *trends* under fixed budgets (not necessarily exact reported numbers)
- Does **not** include large historical run artifacts or trained checkpoints

## Included result summary (PDF)

To make the supplementary package self-contained for review, it also includes a short PDF with:
- the main $T{=}100$ statement-generation table (CH@100 / SH@100 / uniformity metrics),
- the RR64 proof-utility table ($B{=}64$),
- lightweight audits (patch-type call breakdown, token usage, and optional wall-clock timing),
- an additional ReForm-as-seeder handover case study (@16).

See: `submission/icml2026/supplementary_results.pdf`.

## What is included

- Core code:
  - `shinka/`: evolutionary search engine
  - `autoformalization/`: compilation + (optional) semantic judging utilities
  - `examples/icml2026_autoformalization/`: paper-aligned runner + scripts + small bundled benchmarks
- Small bundled benchmarks (`examples/icml2026_autoformalization/benchmark/*.jsonl`) for quick smoke/pilot runs.
- Strict evaluation scripts computing metrics from the SQLite DB:
  - `examples/icml2026_autoformalization/scripts/strict_eval_run.py`

## What is NOT included

- Large experiment outputs, logs, caches, and historical results directories
- Proprietary datasets or massive raw corpora
- Model weights / checkpoints
- Any non-anonymous author identifiers

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 2) Offline smoke test (no network, no API keys)

This verifies the end-to-end wiring without depending on an external LLM server:

```bash
python examples/icml2026_autoformalization/run_evo.py \
  --llm_mode mock \
  --num_generations 1 \
  --max_llm_calls 5 \
  --max_parallel_jobs 1
```

### 3) Pilot run on a bundled benchmark

```bash
python examples/icml2026_autoformalization/scripts/run_dataset_pilot.py \
  --dataset proofnet_test \
  --num_problems 5 \
  --baseline_mode ours \
  --llm_mode auto \
  --max_llm_calls 100 \
  --concurrency 1 \
  --num_generations 200 \
  --use_semantic \
  --no_cycle_consistency \
  --criticlean_base_url "<CRITICLEAN_BASE_URL>" \
  --criticlean_model "<CRITICLEAN_MODEL_ID>"
```

Then compute strict metrics from the SQLite DB:

```bash
python examples/icml2026_autoformalization/scripts/strict_eval_run.py \
  --run_root <RUN_ROOT_FROM_PILOT>
```

Then compute paper-aligned aggregate metrics (CH@T / SH@T / Gini / Top-10% share over deduplicated semantic-ok counts):

```bash
python examples/icml2026_autoformalization/scripts/paper_summary_run.py \
  --run_root <RUN_ROOT_FROM_PILOT>
```

## Optional: dedicated patch/edit model

You can route **edit proposals** to a dedicated patch model (e.g., a Qwen3 patch-tuned model),
while keeping Gen0 sampling on the main generator model (or on a seed bank).

Via CLI flags:

```bash
python examples/icml2026_autoformalization/scripts/run_dataset_pilot.py \
  --dataset proofnet_test \
  --num_problems 5 \
  --baseline_mode ours \
  --llm_mode auto \
  --max_llm_calls 100 \
  --patch_llm_models "<PATCH_MODEL_NAME>" \
  --patch_openai_llm_base_url "http://127.0.0.1:8009/v1"
```

## Notes on optional services

Only the services you explicitly enable are required:
- Lean compilation defaults to local `lean-interact` (no HTTP server required). You can optionally set `lean_server_url`.
- Semantic judging uses CriticLean when `--use_semantic` is enabled (configure via `--criticlean_base_url/--criticlean_model` or env).
