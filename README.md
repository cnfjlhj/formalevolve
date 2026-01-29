# FormalEvolve

FormalEvolve generates Lean4 theorem statements from informal problems using an evolutionary search loop.

## Layout

- `shinka/`: vendored evolution engine
- `autoformalization/`: Lean compilation and optional semantic judging
- `examples/icml2026_autoformalization/`: pipeline entrypoints, prompts, scripts, bundled benchmarks

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Quick start

### 1) Offline smoke test

```bash
python examples/icml2026_autoformalization/run_evo.py \
  --llm_mode mock \
  --num_generations 1 \
  --max_llm_calls 5 \
  --max_parallel_jobs 1
```

### 2) Multi-problem pilot run

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
  --paper_protocol \
  --out_root /tmp/formalevolve_pilot
```

Output:
- `manifest.json` and `status.json` under `--out_root`
- per-problem folders under `--out_root/runs/`
- per-problem SQLite DB `evolution_db.sqlite`

## Semantic judging via CriticLean

Enable semantic judging and bounded semantic repair:

```bash
python examples/icml2026_autoformalization/scripts/run_dataset_pilot.py \
  --dataset proofnet_test \
  --num_problems 5 \
  --baseline_mode ours \
  --llm_mode auto \
  --openai_llm_base_url "http://<host>:<port>/v1" \
  --llm_models "<GENERATOR_MODEL_ID>" \
  --max_llm_calls 100 \
  --paper_protocol \
  --use_semantic \
  --enable_semantic_repair \
  --criticlean_base_url "<CRITICLEAN_BASE_URL>" \
  --criticlean_model "<CRITICLEAN_MODEL_ID>" \
  --out_root /tmp/formalevolve_pilot_semantic
```

## Dedicated patch model

Route edit proposals to a dedicated patch model:

```bash
python examples/icml2026_autoformalization/scripts/run_dataset_pilot.py \
  --dataset proofnet_test \
  --num_problems 5 \
  --baseline_mode ours \
  --llm_mode auto \
  --openai_llm_base_url "http://<host>:<port>/v1" \
  --llm_models "<GENERATOR_MODEL_ID>" \
  --max_llm_calls 100 \
  --paper_protocol \
  --patch_llm_models "<PATCH_MODEL_ID>" \
  --patch_openai_llm_base_url "http://<host>:<port>/v1"
```

## Seedbanks

Build a seedbank:

```bash
python examples/icml2026_autoformalization/scripts/build_seedbank.py \
  --dataset proofnet_test \
  --num_problems 5 \
  --seeds_per_problem 16 \
  --llm_mode auto \
  --openai_llm_base_url "http://<host>:<port>/v1" \
  --llm_models "<SEED_MODEL_ID>"
```

Reuse a seedbank:

```bash
python examples/icml2026_autoformalization/scripts/run_dataset_pilot.py \
  --dataset proofnet_test \
  --num_problems 20 \
  --baseline_mode ours \
  --llm_mode auto \
  --openai_llm_base_url "http://<host>:<port>/v1" \
  --llm_models "<GENERATOR_MODEL_ID>" \
  --max_llm_calls 100 \
  --paper_protocol \
  --init_programs_root "<INIT_PROGRAMS_ROOT>"
```

## Evaluation

Compute strict metrics from a run folder:

```bash
python examples/icml2026_autoformalization/scripts/strict_eval_run.py --run_root <RUN_ROOT>
```

Compute aggregate metrics:

```bash
python examples/icml2026_autoformalization/scripts/paper_summary_run.py --run_root <RUN_ROOT>
```

## Notes

- If you use an OpenAI-compatible local server, export `OPENAI_API_KEY=EMPTY`.
- Local compilation uses `lean-interact` and may download the Lean toolchain on first use.
  Default Lean toolchain: `v4.15.0`. Override with `LEAN_INTERACT_LEAN_VERSION`.

## Supplementary zip

Create a clean zip for submission:

```bash
bash submission/icml2026/make_supplementary_zip.sh /tmp/formalevolve_supplementary.zip
```

The zip includes `submission/icml2026/supplementary_results.pdf` and excludes `.git/`, virtualenvs, and run outputs.
