# Supplementary material

This zip contains a code snapshot of FormalEvolve and a short results summary PDF:

- `submission/icml2026/supplementary_results.pdf`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Offline smoke test

```bash
python examples/icml2026_autoformalization/run_evo.py \
  --llm_mode mock \
  --num_generations 1 \
  --max_llm_calls 5 \
  --max_parallel_jobs 1
```

## Pilot run

```bash
python examples/icml2026_autoformalization/scripts/run_dataset_pilot.py \
  --dataset proofnet_test \
  --num_problems 5 \
  --baseline_mode ours \
  --llm_mode auto \
  --openai_llm_base_url "http://<host>:<port>/v1" \
  --llm_models "<GENERATOR_MODEL_ID>" \
  --max_llm_calls 100 \
  --concurrency 1 \
  --num_generations 200 \
  --paper_protocol
```

## Evaluation

```bash
python examples/icml2026_autoformalization/scripts/strict_eval_run.py \
  --run_root <RUN_ROOT_FROM_PILOT>
```

```bash
python examples/icml2026_autoformalization/scripts/paper_summary_run.py \
  --run_root <RUN_ROOT_FROM_PILOT>
```

## Notes

- If you use an OpenAI-compatible local server, export `OPENAI_API_KEY=EMPTY`.
- Semantic judging uses CriticLean and is off unless you pass `--use_semantic`.
