# FormalEvolve

[![arXiv](https://img.shields.io/badge/arXiv-2603.19828-b31b1b.svg)](https://arxiv.org/abs/2603.19828)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

FormalEvolve is a neuro-symbolic evolutionary search framework for generating diverse and prover-effective Lean 4 autoformalizations. Instead of returning one formal statement per informal problem, it searches under a fixed generator-call budget and maintains a compilation-feasible archive of semantically accepted candidates for downstream proving.

The paper **“FormalEvolve: Neuro-Symbolic Evolutionary Search for Diverse Autoformalization”** is accepted to **Findings of EMNLP 2026**.

- Paper: https://arxiv.org/abs/2603.19828
- Code: https://github.com/cnfjlhj/formalevolve

## Method overview

FormalEvolve combines:

- LLM-driven mutation, crossover, and bounded patch repair;
- symbolic Lean AST rewrites for call-free structural diversity;
- compilation filtering and optional semantic judging;
- a reusable archive for diverse candidate generation;
- fixed-budget evaluation for both autoformalization and downstream proving.

## Repository layout

- `shinka/`: adapted evolutionary-search engine based on [SakanaAI/ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve).
- `autoformalization/`: Lean compilation, candidate evaluation, and optional semantic judging.
- `examples/formalevolve_autoformalization/`: paper-aligned entrypoints, prompts, benchmark adapters, and evaluation scripts.
- `experiments/`: supporting evaluation and analysis utilities.
- `submission/`: links to the current EMNLP 2026 paper and future ACL Anthology metadata.
- `tests/`: unit and regression tests.

## Installation

FormalEvolve requires Python 3.10 or newer. Lean-backed evaluation uses `lean-interact` and may download the configured Lean toolchain on first use.

```bash
git clone https://github.com/cnfjlhj/formalevolve.git
cd formalevolve
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

The paper-aligned default Lean version is `v4.15.0`. Override it with `LEAN_INTERACT_LEAN_VERSION` when necessary.

## Offline smoke test

This exercises the evolutionary pipeline with a mock generator and does not require an external model endpoint:

```bash
python examples/formalevolve_autoformalization/run_evo.py \
  --llm_mode mock \
  --num_generations 1 \
  --max_llm_calls 5 \
  --max_parallel_jobs 1
```

## Paper-aligned pilot

```bash
python examples/formalevolve_autoformalization/scripts/run_dataset_pilot.py \
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

The pilot writes:

- `manifest.json` and `status.json` under `--out_root`;
- per-problem run directories under `--out_root/runs/`;
- an `evolution_db.sqlite` database for each problem.

## Semantic judging and repair

Enable CriticLean-compatible semantic judging and bounded semantic repair with:

```bash
python examples/formalevolve_autoformalization/scripts/run_dataset_pilot.py \
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

A dedicated patch model can be configured independently:

```bash
python examples/formalevolve_autoformalization/scripts/run_dataset_pilot.py \
  --dataset proofnet_test \
  --num_problems 5 \
  --baseline_mode ours \
  --llm_mode auto \
  --openai_llm_base_url "http://<host>:<port>/v1" \
  --llm_models "<GENERATOR_MODEL_ID>" \
  --patch_llm_models "<PATCH_MODEL_ID>" \
  --patch_openai_llm_base_url "http://<host>:<port>/v1" \
  --max_llm_calls 100 \
  --paper_protocol
```

## Seedbanks

Build a seedbank:

```bash
python examples/formalevolve_autoformalization/scripts/build_seedbank.py \
  --dataset proofnet_test \
  --num_problems 5 \
  --seeds_per_problem 16 \
  --llm_mode auto \
  --openai_llm_base_url "http://<host>:<port>/v1" \
  --llm_models "<SEED_MODEL_ID>"
```

Reuse it with `--init_programs_root <INIT_PROGRAMS_ROOT>` when launching `run_dataset_pilot.py`.

## Evaluation

Compute strict per-problem metrics:

```bash
python examples/formalevolve_autoformalization/scripts/strict_eval_run.py \
  --run_root <RUN_ROOT>
```

Compute aggregate paper metrics:

```bash
python examples/formalevolve_autoformalization/scripts/paper_summary_run.py \
  --run_root <RUN_ROOT>
```

## Tests

Install the test runner into the active environment:

```bash
python -m pip install pytest
```

```bash
python -m pytest -q tests \
  examples/formalevolve_autoformalization/autoformalization-cycle-consistency/examples/test_cycle_consistency_scoring.py
```

## Reproducibility notes

- Set `OPENAI_API_KEY=EMPTY` when an OpenAI-compatible local endpoint requires a non-empty placeholder key.
- External generator, patch, semantic-judge, and prover services are not bundled.
- Run outputs, model weights, credentials, local service URLs, and machine-specific caches are intentionally excluded.
- The bundled benchmark snapshots are derived from ProofNet and CombiBench. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and licenses.

## Citation

```bibtex
@misc{lu2026formalevolve,
  title         = {FormalEvolve: Neuro-Symbolic Evolutionary Search for Diverse Autoformalization},
  author        = {Haijian Lu and Wei Wang and Jing Liu},
  year          = {2026},
  eprint        = {2603.19828},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  note          = {Accepted to Findings of EMNLP 2026}
}
```

Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff).

## License

FormalEvolve is released under the Apache License 2.0. The `shinka/` engine is adapted from ShinkaEvolve, and bundled benchmark snapshots retain their upstream licenses and attribution. See [`NOTICE`](NOTICE), [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), and [`third_party/licenses/`](third_party/licenses/).
