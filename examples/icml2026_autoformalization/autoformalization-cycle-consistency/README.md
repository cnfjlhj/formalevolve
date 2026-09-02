# Cycle Consistency Autoformalization

A lightweight implementation inspired by cycle-consistency style scoring.

## Core idea

Cycle consistency evaluates a formalization by "back-translation":

```
Informal Statement ──[formalize]──> Formal Candidates
                                          │
                                          ▼
                            ┌─────────────────────────┐
                            │  Cycle Consistency      │
                            │  Score = log P(informal │
                            │            | "Informalize: formal") │
                            └─────────────────────────┘
                                          │
                                          ▼
                                    Best Candidate
```

Good formalization → back-translation matches the original statement → high score
Incorrect formalization → back-translation drifts away → low score

## Directory structure

```
autoformalization-cycle-consistency/
├── src/
│   ├── __init__.py
│   ├── config.py            # configuration
│   ├── model_interface.py   # model interface abstraction
│   └── cycle_consistency.py # core implementation
├── examples/
│   ├── demo_with_dummy.py       # smoke test with dummy models
│   └── demo_with_real_models.py # run with real models
├── configs/
└── README.md
```

## Quickstart

### 1) Smoke test (no model required)

```bash
cd examples
python demo_with_dummy.py
```

### 2) Run with real models

First, deploy models (e.g., via vLLM):

```bash
# Terminal 1: serve the formalizer model
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B --port 8000

# Terminal 2: serve the informalizer (back-translation) model
vllm serve meta-llama/Llama-3.2-3B-Instruct --port 8001
```

Then update the config in `examples/demo_with_real_models.py` and run:

```bash
python demo_with_real_models.py
```

## Usage

### Basic usage

```python
from src import (
    CycleConsistencyAutoformalization,
    OpenAICompatibleLLM,
    Config,
)

# 1) Create models
formalizer = OpenAICompatibleLLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    base_url="http://localhost:8000/v1",
)
informalizer = OpenAICompatibleLLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    base_url="http://localhost:8001/v1",
)

# 2) Create the cycle-consistency wrapper
cc = CycleConsistencyAutoformalization(
    formalizer=formalizer,
    informalizer=informalizer,
)

# 3) Run autoformalization
result = cc.autoformalize(
    "For all positive integers n, n squared is greater than or equal to n."
)

print(result.best_formalization)
# theorem forall_n_sq_ge_n (n : ℕ) (h : n > 0) : n^2 ≥ n := by sorry
```

### HuggingFace models (local GPU)

```python
from src import HuggingFaceLLM

# Load the model on GPU
informalizer = HuggingFaceLLM(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    device="auto",
)

# compute_log_prob computes exact log probability
score = informalizer.compute_log_prob(
    prompt="Informalize: theorem (n : ℕ) : n^2 ≥ n",
    completion="For all natural numbers n, n squared is at least n.",
)
print(f"Log prob: {score.log_prob}, Tokens: {score.num_tokens}")
```

### Custom configuration

```python
from src import Config

config = Config()
config.model.num_candidates = 10      # generate more candidates
config.model.temperature = 0.8        # higher diversity
config.cycle_consistency.normalize_by_length = True  # length-normalized scores
config.verbose = True                 # verbose logs

cc = CycleConsistencyAutoformalization(
    formalizer=formalizer,
    informalizer=informalizer,
    config=config,
)
```

## Model interfaces

Three backends are provided:

| Interface | Description | Dependencies |
|------|------|------|
| `OpenAICompatibleLLM` | OpenAI-compatible API (e.g., vLLM/Ollama) | `openai` |
| `HuggingFaceLLM` | Load a HuggingFace model directly | `transformers`, `torch` |
| `DummyLLM` | Testing only, returns fake data | none |

You can also implement your own backend by subclassing `LLMInterface` and implementing
`generate()` and `compute_log_prob()`.

## Dependencies

```bash
# Minimal dependency
pip install openai

# If you want to use HuggingFace models
pip install transformers torch

# Recommended: serve models via vLLM
pip install vllm
```

## Citation

```bibtex
@inproceedings{barbadacosta2025improving,
  title={Improving autoformalization via cycle consistency and incremental
         type-checking using language-model probabilistic programs},
  author={Barba da Costa, Mauricio and Zaiser, Fabian and Collins, Katherine M.
          and Patel, Romir and O'Donnell, Timothy J. and Lew, Alexander K.
          and Tenenbaum, Joshua B. and Mansinghka, Vikash K. and Freer, Cameron E.},
  booktitle={The 5th Workshop on Mathematical Reasoning and AI (MATH-AI)},
  year={2025}
}
```

## Extensions

This implementation covers cycle consistency only. The paper also includes:

- **Incremental Type-Checking**: requires integration with Lean 4
- **SMC sampling**: requires the GenLM framework

For a full reproduction, see the [GenLM](https://github.com/genlm/genlm-control) library.
