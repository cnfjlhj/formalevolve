"""
Configuration for the legacy autoformalization baselines.

Original experiment setup:
- Dataset: PAug/ProofNetSharp, first 10 problems of the test split.
- Unified budget: 12 LLM calls across three methods:
  - Naive: sample 12 candidates in one shot.
  - Rewrite-only: 4 initial + 2 rewrite rounds × 4 = 12.
  - Evolution: 4 initial + 2 evolution rounds × 4 offspring = 12.

Kimina-specific defaults:
- KIMINA_MAX_TOKENS = 2048
- KIMINA_OUTPUT_CLEANUP = True
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Experiment configuration."""

    # =========================================================================
    # Dataset Configuration
    # =========================================================================
    dataset_name: str = "PAug/ProofNetSharp"
    dataset_split: str = "test"
    num_problems: int = 10  # first 10 problems

    # =========================================================================
    # LLM Configuration
    # =========================================================================
    # Keep this as a model identifier, not a machine-specific cache path,
    # so source releases remain portable across environments.
    llm_model: str = "Kimina-Autoformalizer-7B"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1536

    # =========================================================================
    # Budget Configuration (unified 12 LLM calls)
    # =========================================================================
    # Naive: sample 12 candidates in one shot
    naive_n: int = 12

    # Rewrite-only: 4 initial + 2 rounds × 4 = 12
    rewrite_init: int = 4      # initial samples
    rewrite_rounds: int = 2     # number of rewrite rounds
    rewrite_per_round: int = 4  # rewrites per round

    # Evolution: 4 initial + 2 rounds × 4 = 12
    evolve_init: int = 4        # initial samples
    evolve_rounds: int = 2      # number of evolution rounds
    evolve_offspring: int = 4   # offspring per round

    # =========================================================================
    # Evaluation Configuration
    # =========================================================================
    compile_timeout: int = 60   # Lean compile timeout (seconds)
    use_beq_plus: bool = True   # enable BEq+ equivalence check
    lambda_beq: float = 0.5     # BEq+ reward weight

    # =========================================================================
    # Results Configuration
    # =========================================================================
    results_dir: str = "results/autoformalization"
    seed: int = 42


SYSTEM_MESSAGE = "You are an expert in mathematics and Lean 4."

# Kimina-specific configurations
KIMINA_MAX_TOKENS = 2048  # Kimina needs more tokens for complex problems
KIMINA_OUTPUT_CLEANUP = True  # Enable cleanup for Kimina's comment-heavy output

# Default lean header (matches ProofNetSharp dataset format)
DEFAULT_LEAN_HEADER = """import Mathlib

open Function Fintype Subgroup Ideal Polynomial Submodule Zsqrtd RingHom
open scoped BigOperators"""


# =============================================================================
# Prompt Templates
# =============================================================================

# -----------------------------------------------------------------------------
# GENERATION: initial generation (Kimina-style)
# -----------------------------------------------------------------------------
GENERATION_PROMPT = """Please autoformalize the following problem in Lean 4 with the given header.

Header:
{header}

Problem:
{informal}

Write ONLY the theorem statement. Start with `theorem` or `lemma` and end with `:= by sorry`."""


# -----------------------------------------------------------------------------
# REWRITE: fix based on feedback (single-parent self-refine)
# -----------------------------------------------------------------------------
REWRITE_PROMPT = """You are given a Lean 4 formalization attempt that failed verification. Analyze the error and fix it.

Header:
{header}

Original problem:
{informal}

Current attempt:
```lean
{current_statement}
```

Verification result:
{feedback}

Task: Fix the issues identified above. Write ONLY the corrected theorem statement.
Start with `theorem` or `lemma` and end with `:= by sorry`."""


# -----------------------------------------------------------------------------
# EVOLUTION: multi-parent crossover (core improvement)
# -----------------------------------------------------------------------------
EVOLUTION_PROMPT = """You are given multiple formalization attempts for the same problem, each with different strengths and weaknesses.

Header:
{header}

Original problem:
{informal}

=== Parent Formalizations ===
{parent_summaries}
=== End of Parents ===

Analysis task:
1. Identify which parent has the best SYNTAX structure (compiles successfully)
2. Identify which parent captures the correct MATHEMATICAL MEANING (semantic score = 1)
3. If a parent failed compilation, note what syntax pattern to AVOID
4. If a parent failed semantic check, note what mathematical interpretation was WRONG

Synthesis task:
Create a NEW formalization that:
- Uses the syntactic patterns from parents that compiled successfully
- Captures the mathematical meaning from parents with correct semantics
- Avoids the mistakes identified in failed parents

Write ONLY the synthesized theorem statement.
Start with `theorem` or `lemma` and end with `:= by sorry`."""


# -----------------------------------------------------------------------------
# META_RECOMMENDATION: meta-level recommendations (optional; multi-round evolution)
# -----------------------------------------------------------------------------
META_RECOMMENDATION_PROMPT = """Based on the evolution history, provide recommendations for the next generation.

Problem:
{informal}

Evolution history (best candidates per round):
{evolution_history}

Common failure patterns observed:
{failure_patterns}

Provide 2-3 specific recommendations for improving the next generation of formalizations.
Focus on: type choices, quantifier structure, constraint formulation, Mathlib conventions."""
