"""

# Kimina-specific configurations
KIMINA_MAX_TOKENS = 2048  # Kimina needs more tokens for complex problems
KIMINA_OUTPUT_CLEANUP = True  # Enable cleanup for Kimina's comment-heavy output

Configuration for Autoformalization Experiment.

按照 要求.md 定义的实验设定:
- 数据集: PAug/ProofNetSharp, test split 前 10 个样本
- 三种方法统一 12 次 LLM 调用预算:
  - Naive: 一次性采样 12 个 candidate
  - Rewrite-only: 初代 4 个 + 2 轮 rewrite, 每轮 4 个 → 4 + 2×4 = 12
  - Evolution: 初代 4 个 + 2 轮 evolution, 每轮 4 个 offspring → 4 + 2×4 = 12
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
    num_problems: int = 10  # 前 10 个样本

    # =========================================================================
    # LLM Configuration
    # =========================================================================
    # Keep this as a model *identifier* (not a machine-specific local cache path)
    # to avoid leaking environment details in anonymous releases.
    llm_model: str = "Kimina-Autoformalizer-7B"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1536

    # =========================================================================
    # Budget Configuration (统一 12 次 LLM 调用)
    # =========================================================================
    # Naive: 一次性采样 12 个
    naive_n: int = 12

    # Rewrite-only: 初代 4 + 2轮×4 = 12
    rewrite_init: int = 4      # 初代采样数
    rewrite_rounds: int = 2     # rewrite 轮数
    rewrite_per_round: int = 4  # 每轮 rewrite 数

    # Evolution: 初代 4 + 2轮×4 = 12
    evolve_init: int = 4        # 初代采样数
    evolve_rounds: int = 2      # evolution 轮数
    evolve_offspring: int = 4   # 每轮 offspring 数

    # =========================================================================
    # Evaluation Configuration
    # =========================================================================
    compile_timeout: int = 60   # Lean 编译超时 (秒)
    use_beq_plus: bool = True   # 启用 BEq+ 等价检查
    lambda_beq: float = 0.5     # BEq+ 奖励权重

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
# GENERATION: 初始生成 (Kimina 原生格式)
# -----------------------------------------------------------------------------
GENERATION_PROMPT = """Please autoformalize the following problem in Lean 4 with the given header.

Header:
{header}

Problem:
{informal}

Write ONLY the theorem statement. Start with `theorem` or `lemma` and end with `:= by sorry`."""


# -----------------------------------------------------------------------------
# REWRITE: 基于 feedback 修正 (单亲 self-refine)
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
# EVOLUTION: 多亲 crossover (核心改进)
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
# META_RECOMMENDATION: 元层面的优化建议 (可选，用于多轮进化)
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
