"""
Cycle Consistency Autoformalization

A simple implementation of cycle consistency for autoformalization,
based on the MATH-AI 2025 paper.
"""

from .cycle_consistency import (
    CycleConsistencyAutoformalization,
    CycleConsistencyResult,
    FormalizationCandidate,
    autoformalize_with_cycle_consistency,
)

from .model_interface import (
    LLMInterface,
    GenerationResult,
    LogProbResult,
    OpenAICompatibleLLM,
    HuggingFaceLLM,
    DummyLLM,
)

from .config import (
    Config,
    ModelConfig,
    CycleConsistencyConfig,
    PromptConfig,
    DEFAULT_CONFIG,
)

__all__ = [
    # Core
    "CycleConsistencyAutoformalization",
    "CycleConsistencyResult",
    "FormalizationCandidate",
    "autoformalize_with_cycle_consistency",
    # Model interfaces
    "LLMInterface",
    "GenerationResult",
    "LogProbResult",
    "OpenAICompatibleLLM",
    "HuggingFaceLLM",
    "DummyLLM",
    # Config
    "Config",
    "ModelConfig",
    "CycleConsistencyConfig",
    "PromptConfig",
    "DEFAULT_CONFIG",
]

__version__ = "0.1.0"
