"""
Autoformalization Module.

Implements Lean4 autoformalization with three methods:
1. Naive best-of-N sampling
2. Rewrite-only (single chain self-refine)
3. Population Evolution (Shinka-style)

Evaluation pipeline:
- Lean compile check (hard constraint)
- BEq+ equivalence check (optional)
- CriticLean semantic check (0/1)
"""

from .models import Candidate, Problem
from .config import Config
from .lean_env import get_lean_server, is_lean_server_available, shutdown_lean_server
from .evaluator import evaluate_candidate, check_lean_compile, beq_plus_equiv
from .critic_wrapper import critic_eval, close_session
from .baseline import run_naive, run_rewrite, run_evolution, run_experiment

__all__ = [
    # Models
    "Candidate",
    "Problem",
    # Config
    "Config",
    # Lean environment
    "get_lean_server",
    "is_lean_server_available",
    "shutdown_lean_server",
    # Evaluation
    "evaluate_candidate",
    "check_lean_compile",
    "beq_plus_equiv",
    "critic_eval",
    "close_session",
    # Baseline methods
    "run_naive",
    "run_rewrite",
    "run_evolution",
    "run_experiment",
]
