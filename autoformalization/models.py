"""
Data Models for Autoformalization Experiment.

Defines the core data structures:
- Problem: A ProofNetSharp problem with informal/formal statements
- Candidate: A candidate Lean4 formalization with evaluation results
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import uuid


@dataclass
class Problem:
    """
    A ProofNetSharp problem.

    Attributes:
        id: Unique identifier
        nl_statement: Natural language statement (informal)
        nl_proof: Natural language proof (optional)
        lean4_src_header: Lean4 header/imports
        lean4_formalization: Ground truth Lean4 formalization
    """
    id: str
    nl_statement: str
    nl_proof: str
    lean4_src_header: str
    lean4_formalization: str

    @classmethod
    def from_hf(cls, data: Dict[str, Any]) -> "Problem":
        """Create a Problem from HuggingFace dataset entry."""
        return cls(
            id=data["id"],
            nl_statement=data["nl_statement"],
            nl_proof=data.get("nl_proof", ""),
            lean4_src_header=data["lean4_src_header"],
            lean4_formalization=data["lean4_formalization"],
        )


@dataclass
class Candidate:
    """
    A candidate Lean4 formalization.

    Evaluation fields (experiment protocol):
    - compile_ok: whether Lean elaboration succeeds (hard gate)
    - compile_error: compilation error message (if any)
    - s_sem: CriticLean semantic consistency (0 or 1)
    - beq_flag: BEq+ equivalence (0 or 1)
    - critic_raw: raw CriticLean response

    Reward:
    - If compile_ok is False: reward = 0.0
    - Otherwise: reward = s_sem + lambda_beq * beq_flag

    Success:
    - soft_success: compile_ok == True and s_sem == 1
    - strict_success: compile_ok == True and beq_flag == 1
    """
    code: str                    # Lean4 theorem statement
    header: str                  # Import header

    # Evaluation results
    compile_ok: bool = False
    compile_error: Optional[str] = None
    s_sem: int = 0               # CriticLean score (0 or 1)
    beq_flag: int = 0            # BEq+ equivalence (0 or 1)
    critic_raw: str = ""         # Raw CriticLean response

    # Metadata
    generation: int = 0
    parent_id: Optional[str] = None
    candidate_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def reward(self, lambda_beq: float = 0.5) -> float:
        """
        Calculate reward for this candidate.

        Args:
            lambda_beq: Weight for BEq+ bonus (default 0.5)

        Returns:
            float: Reward value
                - 0.0 if compile failed
                - s_sem + lambda_beq * beq_flag otherwise
        """
        if not self.compile_ok:
            return 0.0
        # s_sem ∈ {0, 1}, beq_flag ∈ {0, 1}
        return float(self.s_sem + lambda_beq * self.beq_flag)

    @property
    def soft_success(self) -> bool:
        """Soft success: compiled and CriticLean approved."""
        return self.compile_ok and self.s_sem == 1

    @property
    def strict_success(self) -> bool:
        """Strict success: compiled and BEq+ equivalent."""
        return self.compile_ok and self.beq_flag == 1

    @property
    def full_code(self) -> str:
        """Full Lean code with header."""
        return self.header.strip() + "\n\n" + self.code.strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "candidate_id": self.candidate_id,
            "code": self.code,
            "compile_ok": self.compile_ok,
            "compile_error": self.compile_error,
            "s_sem": self.s_sem,
            "beq_flag": self.beq_flag,
            "critic_raw": self.critic_raw,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "reward": self.reward(),
            "soft_success": self.soft_success,
            "strict_success": self.strict_success,
        }


@dataclass
class EvaluationResult:
    """
    Complete evaluation result for a candidate.

    Used as an intermediate structure during evaluation pipeline.
    """
    compile_ok: bool
    compile_error: Optional[str]
    s_sem: int  # 0 or 1
    beq_flag: int  # 0 or 1
    critic_raw: str

    def apply_to(self, candidate: Candidate) -> Candidate:
        """Apply evaluation results to a candidate."""
        candidate.compile_ok = self.compile_ok
        candidate.compile_error = self.compile_error
        candidate.s_sem = self.s_sem
        candidate.beq_flag = self.beq_flag
        candidate.critic_raw = self.critic_raw
        return candidate
