"""
Cycle Consistency for Autoformalization

Core implementation based on:
"Improving autoformalization via cycle consistency and incremental type-checking
using language-model probabilistic programs" (MATH-AI 2025)

Algorithm:
1. Generate N formalization candidates from informal statement
2. For each candidate, compute cycle consistency score:
   - Construct informalization prompt: "Informalize: {formal}"
   - Compute log P(original_informal | prompt)
3. Select candidate with highest score
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging
import re

from .model_interface import LLMInterface, LogProbResult
from .config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class FormalizationCandidate:
    """A single formalization candidate with its score"""
    formal_statement: str
    log_prob_score: float
    normalized_score: float
    num_tokens: int
    rank: int = 0

    def __repr__(self):
        return f"Candidate(rank={self.rank}, score={self.normalized_score:.2f}, tokens={self.num_tokens})"


@dataclass
class CycleConsistencyResult:
    """Result of cycle consistency autoformalization"""
    # Input
    informal_statement: str

    # Best result
    best_formalization: str
    best_score: float

    # All candidates (sorted by score, descending)
    all_candidates: List[FormalizationCandidate] = field(default_factory=list)

    # Metadata
    num_candidates_generated: int = 0
    total_tokens_used: int = 0

    def __repr__(self):
        return (
            f"CycleConsistencyResult(\n"
            f"  informal='{self.informal_statement[:50]}...',\n"
            f"  best_score={self.best_score:.2f},\n"
            f"  num_candidates={self.num_candidates_generated}\n"
            f")"
        )


class CycleConsistencyAutoformalization:
    """
    Autoformalization using Cycle Consistency scoring.

    Usage:
        # Initialize with your models
        formalizer = HuggingFaceLLM("deepseek-ai/...")
        informalizer = HuggingFaceLLM("meta-llama/Llama-3.2-3B")

        cc = CycleConsistencyAutoformalization(
            formalizer=formalizer,
            informalizer=informalizer,
        )

        # Run autoformalization
        result = cc.autoformalize("For all positive integers n, n² ≥ n")
        print(result.best_formalization)
    """

    def __init__(
        self,
        formalizer: LLMInterface,
        informalizer: LLMInterface,
        config: Optional[Config] = None,
    ):
        """
        Initialize cycle consistency autoformalization.

        Args:
            formalizer: LLM for generating formalizations (informal -> formal)
            informalizer: LLM for computing cycle consistency scores
                          (computes log P(informal | formal))
            config: Configuration object
        """
        self.formalizer = formalizer
        self.informalizer = informalizer
        self.config = config or DEFAULT_CONFIG

    def autoformalize(
        self,
        informal_statement: str,
        num_candidates: Optional[int] = None,
        few_shot_examples: Optional[List[Tuple[str, str]]] = None,
    ) -> CycleConsistencyResult:
        """
        Autoformalize an informal mathematical statement.

        Args:
            informal_statement: Natural language math statement
            num_candidates: Number of candidates to generate (overrides config)
            few_shot_examples: Optional list of (informal, formal) pairs for few-shot

        Returns:
            CycleConsistencyResult with best formalization and all candidates
        """
        n = num_candidates or self.config.model.num_candidates

        logger.info(f"Autoformalize: '{informal_statement[:50]}...' with {n} candidates")

        # Step 1: Generate N formalization candidates
        candidates = self._generate_candidates(
            informal_statement,
            n=n,
            few_shot_examples=few_shot_examples,
        )

        if not candidates:
            logger.warning("No candidates generated!")
            return CycleConsistencyResult(
                informal_statement=informal_statement,
                best_formalization="",
                best_score=float('-inf'),
            )

        logger.info(f"Generated {len(candidates)} candidates")

        # Step 2: Score each candidate using cycle consistency
        scored_candidates = []
        for i, candidate in enumerate(candidates):
            score_result = self._compute_cycle_consistency_score(
                informal_statement=informal_statement,
                formal_statement=candidate,
            )

            # Use normalized score if configured
            if self.config.cycle_consistency.normalize_by_length:
                score = score_result.normalized_log_prob
            else:
                score = score_result.log_prob

            scored_candidates.append(FormalizationCandidate(
                formal_statement=candidate,
                log_prob_score=score_result.log_prob,
                normalized_score=score,
                num_tokens=score_result.num_tokens,
            ))

            if self.config.verbose:
                logger.info(f"  Candidate {i+1}: score={score:.2f}, tokens={score_result.num_tokens}")

        # Step 3: Sort by score (descending) and assign ranks
        scored_candidates.sort(key=lambda x: x.normalized_score, reverse=True)
        for rank, cand in enumerate(scored_candidates):
            cand.rank = rank + 1

        # Best candidate
        best = scored_candidates[0]

        logger.info(f"Best candidate: score={best.normalized_score:.2f}")

        return CycleConsistencyResult(
            informal_statement=informal_statement,
            best_formalization=best.formal_statement,
            best_score=best.normalized_score,
            all_candidates=scored_candidates,
            num_candidates_generated=len(candidates),
            total_tokens_used=sum(c.num_tokens for c in scored_candidates),
        )

    def _generate_candidates(
        self,
        informal_statement: str,
        n: int,
        few_shot_examples: Optional[List[Tuple[str, str]]] = None,
    ) -> List[str]:
        """Generate N formalization candidates."""

        # Build prompt
        prompt = self._build_formalization_prompt(
            informal_statement,
            few_shot_examples=few_shot_examples,
        )

        # Generate N candidates
        results = self.formalizer.generate(
            prompt=prompt,
            system_prompt=self.config.prompt.formalize_system,
            temperature=self.config.model.temperature,
            max_tokens=self.config.model.max_tokens,
            n=n,
        )

        # Extract and clean formalizations
        candidates = []
        for result in results:
            formal = self._extract_lean_code(result.text)
            if formal and formal not in candidates:  # Deduplicate
                candidates.append(formal)

        return candidates

    def _build_formalization_prompt(
        self,
        informal_statement: str,
        few_shot_examples: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """Build the formalization prompt with optional few-shot examples."""

        parts = []

        # Add few-shot examples if provided
        if few_shot_examples:
            for informal, formal in few_shot_examples:
                parts.append(f"Natural language: {informal}")
                parts.append(f"Lean 4:\n```lean4\n{formal}\n```\n")

        # Add the target statement
        parts.append(
            self.config.prompt.formalize_template.format(
                informal_statement=informal_statement
            )
        )

        return "\n".join(parts)

    def _compute_cycle_consistency_score(
        self,
        informal_statement: str,
        formal_statement: str,
    ) -> LogProbResult:
        """
        Compute cycle consistency score for a formalization.

        This is the core of cycle consistency:
        Score = log P(informal_statement | "Informalize: {formal_statement}")
        """

        # Build informalization prompt
        prompt = self.config.prompt.informalize_template.format(
            formal_statement=formal_statement
        )

        # Compute log probability of original informal statement
        result = self.informalizer.compute_log_prob(
            prompt=prompt,
            completion=informal_statement,
            system_prompt=self.config.prompt.informalize_system,
        )

        return result

    def _extract_lean_code(self, text: str) -> Optional[str]:
        """Extract Lean code from LLM output."""

        # Try to find code block
        code_block_match = re.search(r'```lean4?\s*(.*?)```', text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # Try to find theorem statement directly
        theorem_match = re.search(r'(theorem\s+\w+.*?:=\s*by\s+sorry)', text, re.DOTALL)
        if theorem_match:
            return theorem_match.group(1).strip()

        # If no pattern found, try to clean and return as-is
        # (assuming the model output clean Lean code)
        cleaned = text.strip()
        if cleaned.startswith('theorem') or cleaned.startswith('import'):
            return cleaned

        return None


def autoformalize_with_cycle_consistency(
    informal_statement: str,
    formalizer: LLMInterface,
    informalizer: LLMInterface,
    num_candidates: int = 5,
    config: Optional[Config] = None,
) -> CycleConsistencyResult:
    """
    Convenience function for one-shot autoformalization.

    Args:
        informal_statement: Natural language math statement
        formalizer: LLM for generating formalizations
        informalizer: LLM for cycle consistency scoring
        num_candidates: Number of candidates to generate
        config: Optional configuration

    Returns:
        CycleConsistencyResult
    """
    cc = CycleConsistencyAutoformalization(
        formalizer=formalizer,
        informalizer=informalizer,
        config=config,
    )
    return cc.autoformalize(informal_statement, num_candidates=num_candidates)
