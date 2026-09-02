"""
Baseline Methods for Autoformalization Experiment.

Implements three methods under a unified 12-call budget (original experimental protocol):
1. Naive best-of-N sampling: sample 12 candidates in one shot.
2. Rewrite-only: 4 initial + 2 rewrite rounds × 4 = 12.
3. Evolution: 4 initial + 2 evolution rounds × 4 offspring = 12.

Recorded result fields:
- naive_soft_success: compile_ok == True and s_sem == 1
- naive_strict_success: compile_ok == True and beq_flag == 1
"""

import asyncio
import json
import logging
from collections import Counter
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from .models import Candidate, Problem
from .config import Config, GENERATION_PROMPT, REWRITE_PROMPT, EVOLUTION_PROMPT
from .evaluator import evaluate_candidate, batch_evaluate
from .critic_wrapper import close_session

logger = logging.getLogger(__name__)


# ============================================================================
# LLM Call Counter (unified budget tracking)
# ============================================================================

_llm_call_count = 0


def _truncate(s: str, n: int) -> str:
    """
    Hard truncate to avoid prompt blow-up / noise conditioning.
    Keep only the prefix which usually contains the key diagnostic.
    """
    s = (s or "").strip()
    if not s:
        return ""
    return s if len(s) <= n else (s[:n] + "…[truncated]")


def reset_llm_call_count():
    """Reset LLM call counter."""
    global _llm_call_count
    _llm_call_count = 0


def get_llm_call_count() -> int:
    """Get current LLM call count."""
    return _llm_call_count


def increment_llm_call_count():
    """Increment LLM call counter."""
    global _llm_call_count
    _llm_call_count += 1


# ============================================================================
# Data Loading
# ============================================================================

def load_problems(
    dataset_name: str,
    split: str,
    num_problems: Optional[int] = None
) -> List[Problem]:
    """Load problems from HuggingFace dataset."""
    from datasets import load_dataset

    logger.info(f"Loading dataset {dataset_name} ({split})...")
    ds = load_dataset(dataset_name, split=split)

    if num_problems:
        ds = ds.select(range(min(num_problems, len(ds))))

    problems = [Problem.from_hf(ex) for ex in ds]
    logger.info(f"Loaded {len(problems)} problems")
    return problems


# ============================================================================
# LLM Generation
# ============================================================================

def query_llm(prompt: str, config: Config) -> str:
    """Query LLM for code generation."""
    import openai

    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    increment_llm_call_count()

    try:
        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

        response = client.chat.completions.create(
            model=config.llm_model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content or ""

    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return ""


def extract_lean_code(response: str) -> str:
    """Extract Lean code from LLM response."""
    code = response.strip()

    if "```lean" in code:
        code = code.split("```lean")[1].split("```")[0]
    elif "```" in code:
        parts = code.split("```")
        if len(parts) >= 2:
            code = parts[1].split("```")[0]

    return code.strip()


def generate_candidate(
    informal: str,
    header: str,
    config: Config,
    current: Optional[Candidate] = None,
    feedback: str = ""
) -> Candidate:
    """Generate a new candidate using LLM."""
    if current is None:
        prompt = GENERATION_PROMPT.format(informal=informal, header=header)
    else:
        prompt = REWRITE_PROMPT.format(
            current_statement=current.code,
            feedback=feedback or "Please improve the formalization.",
            informal=informal,
            header=header
        )

    response = query_llm(prompt, config)
    code = extract_lean_code(response)

    return Candidate(
        code=code,
        header=header,
        generation=0 if current is None else current.generation + 1,
        parent_id=None if current is None else current.candidate_id
    )


def _format_parent_summary(p: Candidate, idx: int, lambda_beq: float) -> str:
    """
    Format a single parent's summary, highlighting strengths/weaknesses.

    Design goals:
    1. Status is glanceable: COMPILED vs COMPILE_FAILED
    2. Semantic outcome is explicit: CORRECT vs INCORRECT vs N/A
    3. Provide strengths and weaknesses to help the LLM perform crossover
    """
    # Status tags
    if p.compile_ok:
        compile_status = "COMPILED"
        semantic_status = "CORRECT" if p.s_sem == 1 else "INCORRECT"
        beq_status = "YES" if p.beq_flag == 1 else "NO"
    else:
        compile_status = "COMPILE_FAILED"
        semantic_status = "N/A"
        beq_status = "N/A"

    reward = p.reward(lambda_beq)

    # Header line
    header_line = (
        f"[Parent {idx}] "
        f"Syntax: {compile_status} | "
        f"Semantic: {semantic_status} | "
        f"BEq+: {beq_status} | "
        f"Reward: {reward:.2f}"
    )

    # Code block
    code_block = f"```lean\n{p.code.strip()}\n```"

    # Error/feedback section
    feedback_section = ""
    if not p.compile_ok and p.compile_error:
        feedback_section = f"\nCompilation Error:\n{_truncate(p.compile_error, 300)}"
    elif p.compile_ok and p.s_sem == 0 and p.critic_raw:
        feedback_section = f"\nSemantic Issue:\n{_truncate(p.critic_raw, 300)}"

    # Strength/weakness analysis (to help the LLM use this parent effectively)
    strengths = []
    weaknesses = []

    if p.compile_ok:
        strengths.append("Valid Lean4 syntax")
        if p.s_sem == 1:
            strengths.append("Correct mathematical meaning")
        else:
            weaknesses.append("Mathematical interpretation may be wrong")
    else:
        weaknesses.append("Syntax error - avoid this pattern")
        strengths.append("May contain useful mathematical insight")

    analysis = ""
    if strengths:
        analysis += f"\nStrengths: {'; '.join(strengths)}"
    if weaknesses:
        analysis += f"\nWeaknesses: {'; '.join(weaknesses)}"

    return f"{header_line}\n{code_block}{feedback_section}{analysis}"


def generate_evolved_candidate(
    parents: List[Candidate],
    informal: str,
    header: str,
    config: Config,
) -> Candidate:
    """
    Generate evolved candidate from multiple parents (crossover).

    Core idea:
    - Not a naive "take parts of A + parts of B"
    - Let the LLM learn from both successes and failures across parents
    - Syntactically-correct parents provide structure; semantically-correct parents provide meaning
    """
    parent_summaries = [
        _format_parent_summary(p, i, config.lambda_beq)
        for i, p in enumerate(parents, 1)
    ]

    prompt = EVOLUTION_PROMPT.format(
        parent_summaries="\n\n".join(parent_summaries),
        informal=informal,
        header=header
    )

    response = query_llm(prompt, config)
    code = extract_lean_code(response)

    return Candidate(
        code=code,
        header=header,
        generation=max(p.generation for p in parents) + 1,
        parent_id=",".join(p.candidate_id for p in parents)
    )


# ============================================================================
# Archive Management
# ============================================================================

@dataclass
class Archive:
    """Archive for storing evaluated candidates."""
    candidates: List[Candidate] = field(default_factory=list)
    max_size: int = 100
    lambda_beq: float = 0.5

    # Maintain usage counters inside the archive (keeps models.py clean).
    usage_map: Dict[str, int] = field(default_factory=dict)

    def add(self, candidate: Candidate):
        """Add candidate to archive."""
        # Exact-match deduplication (duplicates are wasted budget under small budgets).
        existing = {c.code.strip() for c in self.candidates}
        if candidate.code.strip() in existing:
            return

        self.candidates.append(candidate)
        self.candidates.sort(key=lambda c: c.reward(self.lambda_beq), reverse=True)
        self.candidates = self.candidates[:self.max_size]

        # Initialize usage for new candidates.
        if candidate.candidate_id not in self.usage_map:
            self.usage_map[candidate.candidate_id] = 0

    def add_all(self, candidates: List[Candidate]):
        """Add multiple candidates."""
        for c in candidates:
            self.add(c)

    def get_best(self) -> Optional[Candidate]:
        """Get best candidate."""
        compiled = [c for c in self.candidates if c.compile_ok]
        if compiled:
            return max(compiled, key=lambda c: c.reward(self.lambda_beq))
        return self.candidates[0] if self.candidates else None

    def get_top_k(self, k: int, compiled_only: bool = True) -> List[Candidate]:
        """Get top-k candidates."""
        if compiled_only:
            valid = [c for c in self.candidates if c.compile_ok]
        else:
            valid = self.candidates
        return valid[:k]

    # Previous version: pure top-k greedy sampling can cause parent collapse.
    # def sample_parents(self, k: int) -> List[Candidate]:
    #     """Sample k parents for evolution."""
    #     compiled = [c for c in self.candidates if c.compile_ok]
    #     if len(compiled) >= k:
    #         # Select top-k by reward
    #         return sorted(compiled, key=lambda c: c.reward(self.lambda_beq), reverse=True)[:k]
    #     # If not enough compiled, include non-compiled
    #     return self.candidates[:k]

    # Evolution v2 sampling (non-greedy + usage penalty + enforced diversity).
    def sample_two_parents_v2(
        self,
        top_k: int = 10,
        alpha: float = 3.0,
        beta: float = 0.3,
    ) -> List[Candidate]:
        """
        Sample two parents for crossover:
        - parent1: sample from top-K compiled by exp(alpha * reward_norm) * exp(-beta * usage) (exploitation)
        - parent2: sample uniformly from the remaining compiled set (exploration), ensuring parent2 != parent1
        """
        import math

        compiled = [c for c in self.candidates if c.compile_ok]
        if not compiled:
            # No compiled candidates: return empty so the caller can skip/fallback.
            return []

        if len(compiled) == 1:
            # Only one parent available: caller may fall back to rewrite.
            return [compiled[0]]

        # Candidates are already sorted by reward, but we still take an explicit top list.
        top = compiled[: min(top_k, len(compiled))]

        # Normalize rewards to avoid scale issues.
        rewards = [c.reward(self.lambda_beq) for c in top]
        max_r = max(rewards) if rewards else 0.0
        norm_rewards = [r / max_r if max_r > 0 else 0.0 for r in rewards]

        # Sampling weights: higher reward => more likely; higher usage => more penalized.
        weights = []
        for c, r in zip(top, norm_rewards):
            u = self.usage_map.get(c.candidate_id, 0)
            w = math.exp(alpha * r) * math.exp(-beta * u)
            weights.append(w)

        # Guard numerical issues.
        total_w = sum(weights)
        if total_w <= 0:
            probs = [1.0 / len(top)] * len(top)
        else:
            probs = [w / total_w for w in weights]

        # Sample parent1 by weight.
        idx1 = random.choices(range(len(top)), weights=probs, k=1)[0]
        p1 = top[idx1]

        # parent2: two-level fallback (exploration ≠ injecting junk into crossover)
        # Prefer compiled candidates with reward > 0 (at least semantically correct).
        pool2 = [
            c for c in compiled
            if c.candidate_id != p1.candidate_id and c.reward(self.lambda_beq) > 0
        ]
        # If too few, fall back to all compiled.
        if not pool2:
            pool2 = [c for c in compiled if c.candidate_id != p1.candidate_id]
        if not pool2:
            # Extreme guard: if pool2 is empty, return a single parent.
            return [p1]

        p2 = random.choice(pool2)

        # Update usage counts.
        self.usage_map[p1.candidate_id] = self.usage_map.get(p1.candidate_id, 0) + 1
        self.usage_map[p2.candidate_id] = self.usage_map.get(p2.candidate_id, 0) + 1

        return [p1, p2]


# ============================================================================
# Method 1: Naive Best-of-N Sampling
# ============================================================================

async def run_naive(problem: Problem, config: Config) -> Candidate:
    """
    Naive baseline: sample N candidates in one shot and pick the best.

    Budget: N = 12 LLM calls.
    """
    logger.info(f"[Naive] Running for {problem.id} (N={config.naive_n})")

    candidates = []
    for i in range(config.naive_n):
        logger.info(f"  Generating candidate {i+1}/{config.naive_n}")
        cand = generate_candidate(
            informal=problem.nl_statement,
            header=problem.lean4_src_header,
            config=config
        )
        candidates.append(cand)

    # Evaluate all
    logger.info(f"  Evaluating {len(candidates)} candidates...")
    evaluated = await batch_evaluate(candidates, problem, config)

    # Select best
    archive = Archive(lambda_beq=config.lambda_beq)
    archive.add_all(evaluated)

    best = archive.get_best()
    if best:
        logger.info(f"  Best: compile={best.compile_ok}, s_sem={best.s_sem}, "
                   f"beq={best.beq_flag}, reward={best.reward(config.lambda_beq):.2f}")

    return best


# ============================================================================
# Method 2: Rewrite-only (Single Chain Self-Refine)
# ============================================================================

async def run_rewrite(problem: Problem, config: Config) -> Candidate:
    """
    Rewrite-only baseline: a single-chain self-refine (ATF-style).

    Budget: 4 initial + 2 rewrite rounds × 4 = 12.
    """
    logger.info(f"[Rewrite] Running for {problem.id}")
    logger.info(f"  Init={config.rewrite_init}, Rounds={config.rewrite_rounds}, "
               f"PerRound={config.rewrite_per_round}")

    archive = Archive(lambda_beq=config.lambda_beq)

    # Initial generation
    init_candidates = []
    for i in range(config.rewrite_init):
        logger.info(f"  Initial candidate {i+1}/{config.rewrite_init}")
        cand = generate_candidate(
            informal=problem.nl_statement,
            header=problem.lean4_src_header,
            config=config
        )
        init_candidates.append(cand)

    # Evaluate initial
    evaluated = await batch_evaluate(init_candidates, problem, config)
    archive.add_all(evaluated)

    # Rewrite rounds
    for round_num in range(1, config.rewrite_rounds + 1):
        logger.info(f"  Round {round_num}/{config.rewrite_rounds}")

        # Select best parent for rewrite
        best = archive.get_best()
        if best is None:
            best = archive.candidates[0] if archive.candidates else None

        if best is None:
            logger.warning("  No candidates available, skipping round")
            continue

        # Avoid feeding full logs/CoT into prompts (prompt bloat + noise-as-signal)
        MAX_FB_CHARS = 300

        # Generate feedback
        if not best.compile_ok and best.compile_error:
            feedback = (
                "Lean compilation failed:\n"
                f"{_truncate(best.compile_error, MAX_FB_CHARS)}\n"
                "Please fix the syntax/type errors."
            )
        elif best.s_sem == 0:
            feedback = (
                "Semantic check failed. Reason:\n"
                f"{_truncate(best.critic_raw, MAX_FB_CHARS)}\n"
                "Please improve the formalization."
            )
        else:
            feedback = "Please improve the formalization to better match the mathematical statement."

        # Generate rewrites
        new_candidates = []
        for i in range(config.rewrite_per_round):
            logger.info(f"    Rewrite {i+1}/{config.rewrite_per_round}")
            cand = generate_candidate(
                informal=problem.nl_statement,
                header=problem.lean4_src_header,
                config=config,
                current=best,
                feedback=feedback
            )
            new_candidates.append(cand)

        # Evaluate
        evaluated = await batch_evaluate(new_candidates, problem, config)
        archive.add_all(evaluated)

    best = archive.get_best()
    if best:
        logger.info(f"  Final best: compile={best.compile_ok}, s_sem={best.s_sem}, "
                   f"beq={best.beq_flag}, reward={best.reward(config.lambda_beq):.2f}")

    return best


# ============================================================================
# Method 3: Population Evolution (Shinka-style)
# ============================================================================

async def run_evolution(problem: Problem, config: Config) -> Candidate:
    """
    Population Evolution: LLM-guided population evolution (Shinka-style)。

    Budget: 4 initial + 2 evolution rounds × 4 offspring = 12.
    """
    logger.info(f"[Evolution] Running for {problem.id}")
    logger.info(f"  Init={config.evolve_init}, Rounds={config.evolve_rounds}, "
               f"Offspring={config.evolve_offspring}")

    archive = Archive(lambda_beq=config.lambda_beq)

    # Initial generation
    init_candidates = []
    for i in range(config.evolve_init):
        logger.info(f"  Initial candidate {i+1}/{config.evolve_init}")
        cand = generate_candidate(
            informal=problem.nl_statement,
            header=problem.lean4_src_header,
            config=config
        )
        init_candidates.append(cand)

    # Evaluate initial
    evaluated = await batch_evaluate(init_candidates, problem, config)
    archive.add_all(evaluated)

    # Track parent pairing diversity (helps verify evolution behavior).
    parent_pairs = set()
    parent_pair_counter = Counter()  # number of times each pair is sampled

    # Evolution rounds
    for round_num in range(1, config.evolve_rounds + 1):
        logger.info(f"  Round {round_num}/{config.evolve_rounds}")

        # Previous version: sample parents once per round + alternating rewrite/crossover => degraded.
        # num_parents = min(2, len(archive.candidates))
        # parents = archive.sample_parents(num_parents)
        # if not parents:
        #     logger.warning("  No parents available, skipping round")
        #     continue

        # Current version: re-sample parents for each offspring (avoids collapse).
        new_candidates = []
        for i in range(config.evolve_offspring):
            logger.info(f"    Offspring {i+1}/{config.evolve_offspring}")

            # Re-sample parents for each offspring.
            parents = archive.sample_two_parents_v2(
                top_k=10,
                alpha=3.0,
                beta=0.3,
            )

            if not parents:
                logger.warning("    No compiled parents available, skipping offspring")
                continue

            if len(parents) == 1:
                # Only one parent: fall back to rewrite (the only allowed degeneration case).
                parent = parents[0]
                MAX_FB_CHARS = 300
                if not parent.compile_ok and parent.compile_error:
                    feedback = (
                        "Lean compilation failed:\n"
                        f"{_truncate(parent.compile_error, MAX_FB_CHARS)}"
                    )
                elif parent.s_sem == 0:
                    feedback = (
                        "Semantic check failed:\n"
                        f"{_truncate(parent.critic_raw, MAX_FB_CHARS)}"
                    )
                else:
                    feedback = "Please improve the formalization."

                cand = generate_candidate(
                    informal=problem.nl_statement,
                    header=problem.lean4_src_header,
                    config=config,
                    current=parent,
                    feedback=feedback
                )
            else:
                # Normal case: multi-parent crossover (EVOLUTION_PROMPT).
                # Record the parent pair (for diversity audits).
                pair = tuple(sorted(p.candidate_id for p in parents))
                parent_pairs.add(pair)
                parent_pair_counter[pair] += 1

                cand = generate_evolved_candidate(
                    parents=parents,
                    informal=problem.nl_statement,
                    header=problem.lean4_src_header,
                    config=config
                )

            new_candidates.append(cand)

        # Evaluate (defensive empty-list check).
        if new_candidates:
            evaluated = await batch_evaluate(new_candidates, problem, config)
            archive.add_all(evaluated)

            # --- Round sanity stats (cheap but very informative) ---
            # compile_ok count, reward>0 count, beq=1 count
            c_compile = sum(1 for c in evaluated if c.compile_ok)
            c_reward_pos = sum(1 for c in evaluated if c.reward(config.lambda_beq) > 0)
            c_beq = sum(1 for c in evaluated if (c.compile_ok and c.beq_flag == 1))
            logger.info(
                f"  Round {round_num} stats: compile_ok={c_compile}/{len(evaluated)}, "
                f"reward>0={c_reward_pos}/{len(evaluated)}, beq=1={c_beq}/{len(evaluated)}"
            )
        else:
            logger.warning(f"  Round {round_num}: no offspring generated, skipping evaluation")

        # Log parent pairing diversity
        top_pairs = parent_pair_counter.most_common(3)
        logger.info(
            f"  Round {round_num}: unique_pairs={len(parent_pairs)}, "
            f"top_pairs={top_pairs}"
        )

    best = archive.get_best()
    if best:
        logger.info(f"  Final best: compile={best.compile_ok}, s_sem={best.s_sem}, "
                   f"beq={best.beq_flag}, reward={best.reward(config.lambda_beq):.2f}")

    return best


# ============================================================================
# Main Experiment Runner
# ============================================================================

async def run_experiment(config: Config):
    """Run full experiment comparing all three methods."""
    logger.info("=" * 60)
    logger.info("Autoformalization Experiment")
    logger.info("=" * 60)
    logger.info(f"Dataset: {config.dataset_name} ({config.dataset_split})")
    logger.info(f"Problems: {config.num_problems}")
    logger.info(f"Model: {config.llm_model}")
    logger.info(f"BEq+: {config.use_beq_plus}")

    random.seed(config.seed)

    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    problems = load_problems(
        config.dataset_name,
        config.dataset_split,
        config.num_problems
    )

    all_results = []

    try:
        for i, problem in enumerate(problems):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Problem {i+1}/{len(problems)}: {problem.id}")
            logger.info(f"{'=' * 60}")
            logger.info(f"Informal: {problem.nl_statement[:100]}...")

            problem_results = {
                "id": problem.id,
                "nl_statement": problem.nl_statement,
            }

            # Method 1: Naive
            reset_llm_call_count()
            naive_best = await run_naive(problem, config)
            naive_calls = get_llm_call_count()
            # Budget sanity check
            assert naive_calls == config.naive_n, f"Naive budget violated: {naive_calls} != {config.naive_n}"

            if naive_best:
                problem_results.update({
                    "naive_compile_ok": naive_best.compile_ok,
                    "naive_s_sem": naive_best.s_sem,
                    "naive_beq_flag": naive_best.beq_flag,
                    "naive_reward": naive_best.reward(config.lambda_beq),
                    "naive_soft_success": naive_best.soft_success,
                    "naive_strict_success": naive_best.strict_success,
                    "naive_code": naive_best.code,
                    "naive_llm_calls": naive_calls,
                })

            # Method 2: Rewrite-only
            reset_llm_call_count()
            rewrite_best = await run_rewrite(problem, config)
            rewrite_calls = get_llm_call_count()
            # Budget sanity check
            expected_rewrite = config.rewrite_init + config.rewrite_rounds * config.rewrite_per_round
            assert rewrite_calls == expected_rewrite, f"Rewrite budget violated: {rewrite_calls} != {expected_rewrite}"

            if rewrite_best:
                problem_results.update({
                    "rewrite_compile_ok": rewrite_best.compile_ok,
                    "rewrite_s_sem": rewrite_best.s_sem,
                    "rewrite_beq_flag": rewrite_best.beq_flag,
                    "rewrite_reward": rewrite_best.reward(config.lambda_beq),
                    "rewrite_soft_success": rewrite_best.soft_success,
                    "rewrite_strict_success": rewrite_best.strict_success,
                    "rewrite_code": rewrite_best.code,
                    "rewrite_llm_calls": rewrite_calls,
                })

            # Method 3: Evolution
            reset_llm_call_count()
            evolve_best = await run_evolution(problem, config)
            evolve_calls = get_llm_call_count()
            # Budget sanity check
            expected_evolve = config.evolve_init + config.evolve_rounds * config.evolve_offspring
            assert evolve_calls == expected_evolve, f"Evolution budget violated: {evolve_calls} != {expected_evolve}"

            if evolve_best:
                problem_results.update({
                    "evolve_compile_ok": evolve_best.compile_ok,
                    "evolve_s_sem": evolve_best.s_sem,
                    "evolve_beq_flag": evolve_best.beq_flag,
                    "evolve_reward": evolve_best.reward(config.lambda_beq),
                    "evolve_soft_success": evolve_best.soft_success,
                    "evolve_strict_success": evolve_best.strict_success,
                    "evolve_code": evolve_best.code,
                    "evolve_llm_calls": evolve_calls,
                })

            all_results.append(problem_results)

            # Save individual problem results
            problem_dir = results_dir / problem.id.replace("|", "_").replace("/", "_")
            problem_dir.mkdir(exist_ok=True)

            with open(problem_dir / "results.json", "w") as f:
                json.dump(problem_results, f, indent=2, ensure_ascii=False)

            if naive_best:
                with open(problem_dir / "naive_best.lean", "w") as f:
                    f.write(naive_best.full_code)
            if rewrite_best:
                with open(problem_dir / "rewrite_best.lean", "w") as f:
                    f.write(rewrite_best.full_code)
            if evolve_best:
                with open(problem_dir / "evolve_best.lean", "w") as f:
                    f.write(evolve_best.full_code)

            # Log progress
            logger.info(f"\nProblem {problem.id} Summary:")
            logger.info(f"  Naive:    soft={problem_results.get('naive_soft_success', False)}, "
                       f"strict={problem_results.get('naive_strict_success', False)}, "
                       f"calls={naive_calls}")
            logger.info(f"  Rewrite:  soft={problem_results.get('rewrite_soft_success', False)}, "
                       f"strict={problem_results.get('rewrite_strict_success', False)}, "
                       f"calls={rewrite_calls}")
            logger.info(f"  Evolve:   soft={problem_results.get('evolve_soft_success', False)}, "
                       f"strict={problem_results.get('evolve_strict_success', False)}, "
                       f"calls={evolve_calls}")

    finally:
        await close_session()

    # Save aggregate results
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Generate summary
    summary = generate_summary(all_results, config)
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print_summary(summary)

    return all_results, summary


def generate_summary(results: List[Dict], config: Config) -> Dict:
    """Generate experiment summary statistics."""
    n = len(results)

    def safe_mean(values):
        valid = [v for v in values if v is not None]
        return sum(valid) / len(valid) if valid else 0

    def safe_rate(values):
        valid = [v for v in values if v is not None]
        return sum(1 for v in valid if v) / len(valid) if valid else 0

    summary = {
        "num_problems": n,
        "config": {
            "dataset": config.dataset_name,
            "split": config.dataset_split,
            "model": config.llm_model,
            "use_beq_plus": config.use_beq_plus,
            "lambda_beq": config.lambda_beq,
        },
        "naive": {
            "compile_rate": safe_rate([r.get("naive_compile_ok") for r in results]),
            "soft_success_rate": safe_rate([r.get("naive_soft_success") for r in results]),
            "strict_success_rate": safe_rate([r.get("naive_strict_success") for r in results]),
            "mean_reward": safe_mean([r.get("naive_reward") for r in results]),
            "mean_llm_calls": safe_mean([r.get("naive_llm_calls") for r in results]),
        },
        "rewrite": {
            "compile_rate": safe_rate([r.get("rewrite_compile_ok") for r in results]),
            "soft_success_rate": safe_rate([r.get("rewrite_soft_success") for r in results]),
            "strict_success_rate": safe_rate([r.get("rewrite_strict_success") for r in results]),
            "mean_reward": safe_mean([r.get("rewrite_reward") for r in results]),
            "mean_llm_calls": safe_mean([r.get("rewrite_llm_calls") for r in results]),
        },
        "evolve": {
            "compile_rate": safe_rate([r.get("evolve_compile_ok") for r in results]),
            "soft_success_rate": safe_rate([r.get("evolve_soft_success") for r in results]),
            "strict_success_rate": safe_rate([r.get("evolve_strict_success") for r in results]),
            "mean_reward": safe_mean([r.get("evolve_reward") for r in results]),
            "mean_llm_calls": safe_mean([r.get("evolve_llm_calls") for r in results]),
        },
    }

    return summary


def print_summary(summary: Dict):
    """Print experiment summary."""
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)

    logger.info(f"Problems evaluated: {summary['num_problems']}")

    for method in ["naive", "rewrite", "evolve"]:
        m = summary[method]
        logger.info(f"\n{method.upper()}:")
        logger.info(f"  Compile rate:       {m['compile_rate']:.1%}")
        logger.info(f"  Soft success rate:  {m['soft_success_rate']:.1%}")
        logger.info(f"  Strict success rate: {m['strict_success_rate']:.1%}")
        logger.info(f"  Mean reward:        {m['mean_reward']:.3f}")
        logger.info(f"  Mean LLM calls:     {m['mean_llm_calls']:.1f}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    config = Config(
        num_problems=10,
        llm_model="gpt-5.1",
        use_beq_plus=True,
        results_dir="results/autoformalization",
    )

    asyncio.run(run_experiment(config))
