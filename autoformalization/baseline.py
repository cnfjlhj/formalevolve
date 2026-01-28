"""
Baseline Methods for Autoformalization Experiment.

按照 要求.md 实现三种方法 (统一 12 次 LLM 调用预算):
1. Naive best-of-N sampling: 一次性采样 12 个 candidate
2. Rewrite-only: 初代 4 个 + 2 轮 rewrite, 每轮 4 个 → 4 + 2×4 = 12
3. Evolution: 初代 4 个 + 2 轮 evolution, 每轮 4 个 offspring → 4 + 2×4 = 12

实验结果记录:
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
# LLM Call Counter (统一预算追踪)
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
    格式化单个 parent 的摘要，结构化地呈现其优缺点。

    格式设计原则：
    1. 状态一目了然：COMPILED vs COMPILE_FAILED
    2. 语义结果明确：CORRECT vs INCORRECT vs N/A
    3. 明确指出 strengths 和 weaknesses，帮助 LLM 做 crossover
    """
    # 状态标签
    if p.compile_ok:
        compile_status = "COMPILED"
        semantic_status = "CORRECT" if p.s_sem == 1 else "INCORRECT"
        beq_status = "YES" if p.beq_flag == 1 else "NO"
    else:
        compile_status = "COMPILE_FAILED"
        semantic_status = "N/A"
        beq_status = "N/A"

    reward = p.reward(lambda_beq)

    # 构建 header 行
    header_line = (
        f"[Parent {idx}] "
        f"Syntax: {compile_status} | "
        f"Semantic: {semantic_status} | "
        f"BEq+: {beq_status} | "
        f"Reward: {reward:.2f}"
    )

    # 代码块
    code_block = f"```lean\n{p.code.strip()}\n```"

    # 错误/反馈信息
    feedback_section = ""
    if not p.compile_ok and p.compile_error:
        feedback_section = f"\nCompilation Error:\n{_truncate(p.compile_error, 300)}"
    elif p.compile_ok and p.s_sem == 0 and p.critic_raw:
        feedback_section = f"\nSemantic Issue:\n{_truncate(p.critic_raw, 300)}"

    # 优缺点分析（帮助 LLM 理解如何利用这个 parent）
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

    核心思想：
    - 不是简单的"取 A 的一部分 + B 的一部分"
    - 而是让 LLM 从不同类型的成功和失败中学习
    - 语法对的 parent 提供结构，语义对的 parent 提供含义
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

    # 修改后版本：在 Archive 内维护 usage，避免侵入 models.py
    usage_map: Dict[str, int] = field(default_factory=dict)

    def add(self, candidate: Candidate):
        """Add candidate to archive."""
        # exact-match 去重（小预算下重复 = 直接浪费）
        existing = {c.code.strip() for c in self.candidates}
        if candidate.code.strip() in existing:
            return

        self.candidates.append(candidate)
        self.candidates.sort(key=lambda c: c.reward(self.lambda_beq), reverse=True)
        self.candidates = self.candidates[:self.max_size]

        # 修改后版本：新加入的 candidate 初始化 usage
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

    # 修改前版本：纯 top-k 贪婪，会导致 parent collapse
    # def sample_parents(self, k: int) -> List[Candidate]:
    #     """Sample k parents for evolution."""
    #     compiled = [c for c in self.candidates if c.compile_ok]
    #     if len(compiled) >= k:
    #         # Select top-k by reward
    #         return sorted(compiled, key=lambda c: c.reward(self.lambda_beq), reverse=True)[:k]
    #     # If not enough compiled, include non-compiled
    #     return self.candidates[:k]

    # 修改后版本：Evolution v2 专用采样（非贪婪 + usage penalty + 强制 diversity）
    def sample_two_parents_v2(
        self,
        top_k: int = 10,
        alpha: float = 3.0,
        beta: float = 0.3,
    ) -> List[Candidate]:
        """
        采样两个父代用于 crossover：
        - parent1：从 top-K compiled 中按 exp(alpha * reward_norm) * exp(-beta * usage) 采样（偏 exploitation）
        - parent2：从剩余 compiled 中均匀随机采样（偏 exploration），并保证 parent2 != parent1
        """
        import math

        compiled = [c for c in self.candidates if c.compile_ok]
        if not compiled:
            # 没有 compiled 的时候，退化返回空，让上层跳过/处理
            return []

        if len(compiled) == 1:
            # 只有一个父代时，上层会 fallback 到 rewrite
            return [compiled[0]]

        # candidates 已按 reward 排过序，但这里仍然显式构造 top 列表
        top = compiled[: min(top_k, len(compiled))]

        # 归一化 reward，避免尺度问题（reward ∈ {0,1,1+lambda} 也可，但仍然规范化）
        rewards = [c.reward(self.lambda_beq) for c in top]
        max_r = max(rewards) if rewards else 0.0
        norm_rewards = [r / max_r if max_r > 0 else 0.0 for r in rewards]

        # 计算采样权重：reward 越高越容易被选中，但 usage 越多越被惩罚
        weights = []
        for c, r in zip(top, norm_rewards):
            u = self.usage_map.get(c.candidate_id, 0)
            w = math.exp(alpha * r) * math.exp(-beta * u)
            weights.append(w)

        # 防止数值问题
        total_w = sum(weights)
        if total_w <= 0:
            probs = [1.0 / len(top)] * len(top)
        else:
            probs = [w / total_w for w in weights]

        # 按权重采样 parent1
        idx1 = random.choices(range(len(top)), weights=probs, k=1)[0]
        p1 = top[idx1]

        # parent2：两级 fallback（exploration ≠ 把垃圾丢进 crossover）
        # 优先：reward > 0 的 compiled（至少 semantic 对过）
        pool2 = [
            c for c in compiled
            if c.candidate_id != p1.candidate_id and c.reward(self.lambda_beq) > 0
        ]
        # 如果太少，再退回到所有 compiled
        if not pool2:
            pool2 = [c for c in compiled if c.candidate_id != p1.candidate_id]
        if not pool2:
            # 极端保护：如果 pool2 为空，就退化返回单父
            return [p1]

        p2 = random.choice(pool2)

        # 更新 usage
        self.usage_map[p1.candidate_id] = self.usage_map.get(p1.candidate_id, 0) + 1
        self.usage_map[p2.candidate_id] = self.usage_map.get(p2.candidate_id, 0) + 1

        return [p1, p2]


# ============================================================================
# Method 1: Naive Best-of-N Sampling
# ============================================================================

async def run_naive(problem: Problem, config: Config) -> Candidate:
    """
    Naive baseline: 一次性采样 N 个 candidate, 选最好的。

    预算: N = 12 次 LLM 调用
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
    Rewrite-only baseline: 单条链 self-refine (ATF-style)。

    预算: 初代 4 个 + 2 轮 rewrite, 每轮 4 个 → 4 + 2×4 = 12
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

    预算: 初代 4 个 + 2 轮 evolution, 每轮 4 个 offspring → 4 + 2×4 = 12
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

    # 记录 parent pairing 多样性（用于验证 evolution 行为）
    parent_pairs = set()
    parent_pair_counter = Counter()  # 统计每个 pair 被采样的次数

    # Evolution rounds
    for round_num in range(1, config.evolve_rounds + 1):
        logger.info(f"  Round {round_num}/{config.evolve_rounds}")

        # 修改前版本：一轮只采一次 parents + 偶数 rewrite/奇数 crossover → 退化
        # num_parents = min(2, len(archive.candidates))
        # parents = archive.sample_parents(num_parents)
        # if not parents:
        #     logger.warning("  No parents available, skipping round")
        #     continue

        # 修改后版本：每个 offspring 都重新采样 parents（避免 collapse）
        new_candidates = []
        for i in range(config.evolve_offspring):
            logger.info(f"    Offspring {i+1}/{config.evolve_offspring}")

            # 修改后版本：每个 offspring 都重新采样 parents
            parents = archive.sample_two_parents_v2(
                top_k=10,
                alpha=3.0,
                beta=0.3,
            )

            if not parents:
                logger.warning("    No compiled parents available, skipping offspring")
                continue

            if len(parents) == 1:
                # 只有一个 parent 时 fallback 到 rewrite（这是唯一允许退化的情况）
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
                # 正常情况：100% 多亲 crossover（EVOLUTION_PROMPT）
                # 记录 parent pair（用于验证多样性）
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

        # Evaluate（添加空列表防御性检查）
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

        # Log parent pairing 多样性
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
