"""
Unified Evaluator for Autoformalization.

按照 要求.md 实现完整评估 pipeline:
1. Lean compile check (硬约束) - 用真 Lean server
2. BEq+ equivalence check (可选)
3. CriticLean semantic check (0/1)

评估顺序考虑成本: compile → BEq+ → Critic
- compile 快，是硬约束
- BEq+ 中等成本
- Critic 最贵，最后调用
"""

import logging
import os
import sys
from typing import Tuple, Optional

# Optional: allow local BEq+ checkout without hard-coding personal paths.
# If unset, BEq+ is simply disabled unless installed as a Python package.
BEQ_PLUS_PATH = os.environ.get("BEQ_PLUS_PATH", "").strip()
if BEQ_PLUS_PATH:
    sys.path.insert(0, BEQ_PLUS_PATH)

from .lean_env import get_lean_server, is_lean_server_available
from .models import Candidate, Problem
from .config import Config
from .critic_wrapper import critic_eval

logger = logging.getLogger(__name__)


# ============================================================================
# Lean Compile Check (真正用 Lean server)
# ============================================================================

def check_lean_compile(candidate: Candidate, timeout: int = 60) -> Tuple[bool, str]:
    """
    用 Lean4 server 检查 theorem 语句是否能 elaboration 通过。

    只看 syntax + type，不管 proof，所以末尾强行补一个占位 proof。

    Args:
        candidate: 候选项
        timeout: 超时秒数

    Returns:
        Tuple of (compile_ok, error_message)
    """
    try:
        from lean_interact import Command
        from lean_interact.interface import CommandResponse, LeanError
    except ImportError as e:
        return False, f"[ImportError] lean_interact not installed: {e}"

    server = get_lean_server()

    header = candidate.header.strip()
    stmt = candidate.code.strip()

    if not stmt:
        return False, "Empty code"

    # 确保有 theorem/lemma 前缀
    if not any(kw in stmt for kw in ["theorem", "lemma", "def", "example"]):
        return False, "Missing theorem/lemma/def/example keyword"

    # 如果用户忘了写 `:=`，强制加上
    if ":=" not in stmt:
        if not stmt.endswith(":="):
            stmt = stmt + " :="

    # 给一个占位 proof，Lean 不会去真正检查 proof 正确性，只要结构合法就行
    if " :=" in stmt and "sorry" not in stmt.lower() and "by" not in stmt:
        stmt = stmt + " by\n  admit"

    full_code = header + "\n\n" + stmt

    try:
        cmd = Command(cmd=full_code)
        res = server.run(cmd, timeout=timeout)

        if isinstance(res, LeanError):
            return False, str(res)

        if isinstance(res, CommandResponse):
            if res.lean_code_is_valid():
                return True, ""
            else:
                # 收集 error message
                errors = [m.data for m in res.messages if m.severity == "error"]
                return False, "\n".join(errors) if errors else "Unknown Lean error"

        # 意料之外的返回
        return False, "Unknown Lean response type"

    except TimeoutError:
        return False, f"[Timeout] Lean compile timeout after {timeout}s"
    except Exception as e:
        # 避免 evaluator 整体崩掉
        return False, f"[Exception] {type(e).__name__}: {e}"


# ============================================================================
# BEq+ Equivalence Check (真正用 BEq+)
# ============================================================================

def beq_plus_equiv(
    candidate: Candidate,
    gt_stmt: str,
    header: str,
    timeout: int = 60,
) -> int:
    """
    用 BEq+ 判断 candidate 与 ground truth formal 是否可互相推导。

    重要：BEq+ 的异常、timeout，一律当 0，绝不当负例（保证高精度、低召回）。

    Args:
        candidate: 候选项
        gt_stmt: ground truth formal statement
        header: Lean header
        timeout: 每个 proof 的超时

    Returns:
        1 表示等价，0 表示"未证等价"（不区分 false negative / 真不等价）
    """
    try:
        from beq_plus import beq_plus
    except ImportError as e:
        logger.warning(f"[BEq+] beq_plus not installed: {e}")
        return 0

    server = get_lean_server()

    cand = candidate.code.strip()
    gt = gt_stmt.strip()
    hdr = header.strip()

    if not cand or not gt:
        return 0

    try:
        ok = beq_plus(
            cand,
            gt,
            hdr,
            server,
            timeout_per_proof=timeout,
            verbose=False,
        )
        return 1 if ok else 0

    except Exception as e:
        # BEq+ 挂了就当没证出来，不能当负例
        logger.debug(f"[BEq+] error: {e}")
        return 0


# ============================================================================
# Full Evaluation Pipeline
# ============================================================================

async def evaluate_candidate(
    cand: Candidate,
    problem: Problem,
    config: Config,
) -> Candidate:
    """
    完整评估 pipeline:
    1) Lean compile
    2) BEq+ (可选)
    3) CriticLean (0/1 语义一致性)

    评估顺序考虑成本: compile fail 的直接丢，BEq+ 只在 compile_ok 时跑，Critic 最后。

    Args:
        cand: 候选项
        problem: 问题
        config: 配置

    Returns:
        更新后的 Candidate (带完整评估字段)
    """
    # 1. Compile check
    ok, err = check_lean_compile(cand, timeout=config.compile_timeout)
    cand.compile_ok = ok
    cand.compile_error = err

    if not ok:
        cand.s_sem = 0
        cand.beq_flag = 0
        cand.critic_raw = ""
        return cand

    # 2. BEq+ (如果启用)
    if config.use_beq_plus:
        cand.beq_flag = beq_plus_equiv(
            cand,
            gt_stmt=problem.lean4_formalization,
            header=problem.lean4_src_header,
            timeout=config.compile_timeout,
        )
    else:
        cand.beq_flag = 0

    # 3. CriticLean
    s_sem, raw = await critic_eval(
        informal=problem.nl_statement,
        formal=cand.code,
    )
    cand.s_sem = s_sem  # 0 or 1
    cand.critic_raw = raw

    return cand


async def batch_evaluate(
    candidates: list[Candidate],
    problem: Problem,
    config: Config,
) -> list[Candidate]:
    """
    批量评估多个候选项。

    Args:
        candidates: 候选项列表
        problem: 问题
        config: 配置

    Returns:
        更新后的候选项列表
    """
    results = []
    for cand in candidates:
        evaluated = await evaluate_candidate(cand, problem, config)
        results.append(evaluated)
    return results


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)

    async def test():
        print("Testing evaluator module...")

        # Create test candidate and problem
        test_candidate = Candidate(
            code="theorem test (n : ℕ) (hn : n % 2 = 1) : 8 ∣ n^2 - 1 :=",
            header="""import Mathlib
open Fintype Group Monoid
open Set Real Ideal Polynomial
open scoped BigOperators"""
        )

        test_problem = Problem(
            id="test_001",
            nl_statement="Prove that for any odd natural number n, 8 divides n^2 - 1.",
            nl_proof="",
            lean4_src_header="""import Mathlib
open Fintype Group Monoid
open Set Real Ideal Polynomial
open scoped BigOperators""",
            lean4_formalization="theorem test (n : ℕ) (hn : n % 2 = 1) : 8 ∣ n^2 - 1 := by sorry"
        )

        test_config = Config(
            compile_timeout=60,
            use_beq_plus=False,  # Disable BEq+ for quick test
        )

        print(f"\nCandidate code: {test_candidate.code}")
        print(f"Informal: {test_problem.nl_statement}")

        result = await evaluate_candidate(test_candidate, test_problem, test_config)

        print(f"\nResults:")
        print(f"  compile_ok: {result.compile_ok}")
        print(f"  compile_error: {result.compile_error}")
        print(f"  s_sem: {result.s_sem}")
        print(f"  beq_flag: {result.beq_flag}")
        print(f"  reward: {result.reward()}")
        print(f"  soft_success: {result.soft_success}")
        print(f"  strict_success: {result.strict_success}")

        # Cleanup
        from .critic_wrapper import close_session
        await close_session()

    asyncio.run(test())
