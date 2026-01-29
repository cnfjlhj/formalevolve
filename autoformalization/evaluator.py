"""
Unified Evaluator for Autoformalization.

Implements the (legacy) evaluation pipeline:
1) Lean compile check (hard constraint) via a real Lean server
2) BEq+ equivalence check (optional)
3) CriticLean semantic check (0/1)

Cost-aware ordering: compile → BEq+ → critic
- compile is cheap and is a hard gate
- BEq+ is medium cost
- critic is the most expensive, so it runs last
"""

import logging
import os
import sys
from typing import Tuple, Optional

                                                                         
                                                                         
BEQ_PLUS_PATH = os.environ.get("BEQ_PLUS_PATH", "").strip()
if BEQ_PLUS_PATH:
    sys.path.insert(0, BEQ_PLUS_PATH)

from .lean_env import get_lean_server, is_lean_server_available
from .models import Candidate, Problem
from .config import Config
from .critic_wrapper import critic_eval

logger = logging.getLogger(__name__)


                                                                              
                                              
                                                                              

def check_lean_compile(candidate: Candidate, timeout: int = 60) -> Tuple[bool, str]:
    """
    Check whether the candidate statement elaborates in Lean via a Lean 4 server.

    This checks syntax + types only (proof is ignored), so we force-append a
    placeholder proof when necessary.

    Args:
        candidate: candidate formalization
        timeout: timeout in seconds

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

                                                      
    if not any(kw in stmt for kw in ["theorem", "lemma", "def", "example"]):
        return False, "Missing theorem/lemma/def/example keyword"

                                                            
    if ":=" not in stmt:
        if not stmt.endswith(":="):
            stmt = stmt + " :="

                                                                       
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
                                         
                errors = [m.data for m in res.messages if m.severity == "error"]
                return False, "\n".join(errors) if errors else "Unknown Lean error"

                                   
        return False, "Unknown Lean response type"

    except TimeoutError:
        return False, f"[Timeout] Lean compile timeout after {timeout}s"
    except Exception as e:
                                             
        return False, f"[Exception] {type(e).__name__}: {e}"


                                                                              
                                      
                                                                              

def beq_plus_equiv(
    candidate: Candidate,
    gt_stmt: str,
    header: str,
    timeout: int = 60,
) -> int:
    """
    Use BEq+ to check whether the candidate is equivalent to the ground-truth formal statement.

    Important: any BEq+ exception/timeout is treated as 0 ("not proven equivalent"),
    never as a negative example (high precision, low recall).

    Args:
        candidate: candidate formalization
        gt_stmt: ground truth formal statement
        header: Lean header
        timeout: per-proof timeout in seconds

    Returns:
        1 means equivalent, 0 means "not proven equivalent" (does not distinguish false
        negatives from true non-equivalence).
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
                                                                          
        logger.debug(f"[BEq+] error: {e}")
        return 0


                                                                              
                          
                                                                              

async def evaluate_candidate(
    cand: Candidate,
    problem: Problem,
    config: Config,
) -> Candidate:
    """
    Full evaluation pipeline:
    1) Lean compile
    2) BEq+ (optional)
    3) CriticLean semantic check (0/1)

    Cost-aware ordering: candidates that fail compilation are dropped early; BEq+ runs only
    when compile_ok=1; critic runs last.

    Args:
        cand: candidate
        problem: problem instance
        config: evaluation config

    Returns:
        Updated candidate with all evaluation fields filled.
    """
                      
    ok, err = check_lean_compile(cand, timeout=config.compile_timeout)
    cand.compile_ok = ok
    cand.compile_error = err

    if not ok:
        cand.s_sem = 0
        cand.beq_flag = 0
        cand.critic_raw = ""
        return cand

                          
    if config.use_beq_plus:
        cand.beq_flag = beq_plus_equiv(
            cand,
            gt_stmt=problem.lean4_formalization,
            header=problem.lean4_src_header,
            timeout=config.compile_timeout,
        )
    else:
        cand.beq_flag = 0

                   
    s_sem, raw = await critic_eval(
        informal=problem.nl_statement,
        formal=cand.code,
    )
    cand.s_sem = s_sem          
    cand.critic_raw = raw

    return cand


async def batch_evaluate(
    candidates: list[Candidate],
    problem: Problem,
    config: Config,
) -> list[Candidate]:
    """
    Evaluate a list of candidates sequentially.

    Args:
        candidates: list of candidates
        problem: problem instance
        config: evaluation config

    Returns:
        Updated candidate list.
    """
    results = []
    for cand in candidates:
        evaluated = await evaluate_candidate(cand, problem, config)
        results.append(evaluated)
    return results


                                                                              
      
                                                                              

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)

    async def test():
        print("Testing evaluator module...")

                                           
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
            use_beq_plus=False,                               
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

                 
        from .critic_wrapper import close_session
        await close_session()

    asyncio.run(test())
