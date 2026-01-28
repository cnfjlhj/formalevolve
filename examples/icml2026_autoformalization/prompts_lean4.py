"""
Lean4 Autoformalization 专用 Prompt 模块

这个模块覆盖 ShinkaEvolve 框架的默认 prompt，使其适合 Lean4 定理形式化任务。

【设计理念】
原始 shinka/prompts 是为通用代码生成/优化设计的，使用：
- "expert software engineer"
- "improve performance"
- "algorithm optimization"

这些对于 Lean4 autoformalization 是不适合的。我们需要：
- Lean4 / Mathlib 专家角色
- 数学形式化的专业知识
- 关注 compile_ok 和 semantic_ok，而不是 "performance"

【使用方式】
在 run_evo.py 中导入前调用 patch_shinka_prompts() 函数。
"""

from typing import List, Dict
import os
import re
import random

# =============================================================================
# Prompt Audit Notes (operational memory)
# =============================================================================
#
# This module is imported by `run_evo.py` and patches Shinka prompts via `patch_all()`.
# If you're trying to answer “which prompts are REALLY used in a run?”, the most
# reliable source is the per-problem `meta_memory.json` logs under the run output.
#
# Example run (CombiBench 100×100, staged full/diff then cross):
# - experiments/combibench100_call100/evolution100/
#   run_YYYYMMDD_HHMMSS__suiteX__combibench__n100__calls100__seed0__conc20__main
#
# What was used there (by scanning `runs/ours/*/meta_memory.json` system_msg fields):
# - Base: `BASE_SYSTEM_MSG`
# - Evolution: `DIFF_SYS_FORMAT`, `FULL_SYS_FORMATS[*]`, `CROSS_SYS_FORMAT`
# - User templates: `DIFF_ITER_MSG`, `FULL_ITER_MSG`, `CROSS_ITER_MSG`
#
# What was NOT observed there:
# - Kimina evolution prompts (KIMINA_*): not referenced by the main runner path.
# - Repair prompts: live in `autoformal_runner.py` and only trigger when repairs happen;
#   for many problems in that run, repairs were enabled but unused (compile_ok stayed 1).
# - Global meta recommendations: supported by this file, but may be empty depending on config.
#
# IMPORTANT: avoid hard-coding domain-specific “Mathlib patterns” (e.g., holomorphic/Complex)
# in `BASE_SYSTEM_MSG`, since CombiBench spans many domains and such hints can mislead.
#
# Quick audit command:
#   python3 -c "from prompts_lean4 import audit_prompt_usage; import json; print(json.dumps(audit_prompt_usage('PATH_TO_RUN'), indent=2))"
#

def audit_prompt_usage(run_root: str) -> dict[str, int]:
    """
    Scan a run directory and count which prompt variants were actually used.

    The function is intentionally lightweight and only relies on `meta_memory.json`
    (which contains the exact `system_msg`/`msg` that were sent to the generator).
    """
    import json
    from pathlib import Path

    def iter_system_msgs(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "system_msg" and isinstance(v, str):
                    yield v
                yield from iter_system_msgs(v)
        elif isinstance(obj, list):
            for it in obj:
                yield from iter_system_msgs(it)

    def categorize(sys_msg: str) -> str:
        s = sys_msg or ""
        if "You are given multiple Lean 4 formalization files" in s:
            return "evo_cross"
        if "You are doing a LOCAL EDIT of a Lean 4 formalization file" in s:
            return "evo_diff"
        if "Rewrite the Lean 4 formalization file to improve its formalization quality" in s:
            return "evo_full_default"
        if "Design a completely different formalization approach" in s:
            return "evo_full_different"
        if "Focus on improving the type structure and hypothesis organization" in s:
            return "evo_full_types"
        if "Align the formalization more closely with Mathlib conventions" in s:
            return "evo_full_mathlib"
        if "Create a simpler, more minimal formalization" in s:
            return "evo_full_simple"
        return "other"

    root = Path(run_root).expanduser().resolve()
    files = sorted(root.glob("runs/ours/*/meta_memory.json"))
    counts: dict[str, int] = {"meta_memory_files": len(files), "system_msg_total": 0}
    for fp in files:
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for m in iter_system_msgs(obj):
            counts["system_msg_total"] += 1
            key = categorize(m)
            counts[key] = counts.get(key, 0) + 1
            if "Global Meta Recommendations" in m:
                counts["has_global_meta_recs"] = counts.get("has_global_meta_recs", 0) + 1
            if "CRITICAL Mathlib patterns you MUST follow" in m:
                counts["has_hardcoded_critical_patterns"] = counts.get(
                    "has_hardcoded_critical_patterns", 0
                ) + 1
    return counts


# =============================================================================
# 基础系统消息（替换 BASE_SYSTEM_MSG）
# =============================================================================

BASE_SYSTEM_MSG = """You are an expert in Lean 4 theorem proving and the Mathlib library.

Your task is to improve a Lean 4 formalization for a given natural language mathematical claim.

Hard requirements (must satisfy all):
1. Output a COMPLETE Lean 4 file (imports + exactly one theorem) that can be compiled as-is.
2. The code MUST start with these two lines (in this order):
   import Mathlib
   import Aesop
3. You MAY add additional `import ...` / `open ...` / `set_option ...` lines after that if needed.
4. Do NOT include any comments.
5. Include EXACTLY ONE `theorem` declaration.
6. The theorem MUST end with `:= by sorry` (do not provide a proof).

Quality goals:
- compile_ok: The file compiles without errors
- semantic_ok: The theorem statement matches the informal mathematical meaning
"""


# =============================================================================
# DIFF 模式（精确修改）
# =============================================================================

DIFF_SYS_FORMAT = """
You are doing a LOCAL EDIT of a Lean 4 formalization file.

Goal:
- Make the smallest possible change to improve compilation and formalization quality.

Rules (local-full protocol):
* Output a SINGLE Lean 4 file inside a ```lean code fence (full file, not a diff).
* The file MUST start with:
  import Mathlib
  import Aesop
* You MAY add/remove additional imports/opens/options after those if needed to fix compilation.
* Do NOT include any comments or explanations.
* Keep changes minimal: do not change the overall mathematical claim unless required for correctness.
* Include EXACTLY ONE `theorem` declaration.
* The theorem MUST start with `theorem` (not lemma/def/example).
* The theorem MUST end with `:= by sorry`.
* Use a unique name starting with `my_` (keep the existing name if it already starts with `my_` and is unique)."""


DIFF_ITER_MSG = """# Current Lean 4 Program

Here is the current formalization we are trying to improve:

```{language}
{code_content}
```

# Evaluation Results

{performance_metrics}{text_feedback_section}

# Instructions

Improve this Lean 4 formalization file. Focus on:

1. **Compile errors**: Fix any type mismatches, unknown identifiers, or syntax issues
2. **Mathematical accuracy**: Ensure the statement correctly captures the informal meaning
3. **Mathlib conventions**: Use standard Mathlib types and type classes
4. **Type precision**: Add explicit type annotations where needed

# Output Format

Output a complete Lean 4 file inside a ```lean code fence.
The file must start with `import Mathlib` and `import Aesop`, include exactly one `theorem`,
and end with `:= by sorry`.
"""


# =============================================================================
# FULL 模式（完全重写）
# =============================================================================

# 默认重写
FULL_SYS_FORMAT_DEFAULT = """
Rewrite the Lean 4 formalization file to improve its formalization quality.

Focus on:
- Correct types and type class constraints
- Appropriate hypothesis structure
- Mathlib naming conventions
- Mathematical accuracy

Output a complete Lean 4 file inside a ```lean code fence.
The file must start with `import Mathlib` and `import Aesop`, include exactly one `theorem`,
use a unique name starting with `my_`, and end with `:= by sorry`.
"""


# 变体 1：不同的数学解释
FULL_SYS_FORMAT_DIFFERENT = """
Design a completely different formalization approach for the same mathematical claim.
Consider alternative mathematical interpretations or structures.

Think about:
- Different base types (e.g., ℕ vs ℤ vs ℝ)
- Different hypothesis organization
- Alternative ways to express the same mathematical idea

Output a complete Lean 4 file inside a ```lean code fence.
The file must start with `import Mathlib` and `import Aesop`, include exactly one `theorem`,
use a unique name starting with `my_`, and end with `:= by sorry`.
"""


# 变体 2：类型和假设优化
FULL_SYS_FORMAT_TYPES = """
Focus on improving the type structure and hypothesis organization.

Consider:
- Using more precise types (e.g., ℕ vs Nat, Finset vs List)
- Better type class constraints
- Cleaner hypothesis structure (e.g., implicit vs explicit arguments)
- Appropriate use of notation

Output a complete Lean 4 file inside a ```lean code fence.
The file must start with `import Mathlib` and `import Aesop`, include exactly one `theorem`,
use a unique name starting with `my_`, and end with `:= by sorry`.
"""


# 变体 3：Mathlib 对齐
FULL_SYS_FORMAT_MATHLIB = """
Align the formalization more closely with Mathlib conventions and existing definitions.

Consider:
- Using Mathlib naming conventions (e.g., snake_case, descriptive names)
- Preferring Mathlib's standard types and definitions
- Related theorems that exist in Mathlib

Output a complete Lean 4 file inside a ```lean code fence.
The file must start with `import Mathlib` and `import Aesop`, include exactly one `theorem`,
use a unique name starting with `my_`, and end with `:= by sorry`.
"""


# 变体 4：简化版本
FULL_SYS_FORMAT_SIMPLE = """
Create a simpler, more minimal formalization that captures the essential mathematical content.

Focus on:
- Removing unnecessary complexity
- Using the simplest types that work
- Capturing the core mathematical claim

Output a complete Lean 4 file inside a ```lean code fence.
The file must start with `import Mathlib` and `import Aesop`, include exactly one `theorem`,
use a unique name starting with `my_`, and end with `:= by sorry`.
"""


# 所有 FULL 变体列表
FULL_SYS_FORMATS = [
    FULL_SYS_FORMAT_DEFAULT,
    FULL_SYS_FORMAT_DIFFERENT,
    FULL_SYS_FORMAT_TYPES,
    FULL_SYS_FORMAT_MATHLIB,
    FULL_SYS_FORMAT_SIMPLE,
]

FULL_SYS_FORMAT_NAMES = [
    "default",
    "different_interpretation",
    "type_optimization",
    "mathlib_alignment",
    "simplification",
]


FULL_ITER_MSG = """# Current Lean 4 Program

Here is the current formalization we are trying to improve:

```{language}
{code_content}
```

# Evaluation Results

{performance_metrics}{text_feedback_section}

# Task

Rewrite the Lean 4 formalization file to improve its formalization quality.

# Output Format

Output a complete Lean 4 file inside a ```lean code fence.
The file must start with `import Mathlib` and `import Aesop`, include exactly one `theorem`,
use a unique name starting with `my_`, and end with `:= by sorry`.
"""


# =============================================================================
# CROSS 模式（组合多个形式化）
# =============================================================================

CROSS_SYS_FORMAT = """
You are given multiple Lean 4 formalization files that formalize the same mathematical claim.
Your task is to combine the best aspects of these formalizations into a single improved file.

Consider combining:
- Best type choices and annotations
- Best hypothesis structure and ordering
- Best naming conventions
- Best use of Mathlib definitions

Output a complete Lean 4 file inside a ```lean code fence.
The file must start with `import Mathlib` and `import Aesop`, include exactly one `theorem`,
use a unique name starting with `my_`, and end with `:= by sorry`.
"""


CROSS_ITER_MSG = """# Current Lean 4 Program

Here is the current formalization:

```{language}
{code_content}
```

# Evaluation Results

{performance_metrics}{text_feedback_section}

# Task

Combine the best aspects of the formalization above with the inspiration formalization below.

Consider combining:
- Type choices and annotations
- Hypothesis structure and ordering
- Naming conventions
- Use of Mathlib definitions

# Output Format

Output a complete Lean 4 file inside a ```lean code fence.
The file must start with `import Mathlib` and `import Aesop`, include exactly one `theorem`,
use a unique name starting with `my_`, and end with `:= by sorry`.
"""


# =============================================================================
# 辅助函数
# =============================================================================

def perf_str(combined_score: float, public_metrics: Dict[str, float]) -> str:
    """格式化性能指标字符串。"""
    result = f"Combined score: {combined_score:.2f}\n"
    for key, value in public_metrics.items():
        if isinstance(value, float):
            result += f"  {key}: {value:.2f}\n"
        else:
            result += f"  {key}: {value}\n"
    return result.rstrip()


def format_text_feedback_section(text_feedback) -> str:
    """格式化文本反馈部分。"""
    if not text_feedback or not str(text_feedback).strip():
        return ""

    feedback_text = text_feedback
    if isinstance(feedback_text, list):
        feedback_text = "\n".join(feedback_text)

    return f"""
# Feedback on Current Formalization

{str(feedback_text).strip()}
"""


def construct_eval_history_msg(
    inspiration_programs,
    language: str = "lean",
    include_text_feedback: bool = False,
    max_attempts: int = 2,  # Kimina 8k context limit - reduced from 3 to 2
) -> str:
    """构建历史评估程序的消息。

    限制最多 max_attempts 个尝试，避免超出模型上下文限制。
    """
    if not inspiration_programs:
        return ""

    # 限制历史尝试数量，取最后 max_attempts 个
    programs_to_show = list(inspiration_programs)[-max_attempts:]

    result = "# Previous Formalization Attempts\n\n"
    for i, prog in enumerate(programs_to_show):
        result += f"## Attempt {i + 1}\n\n"
        result += f"```{language}\n{prog.code}\n```\n\n"
        result += f"Score: {prog.combined_score:.2f}\n"
        if prog.public_metrics:
            for key, value in prog.public_metrics.items():
                if isinstance(value, float):
                    result += f"  {key}: {value:.2f}\n"
                else:
                    result += f"  {key}: {value}\n"
        result += "\n"

        if include_text_feedback and prog.text_feedback:
            feedback_text = prog.text_feedback
            if isinstance(feedback_text, list):
                feedback_text = "\n".join(feedback_text)
            if str(feedback_text).strip():
                result += f"Feedback: {str(feedback_text).strip()}\n\n"

    return result


def get_cross_component(
    archive_inspirations,
    top_k_inspirations,
    language: str = "lean",
) -> str:
    """获取交叉组件的灵感程序。"""
    all_inspirations = list(archive_inspirations) + list(top_k_inspirations)

    if not all_inspirations:
        return ""

    # 随机选择一个灵感
    inspiration = random.choice(all_inspirations)

    result = "# Crossover Inspiration\n\n"
    result += f"```{language}\n{inspiration.code}\n```\n\n"
    result += f"Score: {inspiration.combined_score:.2f}\n"
    if inspiration.public_metrics:
        for key, value in inspiration.public_metrics.items():
            if isinstance(value, float):
                result += f"  {key}: {value:.2f}\n"
            else:
                result += f"  {key}: {value}\n"

    return result


# =============================================================================
# Evolution helpers (Lean-only, local-full)
# =============================================================================


def _extract_informal_and_header(task_sys_msg: str | None) -> tuple[str, str]:
    """Extract the verbatim informal/header from `task_sys_msg`.

    This must be robust because if we fail to extract the current problem's informal statement,
    the evolution prompt may omit the target claim and the model can be easily "led astray" by
    inspirations or parent code.

    Supported formats:
    - Current `run_evo.build_task_sys_msg()` format:
        "Reference header ... ```lean ...```" + "Natural language statement: ..."
    - Legacy format used by older prompt templates:
        "**Natural language:** ... **Header:** ... **Output:**" (optionally after "### Your Task")
    """
    if not task_sys_msg:
        return "", ""

    # -------------------------------------------------------------------------
    # Format A: run_evo.build_task_sys_msg() (Lean4 autoformalization_v1)
    # -------------------------------------------------------------------------
    informal = ""
    header = ""

    # Header block (optional)
    m_header = re.search(
        r"Reference header from the dataset.*?\n```lean(?:4)?\s*\n(?P<header>.*?)\n```",
        task_sys_msg,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m_header:
        header = (m_header.group("header") or "").strip()

    # Informal statement
    m_informal = re.search(
        r"Natural language(?: statement)?\s*:\s*\n(?P<informal>.*?)(?:\n\s*\nReturn ONLY the Lean code block\.?|\Z)",
        task_sys_msg,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m_informal:
        informal = (m_informal.group("informal") or "").strip()

    # Header-only is not sufficient; if informal is missing we must try legacy formats.
    if informal:
        return informal, header

    # -------------------------------------------------------------------------
    # Format B: legacy prompt template ("**Natural language:**" / "**Header:**")
    # -------------------------------------------------------------------------
    # Prefer content after "### Your Task" if present, to avoid matching an example block.
    task_section = task_sys_msg
    your_task_match = re.search(r"### Your Task\s*\n", task_sys_msg)
    if your_task_match:
        task_section = task_sys_msg[your_task_match.end():]

    legacy_matches = list(
        re.finditer(
            r"\*\*Natural language:\*\*\n(?P<informal>.*?)\n\n\*\*Header:\*\*\n(?P<header>.*?)\n\n\*\*Output:\*\*",
            task_section,
            flags=re.DOTALL,
        )
    )
    if legacy_matches:
        m = legacy_matches[-1]
        return (m.group("informal").strip(), m.group("header").strip())

    return "", header


def _truncate_meta(meta_text: str | None) -> str:
    """Keep meta recommendations short to preserve context for full informal/header."""
    if not meta_text or meta_text in {"none", "null"}:
        return ""
    max_chars = int(os.environ.get("AUTOFORMAL_META_MAX_CHARS", "1500"))
    meta_text = str(meta_text).strip()
    if max_chars > 0 and len(meta_text) > max_chars:
        return meta_text[:max_chars].rstrip() + "\n... [truncated]"
    return meta_text


# =============================================================================
# 补丁函数：替换 shinka 的默认 prompt
# =============================================================================

def patch_shinka_prompts():
    """
    用 Lean4 专用 prompt 替换 shinka 框架的默认 prompt。

    调用时机：在 import shinka.core 之前调用。
    """
    import shinka.prompts as sp
    import shinka.prompts.prompts_base as sp_base
    import shinka.prompts.prompts_diff as sp_diff
    import shinka.prompts.prompts_full as sp_full
    import shinka.prompts.prompts_cross as sp_cross
    import shinka.prompts.prompts_meta as sp_meta

    # 替换 base
    sp.BASE_SYSTEM_MSG = BASE_SYSTEM_MSG
    sp_base.BASE_SYSTEM_MSG = BASE_SYSTEM_MSG
    sp.perf_str = perf_str
    sp_base.perf_str = perf_str
    sp.format_text_feedback_section = format_text_feedback_section
    sp_base.format_text_feedback_section = format_text_feedback_section
    sp.construct_eval_history_msg = construct_eval_history_msg
    sp_base.construct_eval_history_msg = construct_eval_history_msg

    def construct_individual_program_msg_lean(program, language: str = "lean", include_text_feedback: bool = False) -> str:
        """Lean-oriented program summary message for meta analysis (compact, includes repair metadata)."""
        msg = "# Candidate Program\n\n"
        msg += f"```{language}\n{program.code}\n```\n\n"
        msg += "Performance metrics:\n"
        msg += f"{perf_str(program.combined_score, program.public_metrics)}\n\n"
        meta = getattr(program, "metadata", {}) or {}
        repaired = meta.get("repaired")
        if repaired:
            msg += "Metadata:\n"
            msg += f"- repaired: {repaired}\n"
            if meta.get("original_error_type"):
                msg += f"- original_error_type: {meta.get('original_error_type')}\n"
            if meta.get("repair_attempts") is not None:
                msg += f"- repair_attempts: {meta.get('repair_attempts')}\n"
            msg += "\n"
        return msg

    sp.construct_individual_program_msg = construct_individual_program_msg_lean
    sp_base.construct_individual_program_msg = construct_individual_program_msg_lean

    # 替换 diff
    sp.DIFF_SYS_FORMAT = DIFF_SYS_FORMAT
    sp.DIFF_ITER_MSG = DIFF_ITER_MSG
    sp_diff.DIFF_SYS_FORMAT = DIFF_SYS_FORMAT
    sp_diff.DIFF_ITER_MSG = DIFF_ITER_MSG

    # 替换 full
    sp.FULL_SYS_FORMATS = FULL_SYS_FORMATS
    sp.FULL_ITER_MSG = FULL_ITER_MSG
    sp_full.FULL_SYS_FORMATS = FULL_SYS_FORMATS
    sp_full.FULL_ITER_MSG = FULL_ITER_MSG

    # 替换 cross
    sp.CROSS_SYS_FORMAT = CROSS_SYS_FORMAT
    sp.CROSS_ITER_MSG = CROSS_ITER_MSG
    sp.get_cross_component = get_cross_component
    sp_cross.CROSS_SYS_FORMAT = CROSS_SYS_FORMAT
    sp_cross.CROSS_ITER_MSG = CROSS_ITER_MSG
    sp_cross.get_cross_component = get_cross_component

    # --- Meta prompts (global, short; Lean4-oriented) ---
    META_STEP1_SYSTEM_MSG = (
        "You are an expert in Lean 4 theorem proving and Mathlib. "
        "Summarize ONE candidate theorem statement and its evaluation metrics. "
        "Focus on compile stability and semantic alignment cues."
    )
    META_STEP1_USER_MSG = (
        "# Candidate Program\n"
        "{individual_program_msg}\n\n"
        "# Output Format (keep it short)\n"
        "**Summary (<=1 sentence)**\n"
        "- **Compile**: [compile_ok + if repaired/original error if present]\n"
        "- **Cycle**: [cycle_score if available]\n"
        "- **Issues**: [max 2 bullets]\n"
        "- **Fix Ideas**: [max 2 bullets]\n"
    )
    META_STEP2_SYSTEM_MSG = (
        "You are an expert in Lean 4 theorem proving and Mathlib. "
        "Aggregate recent candidate summaries to extract GLOBAL insights. "
        "Be concrete and evidence-based."
    )
    META_STEP2_USER_MSG = (
        "# Recent Candidate Summaries\n"
        "{individual_summaries}\n\n"
        "# Previous Insights (if any)\n"
        "{previous_insights}\n\n"
        "# Current Best Candidate\n"
        "{best_program_info}\n\n"
        "# Instructions\n"
        "Update the global insights as short bullet points.\n"
        "Include both compile-related and semantic/cycle-related patterns.\n\n"
        "## Compile Patterns (<=4 bullets)\n"
        "- ...\n\n"
        "## Semantic/Cycle Patterns (<=4 bullets)\n"
        "- ...\n\n"
        "## Common Mathlib Pitfalls (<=4 bullets)\n"
        "- ...\n"
    )
    META_STEP3_SYSTEM_MSG = (
        "You generate short, actionable mutation rules for improving Lean 4 theorem statements."
    )
    META_STEP3_USER_MSG = (
        "# Global Insights\n"
        "{global_insights}\n\n"
        "# Previous Rules (if any)\n"
        "{previous_recommendations}\n\n"
        "# Current Best Candidate\n"
        "{best_program_info}\n\n"
        "# Output\n"
        "Return a numbered list of {max_recommendations} short mutation rules.\n"
        "Each rule must be ONE line (<=120 chars), concrete, and Lean/Mathlib-specific.\n"
        "Do NOT include explanations.\n"
    )

    sp.META_STEP1_SYSTEM_MSG = META_STEP1_SYSTEM_MSG
    sp.META_STEP1_USER_MSG = META_STEP1_USER_MSG
    sp.META_STEP2_SYSTEM_MSG = META_STEP2_SYSTEM_MSG
    sp.META_STEP2_USER_MSG = META_STEP2_USER_MSG
    sp.META_STEP3_SYSTEM_MSG = META_STEP3_SYSTEM_MSG
    sp.META_STEP3_USER_MSG = META_STEP3_USER_MSG
    sp_meta.META_STEP1_SYSTEM_MSG = META_STEP1_SYSTEM_MSG
    sp_meta.META_STEP1_USER_MSG = META_STEP1_USER_MSG
    sp_meta.META_STEP2_SYSTEM_MSG = META_STEP2_SYSTEM_MSG
    sp_meta.META_STEP2_USER_MSG = META_STEP2_USER_MSG
    sp_meta.META_STEP3_SYSTEM_MSG = META_STEP3_SYSTEM_MSG
    sp_meta.META_STEP3_USER_MSG = META_STEP3_USER_MSG

    print("[Lean4 Prompts] Successfully patched shinka prompts for Lean4 autoformalization")


# =============================================================================
# 初始化 prompt（替换 prompts_init.py）
# =============================================================================

INIT_SYSTEM_MSG = """You are a Lean4 expert with Mathlib knowledge.
Follow Mathlib naming conventions and type class patterns.
Do not invent lemmas or tactics that don't exist.
"""

INIT_USER_MSG = """Language: {language}

{task_description}
"""


def patch_init_prompts():
    """替换初始化 prompt 和 sampler 方法。"""
    import shinka.prompts.prompts_init as sp_init
    from shinka.core.sampler import PromptSampler
    import numpy as np

    sp_init.INIT_SYSTEM_MSG = INIT_SYSTEM_MSG
    sp_init.INIT_USER_MSG = INIT_USER_MSG

    # Patch PromptSampler.initial_program_prompt 避免 task_sys_msg 重复
    def patched_initial_program_prompt(self):
        """Generate the prompt for the initial program (patched for Lean4)."""
        if self.task_sys_msg is None:
            sys_msg = INIT_SYSTEM_MSG
            task_description = "The user has not provided a task description."
            user_msg = INIT_USER_MSG.format(
                language=self.language,
                task_description=task_description,
            )
        else:
            # task_sys_msg 已经是完整的任务描述，直接用作 system message
            # user message 只需要简单的触发语
            sys_msg = self.task_sys_msg
            user_msg = f"Language: {self.language}\n\nPlease provide your formalization."
        return sys_msg, user_msg

    # Patch PromptSampler.sample() - Evolution 阶段不使用 task_sys_msg 的 Example 部分
    # 原始实现把 task_sys_msg + DIFF_SYS_FORMAT 等附加在一起，导致 prompt 过长且 LLM 重复 Example
    original_sample = PromptSampler.sample

    def patched_sample(self, parent, archive_inspirations, top_k_inspirations, meta_recommendations=None):
        """Evolution prompt (Lean4): keep full informal/header; diff=local-full; cross picks inspirations uniformly."""
        import logging
        import os
        import numpy as np

        logger = logging.getLogger(__name__)

        # Base system message (avoid repeating the example in task_sys_msg)
        sys_msg = BASE_SYSTEM_MSG

        # Extract verbatim task context (informal + header) from task_sys_msg.
        #
        # IMPORTANT: The user message MUST always include the current informal statement,
        # otherwise the model can be led astray by inspirations/parent code.
        raw_task_sys_msg = getattr(self, "task_sys_msg", None)
        informal_txt, _header_txt = _extract_informal_and_header(raw_task_sys_msg)
        task_context_msg = ""
        if informal_txt:
            task_context_msg = "# Task Context (verbatim)\n\n"
            if informal_txt:
                task_context_msg += "## Natural language statement\n\n"
                task_context_msg += informal_txt.strip() + "\n\n"
        elif raw_task_sys_msg:
            # Fallback: ensure the informal claim is still present even if extraction
            # fails due to format drift (or a too-long header pushing the informal
            # beyond truncation).
            raw = str(raw_task_sys_msg).strip()
            max_chars = int(os.environ.get("AUTOFORMAL_TASK_CONTEXT_FALLBACK_MAX_CHARS", "3500"))
            fallback_informal = ""
            m = re.search(r"(?is)Natural language(?: statement)?\s*:\s*\n", raw)
            if m:
                tail = raw[m.end() :]
                cut = re.search(r"(?is)\n\s*\nReturn ONLY the Lean code block", tail)
                if cut:
                    tail = tail[: cut.start()]
                fallback_informal = tail.strip()
                if max_chars > 0 and len(fallback_informal) > max_chars:
                    fallback_informal = fallback_informal[:max_chars].rstrip() + "\n... [truncated]"

            task_context_msg = "# Task Context (fallback; informal missing)\n\n"
            if fallback_informal:
                task_context_msg += "## Natural language statement\n\n"
                task_context_msg += fallback_informal + "\n\n"
            if not fallback_informal:
                # Last resort: include a tail slice of the raw task message.
                tail = raw[m.start() :] if m else raw
                if max_chars > 0 and len(tail) > max_chars:
                    tail = "... [truncated head]\n" + tail[-max_chars:].lstrip()
                task_context_msg += "## Raw task context (tail)\n\n" + tail + "\n\n"

        # Sample patch type.
        #
        # Default: disable `cross` if no inspirations (archive/top-k) are available.
        # Demo override: allow forcing `cross` even with empty inspirations so that
        # cross-problem few-shot injection can be inspected in isolation.
        allow_cross_no_insp = str(os.environ.get("AUTOFORMAL_ALLOW_CROSS_NO_INSP", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if (not allow_cross_no_insp) and len(archive_inspirations) == 0 and len(top_k_inspirations) == 0:
            valid_types = [t for t in self.patch_types if t != "cross"]
            valid_probs = [
                p for t, p in zip(self.patch_types, self.patch_type_probs) if t != "cross"
            ]
            if valid_probs:
                valid_probs = [p / sum(valid_probs) for p in valid_probs]
                patch_type = np.random.choice(valid_types, p=valid_probs)
            else:
                patch_type = "full"
        else:
            patch_type = np.random.choice(self.patch_types, p=self.patch_type_probs)

        # Clear last cross selection for logging consistency.
        setattr(self, "_last_cross_inspiration_ids", [])

        # Patch-type specific system constraints
        if patch_type == "diff":
            sys_msg += DIFF_SYS_FORMAT
        elif patch_type == "full":
            full_variant_idx = np.random.randint(0, len(FULL_SYS_FORMATS))
            sys_msg += FULL_SYS_FORMATS[full_variant_idx]
        elif patch_type == "cross":
            sys_msg += CROSS_SYS_FORMAT

        # Text feedback section (optional)
        text_feedback_section = ""
        if self.use_text_feedback:
            text_feedback_section = "\n" + format_text_feedback_section(parent.text_feedback)

        # Build the core iteration prompt
        if patch_type == "diff":
            iter_msg = DIFF_ITER_MSG.format(
                language=self.language,
                code_content=parent.code,
                performance_metrics=perf_str(parent.combined_score, parent.public_metrics),
                text_feedback_section=text_feedback_section,
            )
        elif patch_type == "full":
            iter_msg = FULL_ITER_MSG.format(
                language=self.language,
                code_content=parent.code,
                performance_metrics=perf_str(parent.combined_score, parent.public_metrics),
                text_feedback_section=text_feedback_section,
            )
        elif patch_type == "cross":
            iter_msg = CROSS_ITER_MSG.format(
                language=self.language,
                code_content=parent.code,
                performance_metrics=perf_str(parent.combined_score, parent.public_metrics),
                text_feedback_section=text_feedback_section,
            )
            # Optional: cross-problem few-shot memory (injected by launcher per problem).
            #
            # This is intentionally scoped to `cross` to avoid contaminating diff/full steps.
            few_shot_path = os.environ.get("AUTOFORMAL_CROSS_PROBLEM_FEW_SHOTS_PATH", "").strip()
            if few_shot_path:
                try:
                    import json as _json
                    from pathlib import Path as _Path

                    raw = _Path(few_shot_path).read_text(encoding="utf-8")
                    obj = _json.loads(raw)
                    selected = obj.get("selected") if isinstance(obj, dict) else obj
                    if isinstance(selected, list) and selected:
                        fs_msg = (
                            "# Cross-problem Few-shot Examples (retrieved)\n\n"
                            "These examples come from *other* problems and are provided only as style/structure guidance.\n"
                            "Do NOT copy their theorem statements; you must formalize the CURRENT task.\n\n"
                        )
                        max_k = 2
                        for i, ex in enumerate(selected[:max_k]):
                            if not isinstance(ex, dict):
                                continue
                            informal_ex = str(ex.get("informal", "") or "").strip()
                            code_ex = str(ex.get("code", "") or "").strip()
                            if not informal_ex or not code_ex:
                                continue
                            # Keep prompts compact (Kimina context constraints).
                            if len(code_ex) > 2500:
                                code_ex = code_ex[:2500].rstrip() + "\n-- [truncated]"
                            fs_msg += f"## Example {i+1}\n\n"
                            fs_msg += "Natural language statement:\n"
                            fs_msg += informal_ex + "\n\n"
                            fs_msg += f"```{self.language}\n{code_ex}\n```\n\n"
                        iter_msg += "\n\n" + fs_msg
                except Exception as e:
                    logger.info(f"[CrossProblem] Failed to load few-shots from {few_shot_path}: {type(e).__name__}: {e}")

            # Align with circle_packing_backup/Shinka default: pick inspirations uniformly.
            all_inspirations = list(archive_inspirations or []) + list(top_k_inspirations or [])
            cross_k = int(os.environ.get("AUTOFORMAL_CROSS_K", "1"))
            cross_k = max(1, cross_k)

            selected = []
            selected_ids: List[str] = []
            if all_inspirations:
                available = list(range(len(all_inspirations)))
                for _ in range(min(cross_k, len(all_inspirations))):
                    chosen_idx = int(np.random.choice(available))
                    prog = all_inspirations[chosen_idx]
                    selected.append(prog)
                    pid = getattr(prog, "id", None)
                    if pid:
                        selected_ids.append(pid)
                    available.remove(chosen_idx)

            # Expose for downstream logging (runner can copy into metadata)
            setattr(self, "_last_cross_inspiration_ids", selected_ids)

            if selected:
                insp_msg = "# Crossover Inspirations (sampled)\n\n"
                for i, prog in enumerate(selected):
                    pid = getattr(prog, "id", "")[:8]
                    score = getattr(prog, "combined_score", 0.0) or 0.0
                    insp_msg += f"## Inspiration {i+1} (id={pid}, score={score:.2f})\n\n"
                    insp_msg += f"```{self.language}\n{prog.code}\n```\n\n"
                iter_msg += "\n\n" + insp_msg
        else:
            raise ValueError(f"Invalid patch type: {patch_type}")

        # Always attach global meta recommendations (short) to system message.
        meta_short = _truncate_meta(meta_recommendations)
        if meta_short:
            sys_msg += "\n\n# Global Meta Recommendations (short)\n" + meta_short

        final_user_msg = (task_context_msg + "\n" + iter_msg).strip()

        # Debug logging
        logger.info(f"[Evolution Prompt] patch_type={patch_type}")
        logger.info(f"[Evolution Prompt] sys_msg (first 500 chars):\n{sys_msg[:500]}...")
        logger.info(f"[Evolution Prompt] user_msg (first 500 chars):\n{final_user_msg[:500]}...")

        return (sys_msg, final_user_msg, patch_type)

    PromptSampler.sample = patched_sample
    PromptSampler.initial_program_prompt = patched_initial_program_prompt
    print("[Lean4 Prompts] Successfully patched init prompts and sampler")


def patch_all():
    """替换所有 prompt。"""
    patch_shinka_prompts()
    patch_init_prompts()


# =============================================================================
# Kimina 专用 Evolution Prompt
# =============================================================================
#
# Kimina-Autoformalizer 不理解 SEARCH/REPLACE 或 XML 标记格式。
# 保留原有的启发式信息结构，但简化输出格式要求。
#

KIMINA_EVOLUTION_SYS_MSG = """You are an expert in Lean 4 theorem proving and the Mathlib library.

Your task is to improve a Lean 4 formalization file to better formalize a given natural language mathematical claim.

Key principles:
1. Output must be a COMPLETE Lean 4 file that can be compiled as-is
2. The file must start with:
   import Mathlib
   import Aesop
3. You may add additional imports/opens/options after that if needed
4. Do NOT include any comments
5. Include exactly one theorem, ending with := by sorry

Output only the improved Lean 4 file.
"""


KIMINA_EVOLUTION_ITER_MSG = """# Current Lean 4 Program

Here is the current formalization we are trying to improve:

```{language}
{code_content}
```

# Evaluation Results

{performance_metrics}{text_feedback_section}

# Previous Attempts

{inspiration_section}

# Instructions

Improve this Lean 4 formalization file. Focus on:

1. **Mathematical accuracy**: Ensure the statement correctly captures the informal meaning
2. **Mathlib conventions**: Use standard Mathlib types and type classes
3. **Type precision**: Add explicit type annotations where needed
4. **Hypothesis structure**: Organize hypotheses clearly

# Output Format

Output a complete Lean 4 file (imports + one theorem) inside a ```lean code fence.
The file must start with import Mathlib and import Aesop, and the theorem must end with := by sorry.
"""


def build_kimina_evolution_prompt(
    parent_code: str,
    combined_score: float,
    public_metrics: Dict[str, float],
    text_feedback: str,
    archive_inspirations: list,
    top_k_inspirations: list,
    language: str = "lean",
) -> tuple:
    """
    构建 Kimina 专用的 evolution prompt。

    保留原有的启发式信息结构，但使用简化的输出格式。

    Args:
        parent_code: 父程序代码
        combined_score: 综合分数
        public_metrics: 公开指标
        text_feedback: 文本反馈
        archive_inspirations: archive 灵感程序列表
        top_k_inspirations: top-k 灵感程序列表
        language: 编程语言

    Returns:
        (system_message, user_message) 元组
    """
    sys_msg = KIMINA_EVOLUTION_SYS_MSG

    # 构建性能指标字符串
    perf_metrics = perf_str(combined_score, public_metrics)

    # 构建文本反馈部分
    feedback_section = format_text_feedback_section(text_feedback)

    # 构建灵感程序部分
    inspiration_section = ""
    all_inspirations = list(archive_inspirations or []) + list(top_k_inspirations or [])
    if all_inspirations:
        inspiration_section = "Here are some other successful formalizations for reference:\n\n"
        for i, prog in enumerate(all_inspirations[:3]):  # 最多展示 3 个
            score = prog.combined_score if hasattr(prog, 'combined_score') else 0.0
            code = prog.code if hasattr(prog, 'code') else str(prog)
            inspiration_section += f"**Attempt {i+1}** (score={score:.2f}):\n```{language}\n{code}\n```\n\n"
    else:
        inspiration_section = "(No previous attempts available)"

    # 格式化用户消息
    user_msg = KIMINA_EVOLUTION_ITER_MSG.format(
        language=language,
        code_content=parent_code,
        performance_metrics=perf_metrics,
        text_feedback_section=feedback_section,
        inspiration_section=inspiration_section,
    )

    return sys_msg, user_msg
