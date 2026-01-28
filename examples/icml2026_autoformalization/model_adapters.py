"""
模型适配层：根据不同模型选择合适的 prompt 策略和输出解析逻辑。

【设计理念】
不同的 LLM 对 prompt 的响应方式不同：
- Kimina-Autoformalizer：专门为 Lean4 训练，输出格式固定，不需要复杂指令
- 通用模型（GPT-4、Claude）：需要详细指令，按要求输出 Lean statement

通过适配层，可以在不修改核心代码的情况下支持多种模型。

【使用方式】
```python
from model_adapters import get_model_adapter

adapter = get_model_adapter(model_name)
prompt = adapter.build_prompt(informal, header, theorem_id)
statement = adapter.parse_output(raw_output)
```
"""

import re
from abc import ABC, abstractmethod
from typing import Optional, Tuple


LEAN_DECL_KEYWORDS = [
    "theorem",
    "instance",
    "definition",
    "structure",
    "class",
    "inductive",
    "classInductive",
    "opaque",
    "def",
    "lemma",
    "example",
    "axiom",
    "abbrev",
    "noncomputable",
    "irreducible_def",
]


def remove_lean_comments(text: str) -> str:
    text = re.sub(r"/-(.|\n)*?-/\s*", "", text)
    text = "\n".join([line.split("--")[0].rstrip() for line in text.splitlines()])
    return text


def strip_header_lines(text: str) -> str:
    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("import "):
            continue
        if stripped.startswith("open scoped "):
            continue
        if stripped.startswith("open "):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def strip_evolve_markers(text: str) -> str:
    cleaned = []
    for line in text.splitlines():
        if re.match(r"^\s*(?:#|//|--)?\s*EVOLVE-BLOCK-(?:START|END)\s*$", line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def extract_first_declaration(text: str) -> Optional[str]:
    """提取第一条 Lean 顶层声明。

    【修复】当找不到声明时返回 None 而非原始文本，
    避免 'none' 等无效输出被当作有效 statement 流转。
    """
    if not text:
        return None
    lines = text.splitlines()
    keyword_pattern = re.compile(
        rf"^\s*(?:noncomputable\s+)?(?:{'|'.join(LEAN_DECL_KEYWORDS)})\b"
    )
    decl_idx = None
    for i, line in enumerate(lines):
        if keyword_pattern.search(line):
            decl_idx = i
            break
    if decl_idx is None:
        return None  # 找不到声明时返回 None
    start = decl_idx
    while start > 0 and lines[start - 1].lstrip().startswith("@["):
        start -= 1
    next_idx = None
    for j in range(decl_idx + 1, len(lines)):
        if keyword_pattern.search(lines[j]):
            next_idx = j
            break
    end = next_idx if next_idx is not None else len(lines)
    return "\n".join(lines[start:end]).strip()


def normalize_lean_statement(raw_output: str) -> Optional[str]:
    text = remove_lean_comments(raw_output or "")
    text = strip_header_lines(text)
    text = strip_evolve_markers(text)
    return extract_first_declaration(text)


def normalize_lean_code(raw_output: str) -> Optional[str]:
    """Extract a complete Lean file (imports + declaration) from a model response."""
    text = (raw_output or "").strip()
    if not text:
        return None

    # Strip common thinking tags (Qwen/DeepSeek style).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Prefer explicit Lean code fences.
    fenced = re.search(r"```(?:lean4?|lean)\s*\n(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    else:
        # Fallback: any code fence.
        fenced_any = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if fenced_any:
            text = fenced_any.group(1).strip()

    text = strip_evolve_markers(text).strip()
    if not text:
        return None

    # Must contain at least one top-level declaration.
    if extract_first_declaration(text) is None:
        return None

    return text


class ModelAdapter(ABC):
    """模型适配器基类"""

    @abstractmethod
    def build_prompt(self, informal: str, header: str, theorem_id: Optional[int] = None) -> Tuple[str, str]:
        """
        构建 prompt。

        Args:
            informal: 自然语言数学陈述
            header: Lean4 header
            theorem_id: 可选的定理 ID（用于生成唯一名称）

        Returns:
            (system_message, user_message) 元组
        """
        pass

    @abstractmethod
    def parse_output(self, raw_output: str) -> Optional[str]:
        """
        解析模型输出，提取 Lean statement。

        Args:
            raw_output: 模型的原始输出

        Returns:
            提取的 Lean statement，如果解析失败返回 None
        """
        pass


class KiminaAdapter(ModelAdapter):
    """
    Kimina-Autoformalizer 模型适配器。

    【特点】
    - 专门为 Lean4 形式化训练
    - 输出格式固定：import + 注释 + theorem
    - 不需要复杂指令，简单 prompt 效果更好
    - 需要强制唯一命名避免与 Mathlib 冲突
    """

    def build_prompt(self, informal: str, header: str, theorem_id: Optional[int] = None) -> Tuple[str, str]:
        """构建 Kimina 专用的简单 prompt"""
        system_msg = (
            "You are a Lean 4 expert. Output a complete Lean 4 file (imports + one theorem). "
            "Use unique theorem names with prefix 'my_' to avoid conflicts with Mathlib."
        )

        if theorem_id is not None:
            name_hint = f"\nUse name: my_thm_{theorem_id:03d}"
        else:
            name_hint = "\nUse a unique name like my_theorem_xxx."

        user_msg = f"""Formalize in Lean 4 as a complete file.

Requirements:
- The code must start with:
  import Mathlib
  import Aesop
- You may add additional imports/opens/options after that if needed.
- Do NOT include any comments.
- Include exactly one theorem and end it with := by sorry.
{name_hint}

Natural language statement:
{informal}

Output ONLY the Lean code inside a ```lean code fence."""

        return system_msg, user_msg

    def parse_output(self, raw_output: str) -> Optional[str]:
        """
        解析 Kimina 输出，提取 Lean statement。
        """
        return normalize_lean_code(raw_output)


class GeneralAdapter(ModelAdapter):
    """
    通用模型适配器（GPT-4、Claude 等）。

    【特点】
    - 需要详细的指令
    - 能按要求输出 Python 包装格式
    - 使用 ShinkaEvolve 的标准 prompt 格式
    """

    def build_prompt(self, informal: str, header: str, theorem_id: Optional[int] = None) -> Tuple[str, str]:
        """构建通用模型的详细 prompt"""
        system_msg = f"""You are an expert in Lean 4 and Mathlib formalization.

### Task
Translate the given natural language mathematical statement into a complete Lean 4 file (imports + theorem) that faithfully formalizes the mathematical content.

### Output Protocol
- Output EXACTLY ONE ```lean code block.
- The code MUST start with:
  import Mathlib
  import Aesop
- You MAY add additional imports/opens/options after that if needed.
- Do NOT include any comments.
- Include EXACTLY ONE `theorem` declaration ending with `:= by sorry`.

---

### Example

**Natural language:**
Show that there are infinitely many prime numbers.

**Output:**
```lean
import Mathlib
import Aesop

theorem infinitude_of_primes : ∀ N, ∃ p, N ≤ p ∧ Nat.Prime p := by sorry
```

### Your Task

**Natural language:**
{informal.strip()}

**Output:**"""

        user_msg = "Please provide your formalization."

        return system_msg, user_msg

    def parse_output(self, raw_output: str) -> Optional[str]:
        """
        解析通用模型输出，提取 Lean statement。
        """
        return normalize_lean_code(raw_output)


# =============================================================================
# 适配器注册和选择
# =============================================================================

# 需要使用简易 prompt 的模型关键词列表
# 这些模型专门为 Lean4 形式化训练，有固定的输出格式
SIMPLE_PROMPT_MODELS = [
    "kimina",       # Kimina-Autoformalizer 系列
    "herald",       # Herald 系列模型
    "autoformalizer",  # 其他 autoformalizer 模型
]

# 模型名称到适配器的映射（基于关键词匹配）
_ADAPTER_PATTERNS = [(kw, KiminaAdapter) for kw in SIMPLE_PROMPT_MODELS]


def get_model_adapter(model_name: str) -> ModelAdapter:
    """
    根据模型名称获取对应的适配器。

    Args:
        model_name: 模型名称或路径

    Returns:
        对应的 ModelAdapter 实例

    Example:
        >>> adapter = get_model_adapter("Kimina-Autoformalizer-7B")
        >>> isinstance(adapter, KiminaAdapter)
        True

        >>> adapter = get_model_adapter("gpt-4-turbo")
        >>> isinstance(adapter, GeneralAdapter)
        True
    """
    model_lower = model_name.lower()

    for pattern, adapter_class in _ADAPTER_PATTERNS:
        if re.search(pattern, model_lower):
            return adapter_class()

    # 默认使用通用适配器
    return GeneralAdapter()


def is_kimina_model(model_name: str) -> bool:
    """检查是否是 Kimina 模型"""
    return isinstance(get_model_adapter(model_name), KiminaAdapter)


# =============================================================================
# 便捷函数
# =============================================================================

def formalize_with_adapter(
    model_name: str,
    informal: str,
    header: str,
    raw_output: str,
    theorem_id: Optional[int] = None,
) -> Tuple[str, str, Optional[str]]:
    """
    使用适配器处理形式化任务。

    Args:
        model_name: 模型名称
        informal: 自然语言陈述
        header: Lean4 header
        raw_output: 模型原始输出
        theorem_id: 可选的定理 ID

    Returns:
        (system_msg, user_msg, parsed_statement) 元组
    """
    adapter = get_model_adapter(model_name)
    sys_msg, user_msg = adapter.build_prompt(informal, header, theorem_id)
    statement = adapter.parse_output(raw_output)
    return sys_msg, user_msg, statement
