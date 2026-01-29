"""
Model adapter layer: choose a prompt strategy and output-parsing logic per model.

Design
Different LLMs respond to prompts differently:
- Kimina-Autoformalizer: trained specifically for Lean 4; fixed output format; simpler prompts work better.
- General models (e.g., GPT-4, Claude): need explicit instructions and a strict output protocol.

With this adapter layer, we can support multiple models without touching the core runner/evaluator logic.

Usage
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
    """Extract the first top-level Lean declaration.

    Fix: when no declaration is found, return None instead of the raw text,
    so outputs like 'none' are not treated as valid statements downstream.
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
        return None                                             
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

                                                       
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

                                       
    fenced = re.search(r"```(?:lean4?|lean)\s*\n(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    else:
                                   
        fenced_any = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if fenced_any:
            text = fenced_any.group(1).strip()

    text = strip_evolve_markers(text).strip()
    if not text:
        return None

                                                      
    if extract_first_declaration(text) is None:
        return None

    return text


class ModelAdapter(ABC):
    """Base class for model adapters."""

    @abstractmethod
    def build_prompt(self, informal: str, header: str, theorem_id: Optional[int] = None) -> Tuple[str, str]:
        """
        Build a prompt.

        Args:
            informal: natural-language mathematical statement
            header: Lean 4 header (imports/opens/options)
            theorem_id: optional theorem id (to help generate unique names)

        Returns:
            (system_message, user_message) tuple
        """
        pass

    @abstractmethod
    def parse_output(self, raw_output: str) -> Optional[str]:
        """
        Parse a model response and extract a Lean statement/file.

        Args:
            raw_output: raw model output

        Returns:
            extracted Lean statement/file; None if parsing fails
        """
        pass


class KiminaAdapter(ModelAdapter):
    """
    Adapter for the Kimina-Autoformalizer family.

    Characteristics:
    - Trained specifically for Lean 4 formalization.
    - Output format is relatively fixed (imports + theorem).
    - Simpler prompts often work better than overly complex instructions.
    - Enforces unique naming to avoid conflicts with Mathlib.
    """

    def build_prompt(self, informal: str, header: str, theorem_id: Optional[int] = None) -> Tuple[str, str]:
        """Build a simple Kimina-specific prompt."""
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
        Parse Kimina output and extract Lean code.
        """
        return normalize_lean_code(raw_output)


class GeneralAdapter(ModelAdapter):
    """
    Adapter for general-purpose models (e.g., GPT-4, Claude).

    Characteristics:
    - Needs explicit, detailed instructions.
    - Uses the standard prompt protocol.
    """

    def build_prompt(self, informal: str, header: str, theorem_id: Optional[int] = None) -> Tuple[str, str]:
        """Build a detailed prompt for a general-purpose model."""
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
        Parse a general-model response and extract Lean code.
        """
        return normalize_lean_code(raw_output)


                                                                               
                                
                                                                               

                                                           
                                                                                                  
SIMPLE_PROMPT_MODELS = [
    "kimina",                                
    "herald",                 
    "autoformalizer",                               
]

                                                                 
_ADAPTER_PATTERNS = [(kw, KiminaAdapter) for kw in SIMPLE_PROMPT_MODELS]


def get_model_adapter(model_name: str) -> ModelAdapter:
    """
    Pick the appropriate adapter based on a model name/path.

    Args:
        model_name: model name or local path

    Returns:
        A ModelAdapter instance

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

                               
    return GeneralAdapter()


def is_kimina_model(model_name: str) -> bool:
    """Return True if the model should use the Kimina adapter."""
    return isinstance(get_model_adapter(model_name), KiminaAdapter)


                                                                               
                     
                                                                               

def formalize_with_adapter(
    model_name: str,
    informal: str,
    header: str,
    raw_output: str,
    theorem_id: Optional[int] = None,
) -> Tuple[str, str, Optional[str]]:
    """
    Use the selected adapter to run a formalization I/O round.

    Args:
        model_name: model name
        informal: natural-language statement
        header: Lean 4 header
        raw_output: raw model output
        theorem_id: optional theorem id

    Returns:
        (system_msg, user_msg, parsed_statement) tuple
    """
    adapter = get_model_adapter(model_name)
    sys_msg, user_msg = adapter.build_prompt(informal, header, theorem_id)
    statement = adapter.parse_output(raw_output)
    return sys_msg, user_msg, statement
