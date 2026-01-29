"""
Placeholder guard for Lean4 autoformalization.

Problem:
- The pipeline generates Lean statements that often end with `:= by sorry` (proof placeholder).
- Some generations collapse to the trivial statement `: True := ...`, which compiles but is useless.
- If such programs enter DB/archive, they can be sampled as parent/inspirations and propagate.

This module provides a robust, dependency-light detector for the *top-level type is exactly `True`*
case, so the runner can hard-filter it from selection.
"""

from __future__ import annotations

import os
import re
from typing import Optional


_EVOLVE_MARKER_RE = re.compile(r"^\s*(?:#|//|--)?\s*EVOLVE-BLOCK-(?:START|END)\s*$")
_HEADER_LINE_RE = re.compile(r"^\s*(?:import|open\s+scoped|open)\b")

                                                                          
_LEAN_DECL_KEYWORDS = (
    "theorem",
    "lemma",
    "example",
    "def",
    "abbrev",
    "opaque",
    "axiom",
    "instance",
)
_DECL_START_RE = re.compile(
    rf"(?m)^\s*(?:noncomputable\s+)?(?:{'|'.join(_LEAN_DECL_KEYWORDS)})\b"
)


def _strip_lean_comments(text: str) -> str:
    """Best-effort comment stripping for generated Lean snippets (non-nested block comments)."""
    if not text:
        return ""
                                                                            
    text = re.sub(r"/-.*?-/", "", text, flags=re.DOTALL)
                           
    text = re.sub(r"(?m)--.*$", "", text)
    return text


def extract_first_lean_declaration(code: str) -> str:
    """Extract the first top-level Lean declaration, skipping header/EVOLVE marker lines."""
    if not code:
        return ""
    cleaned_lines = []
    for line in code.splitlines():
        if _EVOLVE_MARKER_RE.match(line):
            continue
        if _HEADER_LINE_RE.match(line.strip()):
            continue
        cleaned_lines.append(line)
    cleaned = _strip_lean_comments("\n".join(cleaned_lines))
    if not cleaned.strip():
        return ""
    m = _DECL_START_RE.search(cleaned)
    if not m:
        return ""
                                                                                  
    return cleaned[m.start() :].strip()


def parse_lean_decl_type(decl: str) -> Optional[str]:
    """Parse the top-level type of a Lean declaration, returning a whitespace-normalized string."""
    s = (decl or "").strip()
    if not s:
        return None

                                                                                               
    depth = 0
    type_colon_idx: Optional[int] = None
    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(depth - 1, 0)
        elif ch == ":" and depth == 0:
                       
            if i + 1 < len(s) and s[i + 1] == "=":
                continue
            type_colon_idx = i
            break

    if type_colon_idx is None:
        return None

                                                                   
    depth = 0
    end_idx = len(s)
    j = type_colon_idx + 1
    while j < len(s) - 1:
        ch = s[j]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(depth - 1, 0)
        elif ch == ":" and depth == 0 and s[j + 1] == "=":
            end_idx = j
            break
        j += 1

    type_str = s[type_colon_idx + 1 : end_idx].strip()
    type_norm = re.sub(r"\s+", " ", type_str).strip()
    return type_norm or None


def is_trivial_true_placeholder(code: str) -> bool:
    """Backward-compatible alias: top-level type exactly `True`."""
    decl = extract_first_lean_declaration(code)
    typ = parse_lean_decl_type(decl)
    return typ == "True"


_ARROW_TO_TRUE_RE = re.compile(r"(?:→|->)\s*True$")
_FORALL_TRUE_RE = re.compile(r"^(?:∀|forall)\s+.*,\s*True$")
_FALSE_ARROW_RE = re.compile(r"^False\s*(?:→|->)\s*")
_REFL_EQ_RE = re.compile(
    r"^\(?\s*([A-Za-z_][A-Za-z0-9_']*)\s*\)?\s*=\s*\(?\s*\1\s*\)?$"
)


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, str(default))).strip())
    except Exception:
        return int(default)


def is_trivial_tautology_placeholder(code: str, *, level: Optional[int] = None) -> bool:
    """
    Heuristic guard for "trivial/tautological" statements that are likely placeholders.

    This is intentionally conservative and syntactic: we only analyze the first
    top-level declaration's type string (normalized).
    """
             
                   
                        
                                                   
                                       
    lvl = int(level) if level is not None else _get_env_int("AUTOFORMAL_TAUTOLOGY_GUARD_LEVEL", 1)
    if lvl <= 0:
        return False
    decl = extract_first_lean_declaration(code)
    typ = parse_lean_decl_type(decl)
    if not typ:
        return False

    if typ == "True":
        return True
    if lvl >= 2:
        if _ARROW_TO_TRUE_RE.search(typ):
            return True
        if _FORALL_TRUE_RE.search(typ):
            return True
        if _FALSE_ARROW_RE.search(typ):
            return True
    if lvl >= 3:
        if _REFL_EQ_RE.search(typ):
            return True
    return False
