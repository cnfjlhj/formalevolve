"""
AutoformalizationRunner: ShinkaEvolve runner with repair_queue support.

================================================================================
                              Design overview
================================================================================

This file implements the main evolution loop for the autoformalization system.

Why a custom runner?
The default ShinkaEvolve EvolutionRunner archives every evaluated program, but our
protocol requires:
- compile_ok=0 candidates must not enter the archive (Constraint B)
- compile_ok=0 candidates should get repair attempts
- repair calls count toward the LLM call budget (Constraint C)

Therefore we implement AutoformalizationRunner as a subclass of EvolutionRunner and
override key methods.

Key ideas:
1. Compilation failure is not an immediate discard; we allow a small number of repairs.
2. If a repair succeeds, we re-evaluate and then decide whether to archive.
3. Every candidate's outcome is recorded (archived vs failure log).

Constraint mapping:
- [A] Novelty only for compile_ok=1: only compile-ok candidates undergo novelty checks
- [B] Archive only compile_ok=1: only compile-ok candidates are stored in the archive
- [C] Repair counts toward budget: repair calls are debited to the call budget
- [E] Semantic repair (optional): triggered only when compile_ok=1 & semantic_ok=0 and
  also counts toward the call budget

================================================================================
                              Protocol version: v1.2
================================================================================
"""

                                                                               
                          
                                                                               
import json
import hashlib
import logging
import os
import re
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

                                                                               
                           
                                                                               
                                                                                     
                                                    
                                      
from shinka.core.runner import (
    EvolutionRunner,
    EvolutionConfig,
    RunningJob,
    FOLDER_PREFIX,
    _lean_local_diff_stats,
)
                                        
                                
from shinka.database import DatabaseConfig, Program
                              
from shinka.launch import JobConfig
                                                          
from shinka.llm import extract_between
from placeholder_guard import is_trivial_tautology_placeholder

                                                                   
try:
    from model_adapters import get_model_adapter, is_kimina_model
except ImportError:
                                                                 
    def get_model_adapter(model_name):
        return None
    def is_kimina_model(model_name):
        return False

logger = logging.getLogger(__name__)


def _env_truthy(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str) -> Optional[int]:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_csv_list(name: str) -> List[str]:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def overwrite_edit_diff_for_final_candidate(
    *,
    patch_path: Optional[Path],
    original_code: str,
    final_code: str,
    lang_ext: str,
) -> Optional[str]:
    """Overwrite `edit.diff` to reflect the FINAL evaluated candidate code.

    Why:
    - `apply_full_patch` writes `edit.diff` for the raw patch application.
    - Later steps (e.g. parent preamble inheritance or EvolAST fallback) can
      overwrite `main.lean`, making the on-disk `edit.diff` misleading.
    """
    if patch_path is None:
        return None
    try:
        from shinka.edit.apply_diff import write_git_diff

        diff_path = Path(patch_path)
        write_git_diff(
            original_code,
            final_code,
            filename=f"original.{lang_ext}",
            out_path=diff_path,
        )
        return diff_path.read_text("utf-8")
    except Exception:
        return None


@contextmanager
def _temporary_env(updates: Dict[str, Optional[str]]):
    """Temporarily override os.environ, restoring prior values on exit."""
    old: Dict[str, Optional[str]] = {}
    for k, v in (updates or {}).items():
        old[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, prev in old.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


def _read_metrics_json(results_dir: str) -> Dict[str, Any]:
    try:
        path = Path(results_dir) / "metrics.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


                                                                               
               
                                                                               

@dataclass
class RepairConfig:
    """
    Compile-repair configuration.

    Design notes:
    - max_repair_attempts defaults to 2 to avoid infinite loops.
    - repair_temperature uses a moderate temperature to reduce no-op retries.
    - enabled can be turned off via CLI for debugging/ablations.
    """
                                                                            
    num_init_candidates_gen0: int = 3

                                               
                                                                      
    max_repair_attempts: int = 2
                                                           
                                                                               
                                                                 
    max_repair_attempts_gen0: int = 5
    repair_temperature: float = 0.7
    enabled: bool = True


@dataclass
class TerminationConfig:
    """
    Termination policy configuration.

    Two types of termination:
    1) Budget exhausted (hard stop): resources are depleted
       - max_llm_calls: generator call budget
       - max_evals: evaluation budget
       - max_time_seconds: wall-clock time limit

    2) Exploration stagnation (soft stop): no progress for a while
       - stagnation_generations: number of generations without improvement
       - triggers a "soft reset" (e.g., more diversity / crossover)
       - after too many soft resets, the run stops
    """
                               
    max_llm_calls: Optional[int] = None
    max_evals: Optional[int] = None
    max_time_seconds: Optional[float] = None

                                      
    stagnation_generations: int = 5
    max_soft_resets: int = 3

                        
    reset_temperature_boost: float = 0.2
    reset_crossover_boost: float = 0.1
                                                                                
                                                             
    reset_parent_usage_penalty_boost: float = 0.5
    reset_parent_usage_penalty_max: float = 5.0


@dataclass
class RepairQueueItem:
    """
    One item in the repair queue.

    When a candidate fails compilation it is not immediately discarded; instead it can be
    repaired and re-evaluated.

    Field notes:
    - program_id: unique identifier
    - exec_fname: path to the program file
    - results_dir: output directory
    - generation: originating generation
    - parent_id: parent program id (for tracing)
    - compile_error_type: error type (used for repair prompt)
    - compile_error_msg: error message (used for repair prompt)
    - original_code: original Lean code (should contain imports + theorem)
    - repair_attempts: number of repair attempts so far
    - total_repair_cost: accumulated API cost during repair
    """
    program_id: str
    exec_fname: str
    results_dir: str
    generation: int
    parent_id: Optional[str]
    compile_error_type: str
    compile_error_msg: str
    original_code: str
    repair_attempts: int = 0
    total_repair_cost: float = 0.0
    total_repair_llm_calls_used: int = 0


                                                                               
                               
                                                                               

@dataclass
class FailureRecord:
    """
    A failure record.

    Constraint B requires that compile_ok=0 candidates do not enter the archive, but we
    still record them for:
    1) analysis (failure distribution / causes)
    2) future prompt/data improvements
    3) debugging and traceability

    final_status values:
    - "repair_success": repair succeeded; candidate moved into the archive
    - "repair_exhausted": repairs exhausted and still failing
    - "no_repair": repair disabled; failure recorded directly
    """
    program_id: str
    generation: int
    compile_error_type: str
    compile_error_msg: str
    statement: str
    repair_attempts: int
    repair_llm_calls_used: int
    final_status: str
    timestamp: float


class FailureBuffer:
    """
    Buffer for failed candidates.

    Implements Constraint B: compile failures are recorded here and never participate
    in parent sampling.
    """

    def __init__(self, buffer_path: Optional[str] = None):
        """
        Initialize the failure buffer.

        Args:
            buffer_path: Optional JSON path. If set, writes on every update.
        """
        self.records: List[FailureRecord] = []
        self.buffer_path = buffer_path

    def add(self, record: FailureRecord):
        """Add a failure record (and auto-save if configured)."""
        self.records.append(record)
        if self.buffer_path:
            self._save()

    def _save(self):
        """Write the buffer to a JSON file."""
        if not self.buffer_path:
            return
        data = [
            {
                "program_id": r.program_id,
                "generation": r.generation,
                "compile_error_type": r.compile_error_type,
                "compile_error_msg": r.compile_error_msg,
                "statement": r.statement[:500],                          
                "repair_attempts": r.repair_attempts,
                "repair_llm_calls_used": int(getattr(r, "repair_llm_calls_used", 0) or 0),
                "final_status": r.final_status,
                "timestamp": r.timestamp,
            }
            for r in self.records
        ]
        with open(self.buffer_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            - total: total records
            - repair_success: number of repair-success cases
            - repair_exhausted: number of repair-exhausted cases
            - no_repair: number of cases without repair
            - error_types: distribution of compile error types
        """
        if not self.records:
            return {"total": 0}

        repair_success = sum(1 for r in self.records if r.final_status == "repair_success")
        repair_exhausted = sum(1 for r in self.records if r.final_status == "repair_exhausted")
        no_repair = sum(1 for r in self.records if r.final_status == "no_repair")

        error_types = {}
        for r in self.records:
            error_types[r.compile_error_type] = error_types.get(r.compile_error_type, 0) + 1

        return {
            "total": len(self.records),
            "repair_success": repair_success,
            "repair_exhausted": repair_exhausted,
            "no_repair": no_repair,
            "error_types": error_types,
        }


                                                                               
                            
                                                                               

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
    """
    Remove Lean comments (single-line and block).
    """
    text = re.sub(r"/-(.|\n)*?-/\s*", "", text)
    text = "\n".join([line.split("--")[0].rstrip() for line in text.splitlines()])
    return text


def strip_header_lines(text: str) -> str:
    """Strip import/open/open scoped lines to avoid header duplication."""
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
    """Strip EVOLVE-BLOCK marker lines."""
    cleaned = []
    for line in text.splitlines():
        if re.match(r"^\s*(?:#|//|--)?\s*EVOLVE-BLOCK-(?:START|END)\s*$", line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def extract_first_declaration(text: str) -> str:
    """Extract the first top-level Lean declaration.

    Fix: when no declaration is found, return an empty string instead of the raw text,
    so placeholders like 'none' are not treated as valid statements downstream.
    """
    if not text:
        return ""
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
        return ""                                                          
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


def normalize_lean_statement(text: str) -> str:
    """Normalize a model response into a compilable Lean statement.

    Fix: filter common placeholders to block invalid "no result" outputs like 'none'.
    """
    if not text:
        return ""

                                                                                           
    INVALID_PLACEHOLDERS = {"none", "null", "nil", "n/a", "na", ""}
    text_lower = text.strip().lower()
    if text_lower in INVALID_PLACEHOLDERS:
        return ""

    cleaned = remove_lean_comments(text)
    cleaned = strip_header_lines(cleaned)
    cleaned = strip_evolve_markers(cleaned)
    result = extract_first_declaration(cleaned)

                                                             
    if result and result.strip().lower() in INVALID_PLACEHOLDERS:
        return ""

    return result


def normalize_lean_code(text: str) -> str:
    """Normalize a Lean snippet into a compile-ready Lean file (keep imports).

    This is used for repair workflows where the candidate is expected to output a
    complete Lean file (imports + theorem). We only strip:
    - markdown code fences
    - EVOLVE-BLOCK markers
    """
    if not text:
        return ""

    raw = str(text)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    m = re.search(r"```(?:lean4?|lean)\s*\n(.*?)```", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    else:
        m2 = re.search(r"```\s*\n(.*?)```", raw, re.DOTALL)
        if m2:
            raw = m2.group(1)

    raw = strip_evolve_markers(raw)
    return raw.strip()


def extract_best_lean_code_block_with_source(
    raw_content: Optional[str],
) -> tuple[Optional[str], str]:
    """Extract the best Lean code block from a model response.

    Repair workflows often include prompts that themselves contain fenced Lean code
    blocks (e.g. the original failing code). Some models echo the prompt before
    producing the final answer, so we prefer the *last* fenced block that contains
    a top-level declaration.
    """
    if not raw_content:
        return None, ""

    text = str(raw_content)
    blocks: list[str] = []
    fence_source = ""

    for m in re.finditer(r"```(?:lean4?|lean)\s*\n(.*?)```", text, re.DOTALL):
        blocks.append(m.group(1))
    if blocks:
        fence_source = "lean_fence"
    else:
        for m in re.finditer(r"```\s*\n(.*?)```", text, re.DOTALL):
            blocks.append(m.group(1))
        if blocks:
            fence_source = "fence"

    for blk in reversed(blocks):
        cand = normalize_lean_code(blk)
        if normalize_lean_statement(cand):
            return cand, fence_source

                                                                               
                                                                                
                                                               
    text = str(raw_content)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = strip_evolve_markers(text)
    text = remove_lean_comments(text)

    if not text:
        return None, ""

    lines = text.splitlines()
    keyword_pattern = re.compile(
        rf"^\s*(?:noncomputable\s+)?(?:{'|'.join(LEAN_DECL_KEYWORDS)})\b"
    )
    theorem_pattern = re.compile(r"^\s*(?:noncomputable\s+)?theorem\b")

    decl_indices: list[int] = [i for i, ln in enumerate(lines) if keyword_pattern.search(ln)]
    if not decl_indices:
        return None, ""

    theorem_indices = [i for i in decl_indices if theorem_pattern.search(lines[i])]
    decl_idx = theorem_indices[-1] if theorem_indices else decl_indices[-1]

    start = decl_idx
    while start > 0 and lines[start - 1].lstrip().startswith("@["):
        start -= 1

    def _cut_end_at_sorry(start_idx: int) -> int:
        for i in range(start_idx, len(lines)):
            if re.search(r":=\s*by\s*sorry\b", lines[i]):
                return i + 1
        for i in range(start_idx, len(lines)):
            if re.search(r"\bby\s+sorry\b", lines[i]):
                return i + 1
        last_sorry = None
        for i in range(start_idx, len(lines)):
            if re.search(r"\bsorry\b", lines[i]):
                last_sorry = i
        return (last_sorry + 1) if last_sorry is not None else len(lines)

    end = _cut_end_at_sorry(start)

                                                                                
                                                   
    import_mathlib_idx = None
    import_any_idx = None
    for i in range(0, start):
        stripped = lines[i].lstrip()
        if stripped.startswith("import Mathlib"):
            import_mathlib_idx = i
        elif stripped.startswith("import "):
            import_any_idx = i
    header_start = (
        import_mathlib_idx
        if import_mathlib_idx is not None
        else (import_any_idx if import_any_idx is not None else 0)
    )

    def _is_preamble_line(line: str) -> bool:
        s = line.strip()
        if not s:
            return True
        if s.startswith("@["):
            return True
        return bool(
            re.match(
                r"^\s*(?:import\b|open\b|open\s+scoped\b|set_option\b|attribute\b|namespace\b|section\b|universe\b|variable\b|variables\b)",
                line,
            )
        )

    header_lines = [ln for ln in lines[header_start:start] if _is_preamble_line(ln)]
    decl_lines = lines[start:end]

                                                                   
    has_mathlib = any(ln.strip().startswith("import Mathlib") for ln in header_lines)
    has_aesop = any(ln.strip().startswith("import Aesop") for ln in header_lines)
    if not has_mathlib:
        header_lines = ["import Mathlib"] + header_lines
    if not has_aesop:
        insert_at = 1 if header_lines and header_lines[0].strip().startswith("import Mathlib") else 0
        header_lines = header_lines[:insert_at] + ["import Aesop"] + header_lines[insert_at:]

    code = "\n".join([*header_lines, "", *decl_lines]).strip()
    return (code, "no_fence") if normalize_lean_statement(code) else (None, "")


def extract_best_lean_code_block(raw_content: Optional[str]) -> Optional[str]:
    code, _source = extract_best_lean_code_block_with_source(raw_content)
    return code


def extract_lean_preamble(code: str) -> str:
    """Extract the preamble (imports/opens/options) before the first declaration."""
    s = normalize_lean_code(code)
    if not s:
        return ""
    keyword_pattern = re.compile(
        rf"^\s*(?:noncomputable\s+)?(?:{'|'.join(LEAN_DECL_KEYWORDS)})\b"
    )
    preamble_lines: list[str] = []
    for line in s.splitlines():
        if keyword_pattern.search(line):
            break
        preamble_lines.append(line)
    return "\n".join(preamble_lines).strip()


def extract_lean_preamble_for_inheritance(code: str) -> str:
    """Extract a Lean preamble suitable for *inheritance*.

    Differences vs `extract_lean_preamble`:
    - Ignores standalone attribute annotations like `@[simp]` which belong to the
      following declaration, not the file preamble.
    - Still strips EVOLVE markers via `normalize_lean_code`.
    """
    s = normalize_lean_code(code)
    if not s:
        return ""
    keyword_pattern = re.compile(
        rf"^\s*(?:noncomputable\s+)?(?:{'|'.join(LEAN_DECL_KEYWORDS)})\b"
    )
    preamble_lines: list[str] = []
    for line in s.splitlines():
        if keyword_pattern.search(line):
            break
        if line.lstrip().startswith("@["):
            continue
        preamble_lines.append(line)
    return "\n".join(preamble_lines).strip()


_LEAN_PREAMBLE_DIRECTIVE_RE = re.compile(
    r"^\s*(?:import\b|open\b|open\s+scoped\b|set_option\b|attribute\b|namespace\b|section\b|universe\b|variable\b|variables\b)"
)


def _lean_has_explicit_preamble(code: str) -> bool:
    """Return True iff the candidate explicitly contains a Lean preamble.

    Policy note (experiment semantics):
    - `@[...]` attribute annotations are NOT considered "header/preamble".
    - EVOLVE markers are ignored.
    """
    preamble = extract_lean_preamble_for_inheritance(code or "")
    if not preamble.strip():
        return False
    for raw in preamble.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("--"):
            continue
        if line.startswith("@["):
            continue
        if _LEAN_PREAMBLE_DIRECTIVE_RE.match(raw):
            return True
    return False


def rebase_lean_candidate_on_parent_preamble_if_missing(
    *,
    candidate_code: str,
    parent_code: str,
) -> tuple[str, bool, str]:
    """If the candidate has NO explicit preamble, inherit the parent's preamble.

    This implements the agreed policy:
    - Only when the patch model outputs a theorem without any header/preamble,
      we default to reusing the parent's header.
    - If the candidate includes any explicit preamble directive (import/open/...),
      we keep it unchanged (even if it's incomplete).
    """
    cand = str(candidate_code or "")
    parent = str(parent_code or "")
    if not cand.strip() or not parent.strip():
        return cand, False, "empty_input"

    if _lean_has_explicit_preamble(cand):
        return cand, False, "candidate_has_preamble"

    parent_preamble = extract_lean_preamble_for_inheritance(parent)
    if not parent_preamble.strip():
        return cand, False, "parent_preamble_empty"

    stmt = normalize_lean_statement(cand)
    if not stmt.strip():
        return cand, False, "candidate_statement_empty"

    body = "\n\n".join([parent_preamble.strip(), stmt.strip()]).strip()
    has_markers = ("EVOLVE-BLOCK-START" in cand) or ("EVOLVE-BLOCK-START" in parent)
    if has_markers and ("EVOLVE-BLOCK-START" not in body) and ("EVOLVE-BLOCK-END" not in body):
        body = "-- EVOLVE-BLOCK-START\n" + body + "\n-- EVOLVE-BLOCK-END\n"
    return body, True, "inherited_parent_preamble"


def _strip_attribute_lines(stmt: str) -> str:
                                              
             
              
    if not stmt:
        return ""
    return re.sub(r"(?m)^\s*@\[.*\]\s*\n?", "", stmt).strip()


def _canonicalize_lean_for_exact_match(text: str) -> str:
    """Canonicalize a Lean program for *exact duplicate* detection (hard gate).

    Scope:
    - Used only for equality checks (offspring==parent / offspring==cross inspiration).
    - NOT a GTED substitute (no structural metric; no similarity scoring).
    """
    code = normalize_lean_code(text or "")
    if not code:
        return ""

    preamble = extract_lean_preamble(code)
    stmt = normalize_lean_statement(code)
    stmt = _strip_attribute_lines(stmt)
    if not stmt:
        return ""

                                                                                     
    stmt = re.sub(r"\s+", " ", stmt).strip()

                                                                                       
    stmt = re.sub(r"^(noncomputable\s+)?lemma\b", r"\1theorem", stmt)

                                                                                         
    stmt = re.sub(
        r"^(noncomputable\s+)?(theorem|def|definition|example|axiom|abbrev|instance)\s+[^\s:(]+",
        r"\1\2 __NAME__",
        stmt,
    )

    pre_lines = [ln.strip() for ln in (preamble or "").splitlines() if ln.strip()]
    pre_norm = "\n".join(pre_lines)
    return (pre_norm + "\n" + stmt) if pre_norm else stmt


def _is_theorem_statement(text: str) -> bool:
    stmt = normalize_lean_statement(text or "")
    stmt = _strip_attribute_lines(stmt)
    if not stmt:
        return False
    return bool(re.match(r"^\s*(?:noncomputable\s+)?theorem\b", stmt))


def _check_required_lean_import_prefix(code: str) -> tuple[bool, str]:
    """Check whether a Lean program starts with the required imports.

    We allow leading blank lines and comment lines (including EVOLVE markers), but
    the first two *non-empty, non-comment* lines must be:
      1) import Mathlib
      2) import Aesop
    """
    s = normalize_lean_code(code or "")
    if not s:
        return False, "empty_program"

    meaningful: list[str] = []
    for raw in s.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("--"):
            continue
        meaningful.append(line)
        if len(meaningful) >= 2:
            break

    if len(meaningful) < 2:
        return False, "missing_required_import_lines"
    if meaningful[0] != "import Mathlib":
        return False, "missing_import_mathlib"
    if meaningful[1] != "import Aesop":
        return False, "missing_import_aesop"
    return True, ""


def build_repair_prompt(
    original_code: str,
    compile_error_type: str,
    compile_error_msg: str,
    informal: str,
    reference_header: str = "",
) -> tuple[str, str]:
    """
    Build a compilation-repair prompt.

    Motivation:
    When compilation fails, we ask the LLM to repair the errors. This helper constructs
    the repair prompt.

    Prompt design principles
    1. Provide full context: informal statement + current Lean code (imports + theorem)
    2. Provide explicit compiler feedback: error_type and error_msg
    3. Constrain the scope: "fix compilation only; do not change mathematical meaning"

    Why emphasize "do not change meaning"?
    If the LLM makes large edits while repairing, it may:
    - change the mathematical meaning
    - introduce new semantic errors
    - push the search away from correct regions

    Therefore we explicitly constrain the repair to minimal necessary compilation fixes.

    Args:
        original_code: the original Lean code with compile errors (imports + theorem)
        compile_error_type: coarse error category (e.g., type_mismatch)
        compile_error_msg: compiler error message
        informal: natural-language statement (semantic target)

    Returns:
        (system_message, user_message) tuple
    """
    sys_msg = """You are an expert in Lean 4 theorem proving and Mathlib.

You are doing SYNTAX / COMPILATION REPAIR.

You will receive:
- A natural language statement (the semantic target)
- A Lean 4 file that fails to compile
- Compiler error feedback

Your job:
- Produce a corrected Lean 4 file that compiles.

IMPORTANT:
- Fix compilation only; do NOT change the intended mathematical meaning.
- Keep changes minimal (types, identifiers, imports, binder annotations, etc.).
"""

    orig_code = (original_code or "").strip()
                                                                             
    ref_hdr_block = ""

    user_msg = f"""Natural language statement:
{informal.strip()}
{ref_hdr_block}
Current Lean code (does NOT compile):
<CURRENT_CODE>
{orig_code}
</CURRENT_CODE>

Compiler feedback:
- Error type: {compile_error_type}
- Error message:
```
{compile_error_msg[:500]}
```

IMPORTANT:
- The <CURRENT_CODE> above is INPUT; do NOT repeat it verbatim.
- Your output MUST address the compiler feedback and produce compiling Lean code.
- If you output the same code again, it will be treated as a failed repair.

Hint (often relevant for type mismatches in equalities):
- If a goal looks like `f = f 0` where `f : α → β`, then `f 0 : β` is not a function.
  A minimal compilation fix is to make the RHS a function (e.g. `f = fun _ => f 0`)
  or rewrite to a pointwise statement (e.g. `∀ z, f z = f 0`), keeping the intended meaning.

Output requirements:
1) Output EXACTLY ONE ```lean code block (no extra text).
2) The code MUST start with:
   import Mathlib
   import Aesop
3) You may add additional imports/opens/options after that if needed.
4) Do NOT include any comments.
5) Include EXACTLY ONE theorem, and end it with `:= by sorry`.
"""

    return sys_msg, user_msg


def build_semantic_repair_prompt(
    *,
    original_code: str,
    informal: str,
    accuracy_confirmation: str,
    reference_header: str = "",
    previous_compile_error: str = "",
) -> tuple[str, str]:
    sys_msg = """You are an expert in mathematics and Lean 4 (Mathlib).

You are doing SEMANTIC REPAIR.

You will receive:
- A natural language statement (the semantic target)
- A Lean 4 file that is intended to formalize it
- Critic feedback (Accuracy Confirmation) describing mismatches

Your job:
- Modify the Lean code so that the theorem statement matches the natural language statement.

Rules:
- You MAY change hypotheses and the conclusion if needed to match the semantics.
- You MAY adjust/add imports/opens/options as needed to keep the file compiling.
- Do NOT "solve" the task by weakening it to `True` or a tautology.
- Do NOT include any comments.
- Output a complete Lean 4 file starting with:
  import Mathlib
  import Aesop
- Include exactly one theorem, ending with `:= by sorry`.
"""

                                                                             
    ref_hdr_block = ""

    prev_compile = (previous_compile_error or "").strip()
    compile_feedback_block = ""
    if prev_compile:
        compile_feedback_block = f"\n\nCompiler feedback from the previous attempt:\n```\n{prev_compile[:800]}\n```\n"

    user_msg = f"""Natural language statement:
{informal.strip()}
{ref_hdr_block}
Current Lean code:
<CURRENT_CODE>
{(original_code or '').strip()}
</CURRENT_CODE>

Critic feedback (Accuracy Confirmation):
{accuracy_confirmation.strip() if accuracy_confirmation else "(missing)"}
{compile_feedback_block}
Goal:
- Modify the Lean theorem so that CriticLean would judge it as an exact formalization of the Natural language statement.
- Address every mismatch mentioned in the Critic feedback.

Output requirements:
1) Output EXACTLY ONE ```lean code block (no extra text).
2) The code MUST start with:
   import Mathlib
   import Aesop
3) You may add additional imports/opens/options after that if needed.
4) Do NOT include any comments.
5) Include EXACTLY ONE theorem, and end it with `:= by sorry`.

IMPORTANT:
- The <CURRENT_CODE> above is INPUT; do NOT repeat it verbatim.

Hint (often relevant when the informal claim says “f is constant on Ω”):
- Make the *constancy* explicit in the statement, e.g.:
  * `∃ c, ∀ z ∈ Ω, f z = c`, or
  * `∀ z w, z ∈ Ω → w ∈ Ω → f z = f w`.
  These are often easier for CriticLean to verify than a statement that only mentions a single pair of points.
"""
    return sys_msg, user_msg


def _extract_accuracy_confirmation_from_critic_raw(reasons: str) -> str:
    """Extract the '5. Accuracy Confirmation' section from CriticLean reasons."""
    s = (reasons or "").strip()
    if not s:
        return ""
    m = re.search(r"(?is)\\b5\\s*\\.?\\s*Accuracy\\s*Confirmation\\s*:?\\s*(.*)\\Z", s)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(?is)\\b5\\s*\\.?\\s*Accuracy\\s*Confirmation\\b", s)
    if not m2:
        return ""
    return s[m2.end() :].lstrip(" :\\n\\t").strip()


                                                                               
                         
                                                                               

class AutoformalizationRunner(EvolutionRunner):
    """
    Extended EvolutionRunner with a repair_queue.

    ================================================================================
                        This is the "brain" of the evolution loop
    ================================================================================

    Inheritance
    AutoformalizationRunner → EvolutionRunner → (ShinkaEvolve core)

    Key overridden methods
    - _process_completed_job(): process completed evaluations and add repair logic

    Added features
    1. repair_queue: manage candidates that need repair
    2. failure_buffer: record all failed candidates
    3. termination logic: failure categorization and termination strategy

    Spec constraints
    - [A] Novelty only for compile_ok=1: implemented in _process_completed_job
    - [B] Archive only compile_ok=1: implemented in _process_completed_job
    - [C] Repair counts toward budget: implemented in _process_repair
    - [E] Semantic repair (optional): second-stage repair for compile_ok=1 & semantic_ok=0
    """

    def __init__(
        self,
        evo_config: EvolutionConfig,
        job_config: JobConfig,
        db_config: DatabaseConfig,
        repair_config: Optional[RepairConfig] = None,
        termination_config: Optional[TerminationConfig] = None,
        problem_config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ):
        """
        Initialize AutoformalizationRunner.

        Args:
            evo_config: evolution config (from ShinkaEvolve)
            job_config: job config
            db_config: database config
            repair_config: repair config (optional; defaults to RepairConfig())
            termination_config: termination config (optional)
            problem_config: problem config (informal, header, etc.)
            verbose: enable verbose logging
        """
                                  
        super().__init__(evo_config, job_config, db_config, verbose)

                        
        self.repair_config = repair_config or RepairConfig()

                             
        self.termination_config = termination_config or TerminationConfig()

                                                  
        self.problem_config = problem_config or self._load_problem_config()

                                                                                 
         
                                                                                               
                                                                                                         
                                                                                                
                                                                                        
                                                                                      
        self.enable_evolast_fallback: bool = _env_truthy("AUTOFORMAL_ENABLE_EVOLAST_FALLBACK", default=True)
        self.seedbank_debit_calls: bool = _env_truthy("AUTOFORMAL_SEEDBANK_DEBIT_CALLS", default=False)
        self.seedbank_calls_per_seed: int = int(_env_int("AUTOFORMAL_SEEDBANK_CALLS_PER_SEED") or 0)
        self.seedbank_debit_seed0: bool = _env_truthy("AUTOFORMAL_SEEDBANK_DEBIT_SEED0", default=False)
        self.seedbank_debited_calls: int = 0

                                                                     
        self.llm_mode_requested: str = ""
        self.llm_mode_effective: str = ""
        self.llm_unavailable: bool = False
        self.llm_unavailable_reason: str = ""
        self.baseline_mode: str = str(
            self.problem_config.get("baseline_mode")
            or os.environ.get("AUTOFORMAL_BASELINE_MODE", "ours")
            or "ours"
        ).strip()
                                                                            
        if _env_truthy("AUTOFORMAL_DISABLE_SEMANTIC", default=False):
            try:
                self.problem_config["use_semantic"] = False
            except Exception:
                pass
                          
                                                                        
                                                                                
        env_val = os.environ.get("AUTOFORMAL_ENABLE_SEMANTIC_REPAIR")
        cfg_val = self.problem_config.get("enable_semantic_repair", None)
        if env_val is not None and str(env_val).strip() != "":
            semantic_repair_enabled = _env_truthy(
                "AUTOFORMAL_ENABLE_SEMANTIC_REPAIR", default=False
            )
        elif cfg_val is not None and str(cfg_val).strip() != "":
            semantic_repair_enabled = str(cfg_val).strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
        else:
            semantic_repair_enabled = False
        self.semantic_repair_enabled = bool(semantic_repair_enabled)
        self.semantic_repair_max_attempts: Optional[int] = _env_int("AUTOFORMAL_SEMANTIC_REPAIR_MAX_ATTEMPTS")
        self.semantic_repair_max_attempts_gen0: Optional[int] = _env_int(
            "AUTOFORMAL_SEMANTIC_REPAIR_MAX_ATTEMPTS_GEN0"
        )
        self.semantic_repair_temperature: float = _env_float("AUTOFORMAL_SEMANTIC_REPAIR_TEMPERATURE", 0.7)

                                                                                                   
                                                                            
        self.semantic_repair_start_calls: Optional[int] = _env_int("AUTOFORMAL_SEMANTIC_REPAIR_START_CALLS")
        if self.semantic_repair_start_calls is not None and int(self.semantic_repair_start_calls) <= 0:
            self.semantic_repair_start_calls = None

                                                                                        
                                                                     
        self.repair_skip_error_types: set[str] = {
            t.strip().lower() for t in _env_csv_list("AUTOFORMAL_REPAIR_SKIP_ERROR_TYPES")
        }

        self._configure_llm_clients()
        self._write_run_metadata()
        self._write_run_config()

                                                       
        self.repair_queue: List[RepairQueueItem] = []

                                                             
        failure_buffer_path = f"{self.results_dir}/failure_buffer.json"
        self.failure_buffer = FailureBuffer(failure_buffer_path)

                                         
        self.total_repair_llm_calls = 0
        self.total_repair_evals = 0
        self.total_repair_cost = 0.0
        self.total_semantic_repair_llm_calls = 0
        self.total_semantic_repair_evals = 0
        self.total_semantic_repair_cost = 0.0
        self.total_semantic_repair_successes = 0

                               
        self.start_time = time.time()
        self.soft_resets_count = 0
        self.best_fitness_tuple: Optional[tuple] = None
        self.generations_without_improvement = 0
        self.termination_reason: Optional[str] = None
        self._last_meta_update_generation: int = -1

                                                                                                 
                                                                                                 
                          
        self._file_seed0_pruned: bool = False

                        
        logger.info("=" * 60)
        logger.info("AutoformalizationRunner initialized (spec v1.2)")
        logger.info(f"  Repair enabled: {self.repair_config.enabled}")
        logger.info(f"  Max repair attempts: {self.repair_config.max_repair_attempts}")
        logger.info(f"  Repair temperature: {self.repair_config.repair_temperature}")
        logger.info(f"  Max LLM calls: {self.termination_config.max_llm_calls}")
        logger.info(f"  Stagnation gens: {self.termination_config.stagnation_generations}")
        logger.info(f"  LLM mode: requested={self.llm_mode_requested}, effective={self.llm_mode_effective}")
        if self.llm_unavailable:
            logger.info(f"  LLM unavailable: {self.llm_unavailable_reason}")
        logger.info("=" * 60)

    def _configure_llm_clients(self) -> None:
        """Configure generator/repair LLM client.

        Requirements:
        - No-LLM smoke tests must run without network dependencies.
        - When `--llm_mode=auto`, probe base_url and fallback to MockLLM if unreachable,
          recording `llm_unavailable=1` + reason for audit.
        """
        from offline_llm import MockLLMClient, ReplayLLMClient, load_mock_statements, probe_openai_base_url

        def truthy(v: Any) -> bool:
            return str(v or "").strip().lower() in {"1", "true", "yes", "y", "on"}

        requested = str(
            self.problem_config.get("llm_mode")
            or os.environ.get("AUTOFORMAL_LLM_MODE", "auto")
            or "auto"
        ).strip().lower()

        no_llm = truthy(self.problem_config.get("no_llm")) or truthy(os.environ.get("AUTOFORMAL_NO_LLM"))
        replay_path = (
            self.problem_config.get("replay_path")
            or os.environ.get("AUTOFORMAL_REPLAY_PATH")
            or ""
        )
        mock_path = (
            self.problem_config.get("mock_statements_path")
            or os.environ.get("AUTOFORMAL_MOCK_STATEMENTS_PATH")
            or ""
        )
        mock_statements = load_mock_statements(mock_path)

        self.llm_mode_requested = requested
        self.llm_unavailable = False
        self.llm_unavailable_reason = ""

                                                                  
        if no_llm:
                                                                                 
            original_model_name = self.llm.model_names[0] if hasattr(self.llm, 'model_names') and self.llm.model_names else "mock"
            self.llm = MockLLMClient(statements=mock_statements, model_name=original_model_name)
            self.llm_mode_effective = "mock"
            self.llm_unavailable = True
            self.llm_unavailable_reason = "no_llm=1"
        elif requested == "mock":
                                                                                 
            original_model_name = self.llm.model_names[0] if hasattr(self.llm, 'model_names') and self.llm.model_names else "mock"
            self.llm = MockLLMClient(statements=mock_statements, model_name=original_model_name)
            self.llm_mode_effective = "mock"
            self.llm_unavailable = True
            self.llm_unavailable_reason = "llm_mode=mock"
        elif requested == "replay":
            try:
                self.llm = ReplayLLMClient(jsonl_path=replay_path)
                self.llm_mode_effective = "replay"
                self.llm_unavailable = True
                self.llm_unavailable_reason = f"llm_mode=replay:{replay_path}"
            except Exception as e:
                                                                              
                original_model_name = self.llm.model_names[0] if hasattr(self.llm, 'model_names') and self.llm.model_names else "mock"
                self.llm = MockLLMClient(statements=mock_statements, model_name=original_model_name)
                self.llm_mode_effective = "mock"
                self.llm_unavailable = True
                self.llm_unavailable_reason = f"replay_failed:{type(e).__name__}:{e}"
        elif requested == "auto":
            base_url = (
                os.environ.get("OPENAI_LLM_BASE_URL")
                or self.problem_config.get("openai_llm_base_url")
                or ""
            )
            avail = probe_openai_base_url(base_url)
            if not avail.ok:
                original_model_name = self.llm.model_names[0] if hasattr(self.llm, 'model_names') and self.llm.model_names else "mock"
                self.llm = MockLLMClient(statements=mock_statements, model_name=original_model_name)
                self.llm_mode_effective = "mock"
                self.llm_unavailable = True
                self.llm_unavailable_reason = f"auto_fallback:{avail.reason}"
            else:
                self.llm_mode_effective = "real"
        else:
                                                                                            
            self.llm_mode_effective = "real"

                                                                                              
        if self.llm_mode_effective in {"mock", "replay"}:
            self.meta_llm = None
            self.novelty_llm = None
            try:
                self.meta_summarizer.meta_llm_client = None
            except Exception:
                pass
            try:
                self.novelty_judge.novelty_llm_client = None
            except Exception:
                pass

                                                                                          
                                                                                      
         
              
                                                                                                              
                                                                                           
        self.patch_llm = self.llm
        self.patch_llm_enabled = False
        if self.llm_mode_effective == "real":
            patch_llm = self._patch_llm_override()
            if patch_llm.get("enabled"):
                try:
                    from shinka.llm import LLMClient

                    llm_kwargs = dict(getattr(self.evo_config, "llm_kwargs", {}) or {})
                    self.patch_llm = LLMClient(
                        model_names=list(patch_llm.get("model_names", []) or []),
                        temperatures=llm_kwargs.get("temperatures", 0.75),
                        max_tokens=llm_kwargs.get("max_tokens", 4096),
                        reasoning_efforts=llm_kwargs.get("reasoning_efforts", "auto"),
                        model_sample_probs=llm_kwargs.get("model_sample_probs", None),
                        base_url=patch_llm.get("base_url") or None,
                        verbose=bool(getattr(self, "verbose", False)),
                    )
                    self.patch_llm_enabled = True
                except Exception as e:
                    logger.warning(
                        "[LLM] Failed to enable patch override; falling back to main LLM. "
                        f"error={type(e).__name__}:{e}"
                    )
                    self.patch_llm = self.llm
                    self.patch_llm_enabled = False

                                                                                      
         
                                                                                                            
                                                                 
        try:
            self._sync_generation_llm_budget_caps()
        except Exception:
            pass

    def _repair_llm_override(self) -> Dict[str, Any]:
        """Repair-only LLM override (A: main generation remains unchanged).

        Env:
        - AUTOFORMAL_REPAIR_OPENAI_LLM_BASE_URL: OpenAI-compatible base URL (e.g. Kimina: http://127.0.0.1:8009/v1)
        - AUTOFORMAL_REPAIR_LLM_MODELS: comma-separated model names (first is used)
        """
        base_url = (os.environ.get("AUTOFORMAL_REPAIR_OPENAI_LLM_BASE_URL") or "").strip()
        model_names = _env_csv_list("AUTOFORMAL_REPAIR_LLM_MODELS")
        model_name = model_names[0] if model_names else ""
        return {
            "enabled": bool(base_url and model_name),
            "base_url": base_url,
            "model_names": model_names,
            "model_name": model_name,
        }

    def _patch_llm_override(self) -> Dict[str, Any]:
        """Patch-only LLM override (edit proposals).

        This lets you use a specialized patch model (e.g. Qwen3-Patch) without
        changing Gen0 sampling (which can be served by a different model, or a seed bank).

        Env:
        - AUTOFORMAL_PATCH_OPENAI_LLM_BASE_URL (optional): OpenAI-compatible base URL.
          If empty, uses OPENAI_LLM_BASE_URL/OpenAI SDK default.
        - AUTOFORMAL_PATCH_LLM_MODELS: comma-separated model names (first is used)
        """
        base_url = (os.environ.get("AUTOFORMAL_PATCH_OPENAI_LLM_BASE_URL") or "").strip()
        model_names = _env_csv_list("AUTOFORMAL_PATCH_LLM_MODELS")
        model_name = model_names[0] if model_names else ""
        return {
            "enabled": bool(model_name),
            "base_url": base_url,
            "model_names": model_names,
            "model_name": model_name,
        }

    def _write_run_metadata(self) -> None:
        """Write run-level metadata for auditability (no LLM required)."""
        try:
            repair_llm = self._repair_llm_override()
            patch_llm = self._patch_llm_override()
            data = {
                "spec_version": "v1.2",
                "created_at_unix": time.time(),
                "results_dir": str(self.results_dir),
                "baseline_mode": self.baseline_mode,
                "seed": self.problem_config.get("seed", None),
                "llm_mode_requested": self.llm_mode_requested,
                "llm_mode_effective": self.llm_mode_effective,
                "llm_unavailable": int(bool(self.llm_unavailable)),
                "llm_unavailable_reason": self.llm_unavailable_reason,
                "openai_llm_base_url": os.environ.get("OPENAI_LLM_BASE_URL", ""),
                "llm_models": list(getattr(self.evo_config, "llm_models", []) or []),
                "patch_llm_enabled": int(bool(getattr(self, "patch_llm_enabled", False))),
                "patch_openai_llm_base_url": patch_llm.get("base_url", ""),
                "patch_llm_models": list(patch_llm.get("model_names", []) or []),
                "repair_openai_llm_base_url": repair_llm.get("base_url", ""),
                "repair_llm_models": list(repair_llm.get("model_names", []) or []),
                "no_llm": bool(
                    str(self.problem_config.get("no_llm", "")).strip().lower()
                    in {"1", "true", "yes", "y", "on"}
                ),
                "problem_config_preview": {
                    "informal_preview": (self.problem_config.get("informal", "") or "")[:200],
                    "use_semantic": bool(self.problem_config.get("use_semantic", False)),
                    "use_cycle_consistency": bool(self.problem_config.get("use_cycle_consistency", True)),
                    "cycle_api_base_url": self.problem_config.get("cycle_api_base_url"),
                    "cycle_model_name": self.problem_config.get("cycle_model_name"),
                    "critic_lean_url": os.environ.get("CRITIC_LEAN_URL", ""),
                    "critic_lean_model": os.environ.get("CRITIC_LEAN_MODEL", ""),
                    "lean_server_url": self.problem_config.get("lean_server_url"),
                },
                "evolution_config_preview": {
                    "num_generations": int(getattr(self.evo_config, "num_generations", 0) or 0),
                    "patch_types": list(getattr(self.evo_config, "patch_types", []) or []),
                    "patch_type_probs": list(getattr(self.evo_config, "patch_type_probs", []) or []),
                    "max_patch_attempts": int(getattr(self.evo_config, "max_patch_attempts", 0) or 0),
                    "max_patch_resamples": int(getattr(self.evo_config, "max_patch_resamples", 0) or 0),
                },
                "database_config_preview": {
                    "parent_selection_strategy": getattr(self.db_config, "parent_selection_strategy", None),
                    "cycle_softmax_temperature": getattr(self.db_config, "cycle_softmax_temperature", None),
                    "parent_usage_penalty_alpha": getattr(self.db_config, "parent_usage_penalty_alpha", None),
                    "num_islands": getattr(self.db_config, "num_islands", None),
                    "archive_size": getattr(self.db_config, "archive_size", None),
                    "num_archive_inspirations": getattr(self.db_config, "num_archive_inspirations", None),
                    "num_top_k_inspirations": getattr(self.db_config, "num_top_k_inspirations", None),
                },
                "repair_config_preview": {
                    "num_init_candidates_gen0": int(getattr(self.repair_config, "num_init_candidates_gen0", 0) or 0),
                    "max_repair_attempts": int(getattr(self.repair_config, "max_repair_attempts", 0) or 0),
                    "max_repair_attempts_gen0": int(getattr(self.repair_config, "max_repair_attempts_gen0", 0) or 0),
                    "repair_temperature": float(getattr(self.repair_config, "repair_temperature", 0.0) or 0.0),
                    "skip_error_types": sorted(list(self.repair_skip_error_types or [])),
                },
                "semantic_repair_config_preview": {
                    "enabled": bool(self.semantic_repair_enabled),
                    "start_calls": self.semantic_repair_start_calls,
                    "max_attempts": self.semantic_repair_max_attempts,
                    "max_attempts_gen0": self.semantic_repair_max_attempts_gen0,
                    "temperature": float(self.semantic_repair_temperature),
                },
                "termination_config_preview": {
                    "max_llm_calls": getattr(self.termination_config, "max_llm_calls", None),
                    "max_evals": getattr(self.termination_config, "max_evals", None),
                    "max_time_seconds": getattr(self.termination_config, "max_time_seconds", None),
                },
            }
            out = Path(self.results_dir) / "run_metadata.json"
            out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            return

    def _write_run_config(self) -> None:
        """Write the protocol-critical run configuration (implementation-aligned)."""
        try:
            def _get_env_int(name: str, default: int) -> int:
                try:
                    return int(os.environ.get(name, str(default)))
                except Exception:
                    return default

            def _get_env_float(name: str, default: float) -> float:
                try:
                    return float(os.environ.get(name, str(default)))
                except Exception:
                    return default

            db_cfg = self.db_config
            evo_cfg = self.evo_config
            rep_cfg = self.repair_config
            repair_llm = self._repair_llm_override()
            patch_llm = self._patch_llm_override()

            data = {
                "spec_version": "v1.2",
                "created_at_unix": time.time(),
                "results_dir": str(self.results_dir),
                "baseline_mode": self.baseline_mode,
                "seed": self.problem_config.get("seed", None),
                "protocol": {
                    "enforce_island_separation": bool(
                        getattr(db_cfg, "enforce_island_separation", True)
                    ),
                    "num_islands": int(getattr(db_cfg, "num_islands", 0) or 0),
                    "archive_size": int(getattr(db_cfg, "archive_size", 0) or 0),
                    "parent_selection_strategy": str(
                        getattr(db_cfg, "parent_selection_strategy", "")
                    ),
                    "cycle_softmax_temperature": float(
                        getattr(db_cfg, "cycle_softmax_temperature", 3.5) or 3.5
                    ),
                    "parent_usage_penalty_alpha": float(
                        getattr(db_cfg, "parent_usage_penalty_alpha", 0.05) or 0.05
                    ),
                    "elite_selection_ratio": float(
                        getattr(db_cfg, "elite_selection_ratio", 0.3) or 0.3
                    ),
                    "num_archive_inspirations": int(
                        getattr(db_cfg, "num_archive_inspirations", 0) or 0
                    ),
                    "num_top_k_inspirations": int(
                        getattr(db_cfg, "num_top_k_inspirations", 0) or 0
                    ),
                    "patch_types": list(getattr(evo_cfg, "patch_types", []) or []),
                    "patch_type_probs": list(
                        getattr(evo_cfg, "patch_type_probs", []) or []
                    ),
                    "cross_k": _get_env_int("AUTOFORMAL_CROSS_K", 1),
                    "cross_insp_temperature": _get_env_float(
                        "AUTOFORMAL_CROSS_INSP_TEMPERATURE",
                        _get_env_float("SOFTMAX_TEMPERATURE", 3.5),
                    ),
                    "cross_insp_penalty_alpha": _get_env_float(
                        "AUTOFORMAL_CROSS_INSP_PENALTY_ALPHA", 1.0
                    ),
                    "cross_insp_penalty_window": _get_env_int(
                        "AUTOFORMAL_CROSS_INSP_PENALTY_WINDOW", 50
                    ),
                    "novelty_filter": {
                        "enabled": bool(getattr(evo_cfg, "embedding_model", None)),
                        "embedding_model": getattr(evo_cfg, "embedding_model", None),
                        "openai_embed_base_url": os.environ.get("OPENAI_EMBED_BASE_URL", ""),
                        "novelty_llm_base_url": os.environ.get("OPENAI_NOVELTY_LLM_BASE_URL", ""),
                        "code_embed_sim_threshold": float(
                            getattr(evo_cfg, "code_embed_sim_threshold", 1.0) or 1.0
                        ),
                        "max_novelty_attempts": int(
                            getattr(evo_cfg, "max_novelty_attempts", 0) or 0
                        ),
                        "novelty_llm_models": list(
                            getattr(evo_cfg, "novelty_llm_models", []) or []
                        )
                        if getattr(evo_cfg, "novelty_llm_models", None) is not None
                        else None,
                    },
                    "repair_temperature": float(
                        getattr(rep_cfg, "repair_temperature", 0.0) or 0.0
                    ),
                    "K_gen0": int(
                        getattr(rep_cfg, "max_repair_attempts_gen0", 0) or 0
                    ),
                    "K_gen_ge_1": int(
                        getattr(rep_cfg, "max_repair_attempts", 0) or 0
                    ),
                    "semantic_repair": {
                        "enabled": bool(self.semantic_repair_enabled),
                        "max_attempts": self.semantic_repair_max_attempts,
                        "max_attempts_gen0": self.semantic_repair_max_attempts_gen0,
                        "temperature": float(self.semantic_repair_temperature),
                    },
                },
                "llm": {
                    "llm_mode_requested": self.llm_mode_requested,
                    "llm_mode_effective": self.llm_mode_effective,
                    "llm_unavailable": int(bool(self.llm_unavailable)),
                    "llm_unavailable_reason": self.llm_unavailable_reason,
                    "openai_llm_base_url": os.environ.get("OPENAI_LLM_BASE_URL", ""),
                    "llm_models": list(getattr(evo_cfg, "llm_models", []) or []),
                    "patch_llm_enabled": int(bool(getattr(self, "patch_llm_enabled", False))),
                    "patch_openai_llm_base_url": patch_llm.get("base_url", ""),
                    "patch_llm_models": list(patch_llm.get("model_names", []) or []),
                    "repair_openai_llm_base_url": repair_llm.get("base_url", ""),
                    "repair_llm_models": list(repair_llm.get("model_names", []) or []),
                    "no_llm": bool(
                        str(self.problem_config.get("no_llm", "")).strip().lower()
                        in {"1", "true", "yes", "y", "on"}
                    ),
                },
                "problem_config_preview": {
                    "informal_preview": (self.problem_config.get("informal", "") or "")[
                        :200
                    ],
                    "use_beq": bool(self.problem_config.get("use_beq", False)),
                    "use_semantic": bool(self.problem_config.get("use_semantic", False)),
                    "use_cycle_consistency": bool(
                        self.problem_config.get("use_cycle_consistency", True)
                    ),
                    "cycle_api_base_url": self.problem_config.get("cycle_api_base_url"),
                    "cycle_model_name": self.problem_config.get("cycle_model_name"),
                    "lean_server_url": self.problem_config.get("lean_server_url"),
                },
            }
            out = Path(self.results_dir) / "run_config.json"
            out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            return
    def _load_problem_config(self) -> Dict[str, Any]:
        """
        Load problem configuration from file.

        Why load from a file?
        `problem_config.json` is created by `run_evo.py` at run start and includes:
        - informal: natural-language statement
        - header: Lean 4 header (imports/opens)
        - ground_truth: reference statement (for BEq+)
        - use_beq: whether to enable BEq+

        This design decouples `run_evo.py` and AutoformalizationRunner; changing config does not
        require editing code.
        """
        candidate_paths = [
            Path(str(self.results_dir)) / "problem_config.json",
            Path(__file__).parent / "problem_config.json",                   
        ]
        for config_path in candidate_paths:
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                if self.verbose:
                    logger.info(f"[Config] Loaded problem config from: {config_path}")
                return cfg
                                                                    
        return {
            "informal": "",
            "header": "import Mathlib",
            "ground_truth": "",
            "use_beq": False,
        }

    def generate_initial_program(self):
        """
        Generate initial program with LLM, with Kimina adapter support.

        Why override?
        Kimina-Autoformalizer models respond better to simple prompts:
        - Generic prompts can be too verbose, causing the model to echo examples or output in the wrong format.
        - Using KiminaAdapter to build a simpler prompt tends to work better.

        Returns:
            (initial_code, patch_name, patch_description, api_costs) tuple
        """
        import re

        llm_kwargs = self.llm.get_kwargs()
                                                                                                          
        llm_kwargs["temperature"] = 0.5
        model_name = self.evo_config.llm_models[0] if self.evo_config.llm_models else ""
        adapter = get_model_adapter(model_name)
        total_costs = 0.0

                           
        informal = self.problem_config.get("informal", "")
        header = self.problem_config.get("header", "import Mathlib")

                                               
        if adapter and is_kimina_model(model_name):
                                                 
            sys_msg, user_msg = adapter.build_prompt(informal, header)
            logger.info(f"[Gen0] Using Kimina adapter for model: {model_name}")
        else:
                                                    
            sys_msg, user_msg = self.prompt_sampler.initial_program_prompt()

        msg_history = []

        for attempt in range(self.evo_config.max_patch_attempts):
            budget_reason = self._check_budget_exhausted()
            if budget_reason is not None:
                self.termination_reason = budget_reason
                raise RuntimeError(f"Budget exhausted during Gen0 generation: {budget_reason}")

            response = self.llm.query(
                msg=user_msg,
                system_msg=sys_msg,
                llm_kwargs=llm_kwargs,
                msg_history=msg_history,
            )
            if response is None or response.content is None:
                budget_reason = self._check_budget_exhausted()
                if budget_reason is not None:
                    self.termination_reason = budget_reason
                    raise RuntimeError(f"Budget exhausted during Gen0 generation: {budget_reason}")
                if self.verbose:
                    logger.info(
                        f"  INITIAL PROGRAM ATTEMPT {attempt + 1}/"
                        f"{self.evo_config.max_patch_attempts} "
                        "FAILURE. Error: LLM response content was None."
                    )
                if attempt < self.evo_config.max_patch_attempts - 1:
                    user_msg = (
                        "The previous response was empty. Please try again "
                        "and provide the full code."
                    )
                    if response and response.new_msg_history:
                        msg_history = response.new_msg_history
                    continue
                else:
                    break

            total_costs += response.cost or 0

                                                         
            initial_code = None
            if adapter:
                initial_code = adapter.parse_output(response.content)

                                                           
            if not initial_code:
                initial_code = extract_between(
                    response.content,
                    f"```{self.evo_config.language}",
                    "```",
                    False,
                )

                                                                             
            if not initial_code and self.evo_config.language == "lean":
                                                                                         
                cand = normalize_lean_code(response.content or "")
                if cand and normalize_lean_statement(cand):
                    initial_code = cand

            if initial_code:
                                                                                            
                if self.evo_config.language == "lean" and is_trivial_tautology_placeholder(initial_code):
                    if self.verbose:
                        logger.info(
                            f"  INITIAL PROGRAM ATTEMPT {attempt + 1}/"
                            f"{self.evo_config.max_patch_attempts} REJECTED. "
                            "Reason: trivial/tautological placeholder."
                        )
                    if attempt < self.evo_config.max_patch_attempts - 1:
                        user_msg = (
                            "The previous response produced a trivial placeholder theorem. "
                            "This is NOT allowed. Provide a non-trivial Lean 4 theorem statement that "
                            "formalizes the claim."
                        )
                        if response and response.new_msg_history:
                            msg_history = response.new_msg_history
                        continue
                    break

                patch_name = extract_between(
                    response.content, "<NAME>", "</NAME>", False
                ) or "initial_program"
                patch_description = extract_between(
                    response.content, "<DESCRIPTION>", "</DESCRIPTION>", False
                ) or "Initial program generated by LLM."

                comment_char = "--" if self.evo_config.language == "lean" else "#"

                initial_code = (
                    f"{comment_char} EVOLVE-BLOCK-START\n"
                    f"{initial_code}\n"
                    f"{comment_char} EVOLVE-BLOCK-END\n"
                )
                if self.verbose:
                    logger.info(
                        f"  INITIAL PROGRAM ATTEMPT {attempt + 1}/"
                        f"{self.evo_config.max_patch_attempts} SUCCESS."
                    )
                return initial_code, patch_name, patch_description, total_costs

                                       
            if self.verbose:
                logger.info(
                    f"  INITIAL PROGRAM ATTEMPT {attempt + 1}/"
                    f"{self.evo_config.max_patch_attempts} "
                    "FAILURE. Error: Could not extract code from response."
                )
            if attempt < self.evo_config.max_patch_attempts - 1:
                user_msg = (
                    "The previous response did not contain valid code. "
                    "Please provide the Lean 4 theorem statement directly."
                )
                if response and response.new_msg_history:
                    msg_history = response.new_msg_history

        raise RuntimeError(
            "Failed to generate a valid initial program (non-empty, non-`True` placeholder)."
        )

    def _maybe_apply_staged_patch_schedule(self) -> None:
        """
        Dynamically override patch types/probabilities based on *budget calls used*.

        This enables experiment scripts to do:
        - Stage 1 (early): full/diff (better compile yield)
        - Stage 2 (late): cross (inject cross-problem few-shots / inspirations)

        Controlled by env vars:
        - AUTOFORMAL_STAGED_BUDGET=1
        - AUTOFORMAL_STAGE1_CALLS=<int>
        - AUTOFORMAL_STAGE1_PATCH_TYPES="full,diff"
        - AUTOFORMAL_STAGE1_PATCH_TYPE_PROBS="0.5,0.5"
        - AUTOFORMAL_STAGE2_PATCH_TYPES="cross"
        - AUTOFORMAL_STAGE2_PATCH_TYPE_PROBS="1.0"
        """
        if not _env_truthy("AUTOFORMAL_STAGED_BUDGET", default=False):
            return

        stage1_calls = int(_env_int("AUTOFORMAL_STAGE1_CALLS") or 0)
        if stage1_calls <= 0:
            return

        stage1_types_raw = str(os.environ.get("AUTOFORMAL_STAGE1_PATCH_TYPES", "") or "").strip()
        stage1_probs_raw = str(os.environ.get("AUTOFORMAL_STAGE1_PATCH_TYPE_PROBS", "") or "").strip()
        stage2_types_raw = str(os.environ.get("AUTOFORMAL_STAGE2_PATCH_TYPES", "") or "").strip()
        stage2_probs_raw = str(os.environ.get("AUTOFORMAL_STAGE2_PATCH_TYPE_PROBS", "") or "").strip()

        raw_llm_api_calls = int(getattr(self.llm, "total_calls", 0) or 0)
        budget_calls = raw_llm_api_calls + int(self.seedbank_debited_calls or 0)

        stage = 1 if budget_calls < stage1_calls else 2
        types_raw = stage1_types_raw if stage == 1 else stage2_types_raw
        probs_raw = stage1_probs_raw if stage == 1 else stage2_probs_raw

        patch_types = [t.strip() for t in types_raw.split(",") if t.strip()]
        if not patch_types:
            return

        probs: list[float] = []
        if probs_raw:
            try:
                probs = [float(x.strip()) for x in probs_raw.split(",") if x.strip()]
            except Exception:
                probs = []
        if probs and len(probs) != len(patch_types):
            logger.warning(
                "[StagedPatch] Ignoring probs due to length mismatch: "
                f"types={patch_types}, probs_raw={probs_raw!r}"
            )
            probs = []
        if not probs:
            probs = [1.0 / float(len(patch_types))] * len(patch_types)
        else:
            probs = [max(0.0, float(p)) for p in probs]
            s = float(sum(probs))
            probs = [p / s for p in probs] if s > 0 else [1.0 / float(len(patch_types))] * len(patch_types)

        prev_stage = getattr(self, "_staged_patch_stage", None)
        prev_types = list(getattr(self.prompt_sampler, "patch_types", []) or [])
        if prev_stage != stage or prev_types != patch_types:
            setattr(self, "_staged_patch_stage", stage)
            self.prompt_sampler.patch_types = patch_types
            self.prompt_sampler.patch_type_probs = probs
            logger.info(
                f"[StagedPatch] stage={stage} budget_calls={budget_calls} "
                f"patch_types={patch_types} patch_type_probs={probs}"
            )

    def run_patch(
        self,
        parent_program: Program,
        archive_programs: List[Program],
        top_k_programs: List[Program],
        generation: int,
        novelty_attempt: int = 1,
        resample_attempt: int = 1,
    ) -> tuple[Optional[str], dict, int]:
        """
        Override: hard-reject trivial placeholder `: True := ...` *before* evaluation.

        Why here?
        - If `: True := by sorry` enters DB/archive, it can be sampled as parent/inspiration
          and quickly collapses the search to trivial descendants.
        - Rejecting at patch time prevents wasting evals and prevents on-disk `main.lean`
          from ending up as placeholder for a generation.
        """
        from shinka.edit import apply_diff_patch, apply_full_patch, summarize_diff

        max_patch_attempts = self.evo_config.max_patch_attempts
        if self.verbose:
            logger.info(
                f"Edit Cycle {generation} -> {generation + 1}, "
                f"Max Patch Attempts: {max_patch_attempts}"
            )

                                                                               
        exec_path = Path(self.results_dir) / f"{FOLDER_PREFIX}_{generation}" / f"main.{self.lang_ext}"
        try:
            exec_path.parent.mkdir(parents=True, exist_ok=True)
            exec_path.write_text(parent_program.code or "", encoding="utf-8")
        except Exception:
            pass

                                          
        meta_recs, _, _ = self.meta_summarizer.get_current()

                                                         
        self._maybe_apply_staged_patch_schedule()

        patch_sys, patch_msg, patch_type = self.prompt_sampler.sample(
            parent=parent_program,
            archive_inspirations=archive_programs,
            top_k_inspirations=top_k_programs,
            meta_recommendations=meta_recs,
        )

        cross_inspiration_ids = getattr(self.prompt_sampler, "_last_cross_inspiration_ids", [])

        if patch_type in ["full", "cross"] or (
            patch_type == "diff" and self.evo_config.language == "lean"
        ):
            apply_patch = apply_full_patch
        elif patch_type == "diff":
            apply_patch = apply_diff_patch
        elif patch_type == "paper":
            raise NotImplementedError("Paper edit not implemented.")
        else:
            raise ValueError(f"Invalid patch type: {patch_type}")

        total_costs = 0.0
        msg_history = []
        patch_client = getattr(self, "patch_llm", None) or self.llm
        llm_kwargs = patch_client.get_kwargs()
        if self.llm_selection is not None and patch_client is self.llm:
            model_name = llm_kwargs["model_name"]
            self.llm_selection.update_submitted(model_name)

        code_diff: Optional[str] = None
        num_applied_attempt = 0
        error_attempt: Optional[str] = "Max attempts reached without successful patch."
        patch_name: Optional[str] = None
        patch_description: Optional[str] = None
        output_path_attempt = None
        patch_txt_attempt = None
        patch_path = None
        diff_summary: dict = {}
        diff_local_stats = None
        placeholder_rejections = 0
        exact_duplicate_detected = False
        exact_duplicate_target: str = ""
        evolast_fallback_used = False
        evolast_fallback_reason: str = ""
        evolast_fallback_mode: str = ""
        evolast_fallback_info: dict = {}
        response = None

        for patch_attempt in range(max_patch_attempts):
            budget_reason = self._check_budget_exhausted()
            if budget_reason is not None:
                self.termination_reason = budget_reason
                error_attempt = budget_reason
                break

            response = patch_client.query(
                msg=patch_msg,
                system_msg=patch_sys,
                msg_history=msg_history,
                llm_kwargs=llm_kwargs,
            )
            if response is None or response.content is None:
                budget_reason = self._check_budget_exhausted()
                if budget_reason is not None:
                    self.termination_reason = budget_reason
                    error_attempt = budget_reason
                    break
                if self.verbose:
                    logger.info(
                        f"  PATCH ATTEMPT {patch_attempt + 1}/{max_patch_attempts} FAILURE. "
                        "Error: LLM response content was None."
                    )
                error_attempt = "LLM response content was None."
                num_applied_attempt = 0
                patch_txt_attempt = None
                if patch_attempt < max_patch_attempts - 1:
                    patch_msg = (
                        "The previous attempt to get an edit was not successful because the "
                        "LLM response was empty.\n\n"
                        "Try again and follow the EXACT output protocol:\n"
                        "- Output a COMPLETE Lean 4 file inside a ```lean code fence (full file, not a diff).\n"
                        "- The file MUST start with these two lines (in this order):\n"
                        "  import Mathlib\n"
                        "  import Aesop\n"
                        "- Do NOT include any comments.\n"
                        "- Include EXACTLY ONE `theorem` declaration.\n"
                        "- The theorem MUST end with `:= by sorry`.\n"
                    )
                    if response:
                        msg_history = response.new_msg_history
                    continue
                break

            total_costs += float(response.cost or 0.0)
            patch_name = extract_between(response.content, "<NAME>", "</NAME>", False)
            patch_description = extract_between(
                response.content, "<DESCRIPTION>", "</DESCRIPTION>", False
            )

            (
                updated_code_attempt,
                num_applied_attempt,
                output_path_attempt,
                error_attempt,
                patch_txt_attempt,
                patch_path,
            ) = apply_patch(
                original_str=parent_program.code,
                patch_str=response.content,
                patch_dir=f"{self.results_dir}/{FOLDER_PREFIX}_{generation}",
                language=self.evo_config.language,
                verbose=False,
            )

                                                                                
            if error_attempt is None and num_applied_attempt > 0 and updated_code_attempt is not None:
                                                                                                      
                if self.evo_config.language == "lean":
                    rebased, did_rebase, _reason = rebase_lean_candidate_on_parent_preamble_if_missing(
                        candidate_code=updated_code_attempt,
                        parent_code=parent_program.code or "",
                    )
                    if did_rebase and rebased.strip() and rebased.strip() != (updated_code_attempt or "").strip():
                        updated_code_attempt = rebased
                        try:
                            exec_path.write_text(rebased, encoding="utf-8")
                        except Exception:
                            pass
                                                                                            
                                                                                           
                                                                                           
                        if patch_path:
                            patch_txt_attempt = overwrite_edit_diff_for_final_candidate(
                                patch_path=Path(patch_path),
                                original_code=parent_program.code or "",
                                final_code=rebased,
                                lang_ext=self.lang_ext,
                            ) or patch_txt_attempt

                if self.evo_config.language == "lean" and is_trivial_tautology_placeholder(updated_code_attempt):
                    placeholder_rejections += 1
                    error_attempt = "Rejected trivial placeholder: tautology-like statement."
                    if self.verbose:
                        logger.info(
                            f"  PATCH ATTEMPT {patch_attempt + 1}/{max_patch_attempts} REJECTED. "
                            "Reason: trivial/tautological statement."
                        )
                                                                                                
                    try:
                        exec_path.write_text(parent_program.code or "", encoding="utf-8")
                    except Exception:
                        pass
                                                                            
                    patch_msg = (
                        "The previous output is INVALID because it produced a trivial placeholder theorem. "
                        "This is forbidden.\n\n"
                        "Try again and follow the EXACT output protocol:\n"
                        "- Output a COMPLETE Lean 4 file inside a ```lean code fence (full file, not a diff).\n"
                        "- The file MUST start with these two lines (in this order):\n"
                        "  import Mathlib\n"
                        "  import Aesop\n"
                        "- Do NOT include any comments.\n"
                        "- Include EXACTLY ONE `theorem` declaration.\n"
                        "- The theorem MUST be non-trivial (NOT `: True := by sorry`).\n"
                        "- The theorem MUST end with `:= by sorry`.\n"
                    )
                    msg_history = response.new_msg_history
                    code_diff = None
                    num_applied_attempt = 0
                    continue

                                                                               
                if self.evo_config.language == "lean" and not _is_theorem_statement(updated_code_attempt):
                    error_attempt = "Rejected output protocol: statement must start with `theorem`."
                    if self.verbose:
                        logger.info(
                            f"  PATCH ATTEMPT {patch_attempt + 1}/{max_patch_attempts} REJECTED. "
                            "Reason: statement did not start with `theorem`."
                        )
                    try:
                        exec_path.write_text(parent_program.code or "", encoding="utf-8")
                    except Exception:
                        pass
                    patch_msg = (
                        "The previous output is INVALID because it did not contain a `theorem` declaration.\n\n"
                        "Try again and follow the EXACT output protocol:\n"
                        "- Output a COMPLETE Lean 4 file inside a ```lean code fence (full file, not a diff).\n"
                        "- The file MUST start with these two lines (in this order):\n"
                        "  import Mathlib\n"
                        "  import Aesop\n"
                        "- Do NOT include any comments.\n"
                        "- Include EXACTLY ONE `theorem` declaration (do NOT use lemma/def/example).\n"
                        "- The theorem MUST end with `:= by sorry`.\n"
                    )
                    msg_history = response.new_msg_history
                    code_diff = None
                    num_applied_attempt = 0
                    continue

                                                                                   
                 
                                       
                                                                                                
                                                                                           
                                                                                   
                if self.evo_config.language == "lean":
                    cand_key = _canonicalize_lean_for_exact_match(updated_code_attempt)
                    parent_key = _canonicalize_lean_for_exact_match(parent_program.code or "")
                    if cand_key and parent_key and cand_key == parent_key:
                        exact_duplicate_detected = True
                        exact_duplicate_target = "parent"
                    elif patch_type == "cross" and cand_key:
                        insp_by_id = {}
                        for p in list(archive_programs or []) + list(top_k_programs or []):
                            pid = getattr(p, "id", None)
                            if pid:
                                insp_by_id[pid] = p
                        for pid in cross_inspiration_ids or []:
                            prog = insp_by_id.get(pid)
                            if not prog:
                                continue
                            insp_key = _canonicalize_lean_for_exact_match(getattr(prog, "code", "") or "")
                            if insp_key and insp_key == cand_key:
                                exact_duplicate_detected = True
                                exact_duplicate_target = f"cross_inspiration:{pid}"
                                break

                    if exact_duplicate_detected and self.verbose:
                        logger.info(
                            f"  PATCH ATTEMPT {patch_attempt + 1}/{max_patch_attempts} NOTE. "
                            f"Exact duplicate detected ({exact_duplicate_target})."
                        )

                                                                                                           
                    if exact_duplicate_detected and self.enable_evolast_fallback:
                        try:
                            from evolast_lean import apply_evolast_to_lean_code, parse_rule_weights

                            evolast_fallback_mode = (
                                os.environ.get("AUTOFORMAL_EVOLAST_MODE", "").strip() or "aggressive"
                            )
                            evolast_p = 0.35
                            evolast_max_rewrites = 32
                            try:
                                raw_p = str(os.environ.get("AUTOFORMAL_EVOLAST_P", "") or "").strip()
                                if raw_p:
                                    evolast_p = float(raw_p)
                            except Exception:
                                evolast_p = 0.35
                            try:
                                raw_m = str(os.environ.get("AUTOFORMAL_EVOLAST_MAX_REWRITES", "") or "").strip()
                                if raw_m:
                                    evolast_max_rewrites = int(raw_m)
                            except Exception:
                                evolast_max_rewrites = 32
                            evolast_weights = parse_rule_weights(os.environ.get("AUTOFORMAL_EVOLAST_RULE_WEIGHTS"))
                            fallback_code, info = apply_evolast_to_lean_code(
                                parent_program.code or "",
                                mode=evolast_fallback_mode,
                                p=float(evolast_p),
                                max_rewrites=int(evolast_max_rewrites),
                                rule_weights=dict(evolast_weights),
                            )
                            fallback_code = str(fallback_code or "").strip()
                            if fallback_code and fallback_code.strip() != (parent_program.code or "").strip():
                                updated_code_attempt = fallback_code
                                try:
                                    exec_path.write_text(fallback_code, encoding="utf-8")
                                except Exception:
                                    pass
                                                                                               
                                if patch_path:
                                    patch_txt_attempt = overwrite_edit_diff_for_final_candidate(
                                        patch_path=Path(patch_path),
                                        original_code=parent_program.code or "",
                                        final_code=fallback_code,
                                        lang_ext=self.lang_ext,
                                    ) or patch_txt_attempt
                                evolast_fallback_used = True
                                evolast_fallback_reason = f"llm_exact_duplicate_{exact_duplicate_target or 'parent'}"
                                evolast_fallback_info = info if isinstance(info, dict) else {"info": str(info)}
                                                                                                         
                                patch_type = "evolast"
                                patch_name = f"evolast_{evolast_fallback_mode}"
                                patch_description = "EvolAST fallback on parent after exact-duplicate LLM patch."
                        except Exception as e:
                            if self.verbose:
                                logger.info(
                                    f"[EvolAST] Fallback skipped (error): {type(e).__name__}: {e}"
                                )

                                                                         
                if patch_path:
                    diff_summary = summarize_diff(str(patch_path))
                if patch_type == "diff" and self.evo_config.language == "lean":
                    diff_local_stats = _lean_local_diff_stats(parent_program.code, updated_code_attempt)
                if self.verbose:
                    logger.info(
                        f"  PATCH ATTEMPT {patch_attempt + 1}/{max_patch_attempts} SUCCESS. "
                        f"Output: {output_path_attempt}, Patches Applied: {num_applied_attempt}."
                    )
                code_diff = patch_txt_attempt
                break

                                           
            error_str = str(error_attempt) if error_attempt else "No changes applied."
            patch_msg = (
                "The previous edit was not successful. This was the error message:\n\n"
                + error_str
                + "\n\n"
                "Try again and follow the EXACT output protocol:\n"
                "- Output a COMPLETE Lean 4 file inside a ```lean code fence (full file, not a diff).\n"
                "- The file MUST start with these two lines (in this order):\n"
                "  import Mathlib\n"
                "  import Aesop\n"
                "- Do NOT include any comments.\n"
                "- Include EXACTLY ONE `theorem` declaration.\n"
                "- The theorem MUST end with `:= by sorry`.\n"
            )
            if self.verbose:
                logger.info(
                    f"  PATCH ATTEMPT {patch_attempt + 1}/{max_patch_attempts} FAILURE. "
                    f"Error: '{error_str}', Patches Applied: {num_applied_attempt}."
                )
            msg_history = response.new_msg_history
            code_diff = None

                                                                     
        original_filename = f"original.{self.lang_ext}"
        if original_filename in diff_summary:
            diff_summary = diff_summary[original_filename]

        meta_edit_data = {
            "patch_type": patch_type,
            "api_costs": total_costs,
            "num_applied": num_applied_attempt,
            "patch_name": patch_name,
            "patch_description": patch_description,
            "error_attempt": error_attempt,
            "novelty_attempt": novelty_attempt,
            "resample_attempt": resample_attempt,
            "patch_attempt": patch_attempt + 1,
            "placeholder_true_rejections": placeholder_rejections,
            "exact_duplicate_detected": exact_duplicate_detected,
            "exact_duplicate_target": exact_duplicate_target,
            "evolast_fallback_used": int(bool(evolast_fallback_used)),
            "evolast_fallback_reason": evolast_fallback_reason,
            "evolast_fallback_mode": evolast_fallback_mode,
            "evolast_fallback_info": evolast_fallback_info,
            **llm_kwargs,
            "llm_result": response.to_dict() if response else None,
            "diff_summary": diff_summary,
            "diff_local_stats": diff_local_stats,
            "cross_inspiration_ids": cross_inspiration_ids if patch_type == "cross" else [],
        }
        if self.verbose and num_applied_attempt > 0:
            self._print_metadata_table(meta_edit_data, generation)
        return code_diff, meta_edit_data, num_applied_attempt

    def _run_generation_0(self):
        """
        Run generation 0 with compile repair support.

        Why override?
        - Base EvolutionRunner always inserts gen_0 into DB regardless of compile_ok.
        - For autoformalization we want: compile_ok=0 should NOT be treated as a valid parent.
        - If gen_0 fails to compile, we immediately attempt repair (≤ max_repair_attempts).
        """
        initial_dir = Path(f"{self.results_dir}/{FOLDER_PREFIX}_0")
        initial_dir.mkdir(parents=True, exist_ok=True)

        min_num_init = int(getattr(self.repair_config, "num_init_candidates_gen0", 3))
        logger.info(
            f"[Gen0] Bootstrapping with at least {min_num_init} initial candidates "
            "(continue sampling until a non-placeholder compile_ok=1 seed is found, "
            "or until budget is exhausted)"
        )

        inserted_programs: List[Program] = []
        best_score: Optional[float] = None

                                                                                 
                                                                                   
        problem_cfg = getattr(self, "problem_config", None)
        if not isinstance(problem_cfg, dict):
            problem_cfg = {}
        init_programs_dir_raw = str(problem_cfg.get("init_programs_dir") or "").strip()
        seed_bank_dir: Optional[Path] = None
        if init_programs_dir_raw:
            p = Path(init_programs_dir_raw).expanduser()
            if (p / "seed_0").exists():
                seed_bank_dir = p
            elif (p / "gen_0" / "seed_0").exists():
                seed_bank_dir = p / "gen_0"
            else:
                raise ValueError(
                    "Invalid init_programs_dir: expected a directory containing `seed_0/` "
                    "or a directory containing `gen_0/seed_0/`. "
                    f"Got: {init_programs_dir_raw}"
                )
            logger.info(f"[Gen0] Reusing seed bank from: {seed_bank_dir}")
        reuse_seedbank_eval = bool(seed_bank_dir is not None and _env_truthy("AUTOFORMAL_REUSE_INIT_EVAL", default=False))

        def _load_seedbank_metrics(seed_i: int) -> Optional[dict]:
            if seed_bank_dir is None:
                return None
            candidates = [
                seed_bank_dir / f"seed_{seed_i}" / "results" / "metrics.json",
                seed_bank_dir / f"seed_{seed_i}" / "metrics.json",
            ]
            for mp in candidates:
                if not mp.exists():
                    continue
                try:
                    return json.loads(mp.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning(f"[Gen0] Failed to load seedbank metrics: {mp} ({type(e).__name__}: {e})")
                    return None
            return None

        def _has_non_placeholder_seed(programs: List[Program]) -> bool:
            for p in programs:
                code = str(getattr(p, "code", "") or "")
                if not code:
                    continue
                if self.evo_config.language == "lean" and is_trivial_tautology_placeholder(code):
                    continue
                return True
            return False

        i = 0
        seed_bank_exhausted = False
        while True:
                                                                               
                                                                               
            if i >= min_num_init and _has_non_placeholder_seed(inserted_programs):
                break

            budget_reason = self._check_budget_exhausted()
            if budget_reason is not None:
                self.termination_reason = budget_reason
                logger.info(
                    f"[Gen0] Budget exhausted during bootstrapping at seed={i}: {budget_reason}"
                )
                break

            seed_dir = initial_dir / f"seed_{i}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            exec_fname = str(seed_dir / f"main.{self.lang_ext}")
            results_dir = str(seed_dir / "results")
            Path(results_dir).mkdir(parents=True, exist_ok=True)

            api_costs = 0.0
            patch_name = f"initial_program_{i}"
            patch_description = "Initial program."
            patch_type = "init"

                                                                                        
             
                                              
                                          
                                                                   
             
                        
                                                                                              
             
                                                                                                            
                                                                                                        
                                                                                                      
                                                                                         
            use_seed_bank = seed_bank_dir is not None and (not seed_bank_exhausted)
            use_init_file = (not use_seed_bank) and i == 0 and bool(self.evo_config.init_program_path)
            init_seed_source = "seed_bank" if use_seed_bank else "file" if use_init_file else "llm"

            if use_seed_bank:
                assert seed_bank_dir is not None
                src = seed_bank_dir / f"seed_{i}" / f"main.{self.lang_ext}"
                if not src.exists():
                    if i < min_num_init:
                        raise FileNotFoundError(
                            f"Seed bank is missing {src}. "
                            "Make sure `--num_init_candidates_gen0` matches the number of available seeds, "
                            "or provide a seed bank that contains `seed_i/main.lean` for every i."
                        )
                    seed_bank_exhausted = True
                                                                                                     
                                                                                                   
                                                                                                     
                    if _has_non_placeholder_seed(inserted_programs):
                        logger.warning(
                            f"[Gen0] Seed bank exhausted at seed={i} (missing: {src}); stopping bootstrapping."
                        )
                        if self.termination_reason is None:
                            self.termination_reason = f"gen0_seedbank_exhausted_at_seed_{i}"
                        break

                    logger.warning(
                        f"[Gen0] Seed bank exhausted at seed={i} (missing: {src}) "
                        "AND no compile_ok=1 seed found; falling back to LLM sampling."
                    )
                                                                         
                    continue
                shutil.copy(src, exec_fname)
                patch_name = f"seedbank_seed_{i}"
                patch_description = f"Initial program copied from seed bank: {src}"
                                                                              
                if self.seedbank_debit_calls and self.seedbank_calls_per_seed > 0:
                    if i != 0 or self.seedbank_debit_seed0:
                        self.seedbank_debited_calls += int(self.seedbank_calls_per_seed)
            if use_init_file:
                if self.verbose:
                    logger.info(
                        f"[Gen0] Seed 0 from file: {self.evo_config.init_program_path}"
                    )
                shutil.copy(self.evo_config.init_program_path, exec_fname)
                patch_description = "Initial program from file."

            if (not use_seed_bank) and (not use_init_file):
                if self.verbose:
                    logger.info(f"[Gen0] Generating seed {i} with LLM...")
                try:
                    initial_code, patch_name, patch_description, api_costs = (
                        self.generate_initial_program()
                    )
                    with open(exec_fname, "w", encoding="utf-8") as f:
                        f.write(initial_code)
                except Exception as e:
                    logger.warning(f"[Gen0] Failed to generate seed {i}: {e}")
                    self._add_to_failure_buffer(
                        program_id=str(uuid.uuid4()),
                        generation=0,
                        compile_error_type="gen0_generation_failed",
                        compile_error_msg=str(e),
                        statement="",
                        repair_attempts=0,
                        repair_llm_calls_used=0,
                        final_status="generation_failed",
                    )
                    i += 1
                    continue

            results = None
            rtime = 0.0
            if use_seed_bank and reuse_seedbank_eval:
                metrics_loaded = _load_seedbank_metrics(i)
                if metrics_loaded is not None:
                                                                             
                    try:
                        Path(results_dir, "metrics.json").write_text(
                            json.dumps(metrics_loaded, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        Path(results_dir, "correct.json").write_text(
                            json.dumps({"correct": bool(int(metrics_loaded.get("compile_ok", 0) or 0) == 1)}, indent=2),
                            encoding="utf-8",
                        )
                    except Exception as e:
                        logger.warning(f"[Gen0] Failed to write replayed metrics to results_dir: {e}")
                    results = {"metrics": metrics_loaded, "stdout_log": "", "stderr_log": ""}
                else:
                    logger.warning(
                        f"[Gen0] AUTOFORMAL_REUSE_INIT_EVAL=1 but metrics not found for seed={i}; falling back to evaluation."
                    )

            if results is None:
                                        
                results, rtime = self.scheduler.run(exec_fname, results_dir)

            metrics_val = results.get("metrics", {}) if results else {}
            compile_ok = metrics_val.get("compile_ok", 0)
            compile_error_type = metrics_val.get("compile_error_type", "")
            compile_error_msg = metrics_val.get("compile_error_msg", "")
            statement = metrics_val.get("statement", "")

                                                                    
            try:
                evaluated_code = Path(exec_fname).read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Could not read code for job {exec_fname}. Error: {e}")
                evaluated_code = ""

            if compile_ok == 0:
                logger.warning(
                    f"[Gen0 Repair] seed={i} compile_ok=0, error_type={compile_error_type}"
                )
                if not self.repair_config.enabled:
                    self._add_to_failure_buffer(
                        program_id=str(uuid.uuid4()),
                        generation=0,
                        compile_error_type=compile_error_type,
                        compile_error_msg=compile_error_msg,
                        statement=statement,
                        repair_attempts=0,
                        repair_llm_calls_used=0,
                        final_status="no_repair",
                    )
                    i += 1
                    continue

                                                                   
                 
                                                                                                            
                                                                                                             
                                                      
                if init_seed_source == "seed_bank" and _env_truthy(
                    "AUTOFORMAL_SKIP_SEEDBANK_GEN0_REPAIR", default=False
                ):
                    self._add_to_failure_buffer(
                        program_id=str(uuid.uuid4()),
                        generation=0,
                        compile_error_type=compile_error_type,
                        compile_error_msg=compile_error_msg,
                        statement=statement,
                        repair_attempts=0,
                        repair_llm_calls_used=0,
                        final_status="seedbank_skip_repair",
                    )
                    i += 1
                    continue

                dummy_job = RunningJob(
                    job_id=f"gen0_seed_{i}",
                    exec_fname=exec_fname,
                    results_dir=results_dir,
                    start_time=time.time(),
                    generation=0,
                    parent_id=None,
                    archive_insp_ids=[],
                    top_k_insp_ids=[],
                    code_diff=None,
                    meta_patch_data={
                        "patch_type": patch_type,
                        "patch_name": patch_name,
                        "patch_description": patch_description,
                        "init_seed_source": init_seed_source,
                        "init_seed_index": int(i),
                        "init_seed_bank_path": str(seed_bank_dir)
                        if (init_seed_source == "seed_bank" and seed_bank_dir is not None)
                        else None,
                    },
                )
                repair_item = RepairQueueItem(
                    program_id=str(uuid.uuid4()),
                    exec_fname=exec_fname,
                    results_dir=results_dir,
                    generation=0,
                    parent_id=None,
                    compile_error_type=compile_error_type,
                    compile_error_msg=compile_error_msg,
                    original_code=str(metrics_val.get("code") or evaluated_code or statement or ""),
                    repair_attempts=0,
                )
                repaired_program = self._process_repair(repair_item, dummy_job)
                if repaired_program is not None:
                    inserted_programs.append(repaired_program)
                    best_score = (
                        repaired_program.combined_score
                        if best_score is None
                        else max(best_score, repaired_program.combined_score or 0.0)
                    )
                i += 1
                continue

                                                                            
                                                                                               
                                                                                           
                                                              
            if (
                init_seed_source == "seed_bank"
                and self.evo_config.language == "lean"
                and is_trivial_tautology_placeholder(evaluated_code)
            ):
                logger.info(f"[Gen0] Skipping seed_bank placeholder seed={i}")
                self._add_to_failure_buffer(
                    program_id=str(uuid.uuid4()),
                    generation=0,
                    compile_error_type="seedbank_placeholder",
                    compile_error_msg="Rejected trivial placeholder from seed bank.",
                    statement=normalize_lean_statement(evaluated_code) or "",
                    repair_attempts=0,
                    repair_llm_calls_used=0,
                    final_status="seedbank_placeholder_skipped",
                )
                i += 1
                continue

                                                         
            code_embedding, e_cost = self.get_code_embedding(exec_fname)
            combined_score = metrics_val.get("combined_score", 0.0)
            public_metrics = metrics_val.get("public", {})
            private_metrics = metrics_val.get("private", {})
            text_feedback = metrics_val.get("text_feedback", "")
            stdout_log = results.get("stdout_log", "") if results else ""
            stderr_log = results.get("stderr_log", "") if results else ""

            db_program = Program(
                id=str(uuid.uuid4()),
                code=evaluated_code,
                language=self.evo_config.language,
                parent_id=None,
                generation=0,
                archive_inspiration_ids=[],
                top_k_inspiration_ids=[],
                code_diff=None,
                embedding=code_embedding,
                correct=True,                                              
                combined_score=combined_score,
                public_metrics=public_metrics,
                private_metrics=private_metrics,
                text_feedback=text_feedback,
                metadata={
                    "compute_time": rtime,
                    "api_costs": api_costs,
                    "embed_cost": e_cost,
                    "novelty_cost": 0.0,
                    "patch_type": patch_type,
                    "patch_name": patch_name,
                    "patch_description": patch_description,
                    "init_seed_source": init_seed_source,
                    "init_seed_index": int(i),
                    "init_seed_bank_path": str(seed_bank_dir)
                    if (init_seed_source == "seed_bank" and seed_bank_dir is not None)
                    else None,
                    "stdout_log": stdout_log,
                    "stderr_log": stderr_log,
                    "fitness_tuple": list(
                                                                                             
                        metrics_val.get("fitness_tuple", (1, 0, 0, 0.0))
                    ),
                    "cycle_log_prob": metrics_val.get("cycle_log_prob", None),
                    "cycle_normalized_log_prob": metrics_val.get(
                        "cycle_normalized_log_prob", None
                    ),
                    "cycle_score": metrics_val.get("cycle_score", None),
                },
            )

            self.db.add(db_program, verbose=True)
            inserted_programs.append(db_program)
            best_score = (
                combined_score
                if best_score is None
                else max(best_score, combined_score or 0.0)
            )
            self.meta_summarizer.add_evaluated_program(db_program)
            i += 1

        if not inserted_programs:
            if self.termination_reason is None:
                self.termination_reason = (
                    f"gen0_no_compile_ok_seed_after_{min_num_init}_attempts"
                )
            logger.warning(
                f"[Gen0] No compile_ok=1 seeds found (attempts={i}, repair enabled={self.repair_config.enabled}); "
                "stopping before submitting Gen>=1 jobs."
            )
            self.db.save()
            return

                                                             
        if self.llm_selection is not None and best_score is not None:
            self.llm_selection.set_baseline_score(best_score)

        self.db.save()
        self._update_best_solution()
        logger.info(f"[Gen0] Bootstrapped {len(inserted_programs)} compile_ok=1 seeds")

                                                                                         
                                                                                  
                                             
        if not self._file_seed0_pruned:
            has_non_file_seed = any(
                str((p.metadata or {}).get("init_seed_source", "")).strip().lower()
                in {"llm", "seed_bank"}
                for p in inserted_programs
            )
            if has_non_file_seed:
                delete_fn = getattr(self.db, "delete_file_seed0", None)
                deleted_count = int(delete_fn()) if callable(delete_fn) else 0
                if deleted_count > 0:
                    logger.info(
                        f"[Prune] Removed {deleted_count} file seed0 program(s) from archive/DB "
                        "after Gen0 bootstrapping"
                    )
                self._file_seed0_pruned = True

    def _update_best_solution(self):
        """
        Override: update the `best/` directory more robustly.

        We support two directory layouts:
        - Gen 0: gen_0/seed_X/main.lean (multiple seeds)
        - Gen 1+: gen_N/main.lean (single file)

        We locate the correct source by matching file contents.
        Note: Gen0 may generate `main_repairK.lean` during repair, and the best program may come
        from that file. Therefore we scan `main*.lean` under each seed directory (not only `main.lean`).
        """
        best_programs = self.db.get_top_programs(n=1, correct_only=True)
        if not best_programs:
            if self.verbose:
                logger.debug(
                    "No correct programs found yet, cannot determine best solution."
                )
            return

        best_program = best_programs[0]

        if best_program.id == self.best_program_id:
            return             

        self.best_program_id = best_program.id

                                                      
        gen_dir = Path(self.results_dir) / f"gen_{best_program.generation}"
        source_path = None
        matched_main_file = None

        if best_program.generation == 0:
                                                                        
            for seed_dir in sorted(gen_dir.glob("seed_*")):
                for candidate in sorted(seed_dir.glob(f"main*.{self.lang_ext}")):
                    if not candidate.exists():
                        continue
                    content = candidate.read_text(encoding="utf-8")
                                                                       
                    if content.strip() == best_program.code.strip():
                        source_path = seed_dir
                        matched_main_file = candidate
                        break
                if source_path is not None:
                    break
            if source_path is None:
                logger.warning(
                    f"[Best] Could not find matching seed for best program "
                    f"{best_program.id[:8]} in gen_0"
                )
                return
        else:
                                                   
            source_path = gen_dir

        if not source_path.exists():
            logger.warning(
                f"[Best] Source directory does not exist: {source_path}"
            )
            return

        best_dir = Path(self.results_dir) / "best"

        if best_dir.exists():
            shutil.rmtree(best_dir)

        shutil.copytree(source_path, best_dir)

                                                                          
                                                                     
        if best_program.generation == 0 and matched_main_file is not None:
            best_main = best_dir / f"main.{self.lang_ext}"
            try:
                best_main.write_text(
                    matched_main_file.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            except Exception:
                logger.exception(
                    f"[Best] Failed to overwrite {best_main} from {matched_main_file}"
                )

        if self.verbose:
            logger.info(
                f"New best program found: gen {best_program.generation}, "
                f"id {best_program.id[:6]}... "
                f"Copied from {source_path.name} to {best_dir}"
            )

    def _process_completed_job(self, job: RunningJob):
        """
        Process a completed evaluation job.

        ================================================================================
                This is the core entrypoint for the repair logic
        ================================================================================

        Execution flow
        1. Read evaluation results (metrics)
        2. Check compile_ok:
           - compile_ok = 1 → insert into archive (normal path)
           - compile_ok = 0 → enter repair path
        3. Repair path:
           - If repair enabled → attempt repair
           - If repair disabled → record in failure_buffer

        Constraints enforced
        - [A] compile_ok=0 candidates skip novelty (do not call parent novelty checks)
        - [B] compile_ok=0 candidates are not inserted into archive (do not call db.add)
        """
        end_time = time.time()
        rtime = end_time - job.start_time

                                 
        results = self.scheduler.get_job_results(job.job_id, job.results_dir)

                              
        file_exists = Path(job.exec_fname).exists()
        try:
            evaluated_code = Path(job.exec_fname).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not read code for job {job.job_id}. Error: {e}")
            evaluated_code = ""

                                                                               
                                                                                    
                                                                               
                                                                                     
        if not file_exists:
            logger.warning(
                f"[Patch Failed] File does not exist for gen {job.generation}: "
                f"{job.exec_fname}. Skipping repair (no code to repair)."
            )
            self._add_to_failure_buffer(
                program_id=str(uuid.uuid4()),
                generation=job.generation,
                compile_error_type="patch_generation_failed",
                compile_error_msg="Patch generation failed - no file created",
                statement="",
                repair_attempts=0,
                repair_llm_calls_used=0,
                final_status="patch_failed",
            )
            return

                          
        metrics_val = results.get("metrics", {}) if results else {}
        compile_ok = metrics_val.get("compile_ok", 0)
        compile_error_type = metrics_val.get("compile_error_type", "")
        compile_error_msg = metrics_val.get("compile_error_msg", "")
        statement = metrics_val.get("statement", "")
        original_code = str(metrics_val.get("code") or evaluated_code or "")

                                                                               
                                                  
                                                                               
        if compile_ok == 0:
                                                                             
                                                                        
            try:
                pseudo_program = Program(
                    id=str(uuid.uuid4()),
                    code=evaluated_code,
                    language=self.evo_config.language,
                    parent_id=job.parent_id,
                    generation=job.generation,
                    archive_inspiration_ids=job.archive_insp_ids,
                    top_k_inspiration_ids=job.top_k_insp_ids,
                    combined_score=float(metrics_val.get("combined_score", 0.0) or 0.0),
                    public_metrics=metrics_val.get("public", {}) or {"compile_ok": 0},
                    private_metrics=metrics_val.get("private", {}) or {
                        "compile_error_type": compile_error_type,
                        "compile_error_msg": compile_error_msg,
                    },
                    correct=False,
                    metadata={
                        "patch_type": (job.meta_patch_data or {}).get("patch_type"),
                        "error_attempt": (job.meta_patch_data or {}).get("error_attempt"),
                        "compile_error_type": compile_error_type,
                        "compile_error_msg": compile_error_msg,
                    },
                )
                self.meta_summarizer.add_evaluated_program(pseudo_program)
            except Exception:
                pass
            logger.info(
                f"[Repair] compile_ok=0 detected for gen {job.generation}, "
                f"error_type: {compile_error_type}"
            )

                                                                                    
                                                                                        
            if not statement and evaluated_code:
                                                                 
                statement = normalize_lean_statement(evaluated_code)
                if statement:
                    logger.info(
                        f"[Repair] Recovered statement from file content for gen {job.generation}"
                    )

                                                                      
            if not original_code and not compile_error_msg:
                logger.warning(
                    f"[Repair] Empty code and no error info for gen {job.generation}. "
                    f"Skipping repair (nothing to repair)."
                )
                self._add_to_failure_buffer(
                    program_id=str(uuid.uuid4()),
                    generation=job.generation,
                    compile_error_type="empty_statement",
                    compile_error_msg="Code is empty after normalization",
                    statement="",
                    repair_attempts=0,
                    repair_llm_calls_used=0,
                    final_status="empty_code",
                )
                return

            def _maybe_run_evolast_compile_fallback(
                *,
                reason: str,
                attempt: int,
                write_llm_fail_marker: bool,
            ) -> None:
                """Try EvolAST on parent to produce a compile_ok=1 fallback candidate.

                Contract (per user spec):
                - EvolAST output is NEVER sent to LLM repair.
                - EvolAST is used as a compile-safety fallback and/or diversity booster.
                """
                if not (
                    self.enable_evolast_fallback
                    and self.evo_config.language == "lean"
                    and job.parent_id
                ):
                    return

                                                                                         
                                                                                                          
                if write_llm_fail_marker:
                    try:
                        exec_path = Path(job.exec_fname)
                        suffix = exec_path.suffix or f".{self.lang_ext}"
                        llm_fail_path = exec_path.with_suffix(f".llm_fail{suffix}")
                        if exec_path.exists() and not llm_fail_path.exists():
                            llm_fail_path.write_text(exec_path.read_text(encoding="utf-8"), encoding="utf-8")
                    except Exception:
                        pass

                parent_code = ""
                for p in self.db.get_all_programs():
                    if getattr(p, "id", None) == job.parent_id:
                        parent_code = str(getattr(p, "code", "") or "")
                        break
                if not parent_code.strip():
                    return

                try:
                    from evolast_lean import apply_evolast_to_lean_code, parse_rule_weights

                    evolast_mode = os.environ.get("AUTOFORMAL_EVOLAST_MODE", "").strip() or "aggressive"
                    evolast_p = 0.35
                    evolast_max_rewrites = 32
                    try:
                        raw_p = str(os.environ.get("AUTOFORMAL_EVOLAST_P", "") or "").strip()
                        if raw_p:
                            evolast_p = float(raw_p)
                    except Exception:
                        evolast_p = 0.35
                    try:
                        raw_m = str(os.environ.get("AUTOFORMAL_EVOLAST_MAX_REWRITES", "") or "").strip()
                        if raw_m:
                            evolast_max_rewrites = int(raw_m)
                    except Exception:
                        evolast_max_rewrites = 32
                    evolast_weights = parse_rule_weights(os.environ.get("AUTOFORMAL_EVOLAST_RULE_WEIGHTS"))

                                                                                                
                    seed_material = f"{job.parent_id}|{job.generation}|{attempt}|{reason}"
                    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)
                    fallback_code, fallback_info = apply_evolast_to_lean_code(
                        parent_code,
                        mode=evolast_mode,
                        p=float(evolast_p),
                        max_rewrites=int(evolast_max_rewrites),
                        rule_weights=dict(evolast_weights),
                        seed=seed,
                    )
                    fallback_code = str(fallback_code or "").strip()
                    if not fallback_code or fallback_code.strip() == parent_code.strip():
                        return

                    evolast_fname = job.exec_fname.replace(
                        f".{self.lang_ext}", f"_evolast{attempt}.{self.lang_ext}"
                    )
                    Path(evolast_fname).write_text(fallback_code, encoding="utf-8")
                    evolast_results_dir = Path(job.exec_fname).parent / f"results_evolast{attempt}"
                    evolast_results_dir.mkdir(parents=True, exist_ok=True)

                    results2, rtime2 = self.scheduler.run(str(evolast_fname), str(evolast_results_dir))
                    metrics2 = results2.get("metrics", {}) if results2 else {}
                    if int(metrics2.get("compile_ok", 0) or 0) != 1:
                        return

                    code_embedding2, e_cost2 = self.get_code_embedding(str(evolast_fname))
                    db_program = Program(
                        id=str(uuid.uuid4()),
                        code=fallback_code,
                        language=self.evo_config.language,
                        parent_id=job.parent_id,
                        generation=job.generation,
                        archive_inspiration_ids=job.archive_insp_ids,
                        top_k_inspiration_ids=job.top_k_insp_ids,
                        code_diff=None,
                        embedding=code_embedding2,
                        correct=True,
                        combined_score=float(metrics2.get("combined_score", 0.0) or 0.0),
                        public_metrics=metrics2.get("public", {}) or {"compile_ok": 1},
                        private_metrics=metrics2.get("private", {}) or {},
                        text_feedback=metrics2.get("text_feedback", "") or "",
                        metadata={
                            "compute_time": float(rtime2 or 0.0),
                            **(job.meta_patch_data or {}),
                            "embed_cost": float(e_cost2 or 0.0),
                            "novelty_cost": 0.0,
                            "patch_type": "evolast",
                            "patch_name": f"evolast_{evolast_mode}",
                            "patch_description": f"EvolAST fallback on parent after compile-fail ({reason}).",
                            "evolast_fallback_used": 1,
                            "evolast_fallback_reason": str(reason or ""),
                            "evolast_fallback_attempt": int(attempt),
                            "evolast_fallback_seed": int(seed),
                            "evolast_fallback_mode": evolast_mode,
                            "evolast_fallback_info": fallback_info
                            if isinstance(fallback_info, dict)
                            else {"info": str(fallback_info)},
                        },
                    )
                    self.db.add(db_program, verbose=True)
                    self.db.save()
                    self._update_best_solution()
                    self._post_process_program(db_program, job)
                    logger.info(
                        f"[EvolAST] Fallback SUCCESS: gen={job.generation} attempt={attempt} "
                        f"new_id={db_program.id[:8]} reason={reason}"
                    )
                except Exception as e:
                    if self.verbose:
                        logger.info(
                            f"[EvolAST] Fallback skipped: attempt={attempt} "
                            f"{type(e).__name__}: {e}"
                        )

            if self.repair_config.enabled:
                                                                                                     
                                                                                               
                _maybe_run_evolast_compile_fallback(
                    reason=f"compile_fail_immediate:{compile_error_type or 'unknown'}",
                    attempt=1,
                    write_llm_fail_marker=True,
                )

                                                        
                repair_item = RepairQueueItem(
                    program_id=str(uuid.uuid4()),
                    exec_fname=job.exec_fname,
                    results_dir=job.results_dir,
                    generation=job.generation,
                    parent_id=job.parent_id,
                    compile_error_type=compile_error_type,
                    compile_error_msg=compile_error_msg,
                    original_code=original_code,
                    repair_attempts=0,
                )
                repaired_program = self._process_repair(repair_item, job)

                                                                                                                
                if repaired_program is None:
                    _maybe_run_evolast_compile_fallback(
                        reason=f"compile_fail_repair_exhausted:{compile_error_type or 'unknown'}",
                        attempt=2,
                        write_llm_fail_marker=False,
                    )
            else:
                                                           
                self._add_to_failure_buffer(
                    program_id=str(uuid.uuid4()),
                    generation=job.generation,
                    compile_error_type=compile_error_type,
                    compile_error_msg=compile_error_msg,
                    statement=statement,
                    repair_attempts=0,
                    repair_llm_calls_used=0,
                    final_status="no_repair",
                )
                                                                             
                _maybe_run_evolast_compile_fallback(
                    reason=f"compile_fail_no_repair:{compile_error_type or 'unknown'}",
                    attempt=1,
                    write_llm_fail_marker=True,
                )
                                                                                  
            return

                                                                               
                                   
                                                                               
                                                     
        code_embedding = job.code_embedding
        e_cost = job.embed_cost
        n_cost = job.novelty_cost

        correct_val = False
        stdout_log = ""
        stderr_log = ""
        if results:
                                                                                                                 
            correct_val = True
            stdout_log = results.get("stdout_log", "")
            stderr_log = results.get("stderr_log", "")

        combined_score = metrics_val.get("combined_score", 0.0)
        public_metrics = metrics_val.get("public", {})
        private_metrics = metrics_val.get("private", {})
        text_feedback = metrics_val.get("text_feedback", "")

                                                                                        
                                                                                                    
        if self.evo_config.language == "lean":
            stmt_for_guard = str(statement or "").strip()
            if not stmt_for_guard and evaluated_code:
                stmt_for_guard = normalize_lean_statement(evaluated_code)
            if stmt_for_guard and is_trivial_tautology_placeholder(stmt_for_guard):
                logger.warning(
                    f"[Filter] Trivial placeholder detected at gen {job.generation}; "
                    "marking as incorrect and excluding from archive."
                )
                self._add_to_failure_buffer(
                    program_id=str(uuid.uuid4()),
                    generation=job.generation,
                    compile_error_type="placeholder_trivial_tautology",
                    compile_error_msg="Forbidden placeholder: trivial/tautological statement",
                    statement=stmt_for_guard,
                    repair_attempts=0,
                    repair_llm_calls_used=0,
                    final_status="placeholder_filtered",
                )

                                                                                             
                filtered_program = Program(
                    id=str(uuid.uuid4()),
                    code=evaluated_code,
                    language=self.evo_config.language,
                    parent_id=job.parent_id,
                    generation=job.generation,
                    archive_inspiration_ids=job.archive_insp_ids,
                    top_k_inspiration_ids=job.top_k_insp_ids,
                    code_diff=job.code_diff,
                    embedding=code_embedding,
                    correct=False,
                    combined_score=0.0,
                    public_metrics=public_metrics,
                    private_metrics=private_metrics,
                    text_feedback=text_feedback,
                    metadata={
                        "compute_time": rtime,
                        **(job.meta_patch_data or {}),
                        "embed_cost": e_cost,
                        "novelty_cost": n_cost,
                        "stdout_log": stdout_log,
                        "stderr_log": stderr_log,
                        "hard_filter_reason": "placeholder_trivial_tautology",
                    },
                )
                self.db.add(filtered_program, verbose=True)
                self.db.save()
                return

                                                                   
        db_program = Program(
            id=str(uuid.uuid4()),
            code=evaluated_code,
            language=self.evo_config.language,
            parent_id=job.parent_id,
            generation=job.generation,
            archive_inspiration_ids=job.archive_insp_ids,
            top_k_inspiration_ids=job.top_k_insp_ids,
            code_diff=job.code_diff,
            embedding=code_embedding,
            correct=correct_val,
            combined_score=combined_score,
            public_metrics=public_metrics,
            private_metrics=private_metrics,
            text_feedback=text_feedback,
            metadata={
                "compute_time": rtime,
                **(job.meta_patch_data or {}),
                "embed_cost": e_cost,
                "novelty_cost": n_cost,
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
                "fitness_tuple": list(metrics_val.get("fitness_tuple", (1, 0, 0, 0.0))),
                "cycle_log_prob": metrics_val.get("cycle_log_prob"),
                "cycle_normalized_log_prob": metrics_val.get("cycle_normalized_log_prob"),
                "cycle_score": metrics_val.get("cycle_score", 0.0),
            },
        )

        self.db.add(db_program, verbose=True)
        logger.info(
            f"[Archive] Added compile_ok=1 candidate: gen={job.generation}, "
            f"score={combined_score:.2f}, tuple={metrics_val.get('fitness_tuple')}"
        )

                                                                                             
                                                                                            
                                          
        if not self._file_seed0_pruned and int(getattr(job, "generation", 0) or 0) > 0:
            delete_fn = getattr(self.db, "delete_file_seed0", None)
            deleted_count = int(delete_fn()) if callable(delete_fn) else 0
            if deleted_count > 0:
                logger.info(
                    f"[Prune] Removed {deleted_count} file seed0 program(s) from archive "
                    "after first Gen>=1 compile_ok=1 program entered archive"
                )
            self._file_seed0_pruned = True

                                                             
        self._post_process_program(db_program, job)

    def _process_repair(self, repair_item: RepairQueueItem, original_job: RunningJob):
        """
        Repair compile-fail candidates.

        ================================================================================
                     Core implementation of the repair mechanism
        ================================================================================

        Execution flow
        for attempt in range(max_attempts):
            1. Build repair prompt (with compiler feedback)
            2. Call LLM to generate a repaired version (temp=0)
            3. Constraint C: count toward budget
            4. Extract repaired code
            5. Re-evaluate
            6. Check compile_ok:
               - success → insert into archive, return
               - failure → update error info and continue to next attempt

        Why temp=0?
        Repair aims for correctness rather than diversity. Higher temperatures can create
        more variants, but we want minimal edits that fix compilation, not creative rewrites.

        Args:
            repair_item: the candidate to repair
            original_job: original job object (for context)
        """
        max_attempts = self.repair_config.max_repair_attempts
        if repair_item.generation == 0 and self.repair_config.max_repair_attempts_gen0:
            max_attempts = int(self.repair_config.max_repair_attempts_gen0)

                                                                        
        try:
            skip_types = {t.lower() for t in (getattr(self, "repair_skip_error_types", set()) or set())}
        except Exception:
            skip_types = set()
        err_type_norm = str(repair_item.compile_error_type or "").strip().lower()
        if skip_types and err_type_norm in skip_types:
            logger.info(
                f"[Repair] Skipping compile repair due to policy: error_type={repair_item.compile_error_type}"
            )
            self._add_to_failure_buffer(
                program_id=repair_item.program_id,
                generation=repair_item.generation,
                compile_error_type=repair_item.compile_error_type,
                compile_error_msg=repair_item.compile_error_msg,
                statement=normalize_lean_statement(repair_item.original_code),
                repair_attempts=0,
                repair_llm_calls_used=0,
                final_status="repair_skipped",
            )
            return None

        dump_repair_raw = _env_truthy("AUTOFORMAL_DUMP_REPAIR_RAW")
        dump_dir: Optional[Path] = None
        if dump_repair_raw:
            dump_dir = Path(self.results_dir) / "repair_dumps"
            dump_dir.mkdir(parents=True, exist_ok=True)

        def _maybe_dump_repair_attempt(
            *,
            payload: Dict[str, Any],
            raw_content: Optional[str],
        ) -> None:
            if dump_dir is None:
                return
            try:
                program_id = str(payload.get("program_id") or repair_item.program_id or "unknown")
                generation = int(payload.get("generation") or repair_item.generation or 0)
                attempt = int(payload.get("attempt") or 0)
                stem = f"{program_id}_gen{generation}_attempt{attempt:02d}"
                (dump_dir / f"{stem}.json").write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                if raw_content is not None:
                    (dump_dir / f"{stem}.raw.txt").write_text(raw_content, encoding="utf-8")
            except Exception:
                                                                        
                return

        budget_reason_hit: Optional[str] = None
        for attempt in range(max_attempts):
                                                                             
            budget_reason = self._check_budget_exhausted()
            if budget_reason is not None:
                budget_reason_hit = budget_reason
                self.termination_reason = budget_reason
                logger.info(f"[Repair] Budget exhausted before attempt {attempt + 1}: {budget_reason}")
                break

            repair_item.repair_attempts = attempt + 1
            logger.info(
                f"[Repair] Attempt {attempt + 1}/{max_attempts} for gen {repair_item.generation}"
            )

                                  
            sys_msg, user_msg = build_repair_prompt(
                original_code=repair_item.original_code,
                compile_error_type=repair_item.compile_error_type,
                compile_error_msg=repair_item.compile_error_msg,
                informal=self.problem_config.get("informal", ""),
                reference_header=self.problem_config.get("header", ""),
            )

            llm_kwargs = self.llm.get_kwargs()
            llm_kwargs["temperature"] = self.repair_config.repair_temperature

                                                                                                            
            repair_llm = self._repair_llm_override()
            before_calls = getattr(self.llm, "total_calls", None)
            response = None
            llm_kwargs_eff = dict(llm_kwargs or {})
            try:
                if repair_llm.get("enabled"):
                    logger.info(
                        f"[Repair] Using repair-only LLM override: "
                        f"model={repair_llm.get('model_name','')}, "
                        f"base_url={repair_llm.get('base_url','')}"
                    )
                    llm_kwargs_eff["model_name"] = str(repair_llm.get("model_name") or "")
                    with _temporary_env({"OPENAI_LLM_BASE_URL": str(repair_llm.get("base_url") or "")}):
                        response = self.llm.query(
                            msg=user_msg,
                            system_msg=sys_msg,
                            llm_kwargs=llm_kwargs_eff,
                        )
                else:
                    response = self.llm.query(
                        msg=user_msg,
                        system_msg=sys_msg,
                        llm_kwargs=llm_kwargs_eff,
                    )
            except Exception as e:
                logger.warning(
                    f"[Repair] LLM query failed at attempt {attempt + 1}: {type(e).__name__}: {e}"
                )
                response = None
            dump_payload: Dict[str, Any] = {
                "timestamp_unix": time.time(),
                "results_dir": str(self.results_dir),
                "program_id": repair_item.program_id,
                "generation": repair_item.generation,
                "attempt": attempt + 1,
                "max_attempts": max_attempts,
                "exec_fname": repair_item.exec_fname,
                "parent_id": repair_item.parent_id,
                "compile_error_type": repair_item.compile_error_type,
                "compile_error_msg": repair_item.compile_error_msg,
                "original_code": repair_item.original_code,
                "system_msg": sys_msg,
                "user_msg": user_msg,
                "llm_kwargs": dict(llm_kwargs_eff or {}),
                "response": {
                    "model_name": getattr(response, "model_name", None) if response else None,
                    "cost": float(getattr(response, "cost", 0.0) or 0.0) if response else None,
                    "input_tokens": getattr(response, "input_tokens", None) if response else None,
                    "output_tokens": getattr(response, "output_tokens", None) if response else None,
                },
                "extract_between_code_fence": None,
                "adapter_parsed_statement": None,
                "normalized_statement": None,
                "decision": None,
            }
            after_calls = getattr(self.llm, "total_calls", None)
            if isinstance(before_calls, int) and isinstance(after_calls, int):
                delta = max(after_calls - before_calls, 0)
                calls_used = delta if delta > 0 else 1
                self.total_repair_llm_calls += calls_used
                repair_item.total_repair_llm_calls_used += calls_used
            else:
                                                    
                self.total_repair_llm_calls += 1
                repair_item.total_repair_llm_calls_used += 1

            if response is None or response.content is None:
                logger.warning(f"[Repair] LLM response was None, attempt {attempt + 1}")
                dump_payload["decision"] = "llm_response_none"
                _maybe_dump_repair_attempt(payload=dump_payload, raw_content=None)
                continue

            repair_item.total_repair_cost += response.cost or 0
            self.total_repair_cost += response.cost or 0

            raw_content = response.content

            repaired_code, extracted_source = extract_best_lean_code_block_with_source(raw_content)
            dump_payload["extract_between_code_fence"] = repaired_code

            normalized_source = extracted_source if repaired_code else ""

                                                                           
            if not repaired_code:
                model_name = (
                    str(getattr(response, "model_name", "") or "").strip()
                    or str(repair_llm.get("model_name") or "").strip()
                    or (self.evo_config.llm_models[0] if self.evo_config.llm_models else "")
                )
                adapter = get_model_adapter(model_name)
                parsed_code = adapter.parse_output(raw_content) if adapter else None
                dump_payload["adapter_parsed_code"] = parsed_code
                repaired_code = normalize_lean_code(parsed_code or "")
                if normalize_lean_statement(repaired_code):
                    normalized_source = "adapter"
                else:
                    repaired_code = ""

                                                                                                        
            if not repaired_code:
                repaired_code = normalize_lean_code(raw_content)
                if normalize_lean_statement(repaired_code):
                    normalized_source = "raw_output"
                else:
                    repaired_code = ""

            dump_payload["normalized_code"] = repaired_code
            dump_payload["normalized_source"] = normalized_source
            dump_payload["normalized_statement_preview"] = normalize_lean_statement(repaired_code)[:200]

            if not repaired_code:
                logger.warning(f"[Repair] Empty/invalid Lean code after cleanup, attempt {attempt + 1}")
                dump_payload["decision"] = "empty_code_after_cleanup"
                _maybe_dump_repair_attempt(payload=dump_payload, raw_content=raw_content)
                continue

            dump_payload["decision"] = "code_extracted"
            _maybe_dump_repair_attempt(payload=dump_payload, raw_content=raw_content)

                                                                
            self.total_repair_evals += 1

                                           
            repair_fname = repair_item.exec_fname.replace(
                f".{self.lang_ext}", f"_repair{attempt + 1}.{self.lang_ext}"
            )
            repaired_program = self._build_repaired_program(repaired_code)
            with open(repair_fname, "w") as f:
                f.write(repaired_program)

                             
            results, rtime = self.scheduler.run(repair_fname, repair_item.results_dir)

            if results:
                metrics = results.get("metrics", {})
                new_compile_ok = metrics.get("compile_ok", 0)

                if new_compile_ok == 1:
                                       
                    logger.info(
                        f"[Repair] SUCCESS! Compile fixed after {attempt + 1} attempts"
                    )

                                          
                    db_program = self._add_repaired_to_archive(
                        repair_item,
                        repaired_program,
                        repair_fname,
                        metrics,
                        rtime,
                        original_job,
                    )

                                                                
                    self._add_to_failure_buffer(
                        program_id=repair_item.program_id,
                        generation=repair_item.generation,
                        compile_error_type=repair_item.compile_error_type,
                        compile_error_msg=repair_item.compile_error_msg,
                        statement=normalize_lean_statement(repair_item.original_code),
                        repair_attempts=attempt + 1,
                        repair_llm_calls_used=repair_item.total_repair_llm_calls_used,
                        final_status="repair_success",
                    )
                    return db_program

                                                                                
                repair_item.compile_error_type = metrics.get("compile_error_type", "")
                repair_item.compile_error_msg = metrics.get("compile_error_msg", "")
                repair_item.original_code = str(metrics.get("code") or repaired_code or "")

        if budget_reason_hit is not None:
                                                                              
            self._add_to_failure_buffer(
                program_id=repair_item.program_id,
                generation=repair_item.generation,
                compile_error_type=repair_item.compile_error_type,
                compile_error_msg=repair_item.compile_error_msg,
                statement=normalize_lean_statement(repair_item.original_code),
                repair_attempts=repair_item.repair_attempts,
                repair_llm_calls_used=repair_item.total_repair_llm_calls_used,
                final_status="repair_budget_exhausted",
            )
            return None

                                                   
        logger.warning(
            f"[Repair] EXHAUSTED after {max_attempts} attempts for gen {repair_item.generation}"
        )
        self._add_to_failure_buffer(
            program_id=repair_item.program_id,
            generation=repair_item.generation,
            compile_error_type=repair_item.compile_error_type,
            compile_error_msg=repair_item.compile_error_msg,
            statement=normalize_lean_statement(repair_item.original_code),
            repair_attempts=max_attempts,
            repair_llm_calls_used=repair_item.total_repair_llm_calls_used,
            final_status="repair_exhausted",
        )
        return None

    def _process_semantic_repair(
        self,
        *,
        base_program: Program,
        base_job: RunningJob,
        base_metrics: Dict[str, Any],
        evaluated_code: str,
    ) -> Optional[Program]:
        """Semantic repair for compile_ok=1 but semantic_ok=0 candidates (when enabled).

        Workflow:
        - Provide CriticLean's "5. Accuracy Confirmation" feedback to the LLM.
        - Allow changing hypotheses/conclusion to align semantics.
        - Require the result to still compile.
        - Success criterion: compile_ok=1 AND semantic_ok=1.
        """
        if not self.repair_config.enabled:
            return None

        if not bool(self.semantic_repair_enabled):
            return None

        use_semantic = bool(self.problem_config.get("use_semantic", False))
        if not use_semantic:
            return None

                                                               
        start_calls = getattr(self, "semantic_repair_start_calls", None)
        if start_calls is not None:
            try:
                raw_llm_api_calls = int(getattr(self.llm, "total_calls", 0) or 0)
            except Exception:
                raw_llm_api_calls = 0
            try:
                debited = int(getattr(self, "seedbank_debited_calls", 0) or 0)
            except Exception:
                debited = 0
            budget_calls = raw_llm_api_calls + debited
            if budget_calls < int(start_calls):
                return None

        max_attempts = int(getattr(self.repair_config, "max_repair_attempts", 0) or 0)
        if base_job.generation == 0 and getattr(self.repair_config, "max_repair_attempts_gen0", 0):
            max_attempts = int(getattr(self.repair_config, "max_repair_attempts_gen0") or max_attempts)
                                                                                         
        override: Optional[int] = None
        if base_job.generation == 0 and self.semantic_repair_max_attempts_gen0 is not None:
            override = self.semantic_repair_max_attempts_gen0
        elif self.semantic_repair_max_attempts is not None:
            override = self.semantic_repair_max_attempts
        if override is not None:
            max_attempts = int(override)
        if max_attempts <= 0:
            return None

        informal = str(self.problem_config.get("informal", "") or "")
        reference_header = str(self.problem_config.get("header", "") or "")

        current_code = str(base_metrics.get("code") or normalize_lean_code(evaluated_code) or evaluated_code or "")
        current_accuracy = str(
            base_metrics.get("critic_accuracy_confirmation")
            or _extract_accuracy_confirmation_from_critic_raw(base_metrics.get("critic_raw", ""))
            or ""
        ).strip()
        previous_compile_error = ""

                                                                                                      
        if not current_accuracy:
            return None

        for attempt in range(max_attempts):
            budget_reason = self._check_budget_exhausted()
            if budget_reason is not None:
                self.termination_reason = budget_reason
                logger.info(f"[SemanticRepair] Budget exhausted before attempt {attempt + 1}: {budget_reason}")
                break

            logger.info(
                f"[SemanticRepair] Attempt {attempt + 1}/{max_attempts} "
                f"for gen {base_job.generation} (base_id={base_program.id[:8]})"
            )

            sys_msg, user_msg = build_semantic_repair_prompt(
                original_code=current_code,
                informal=informal,
                accuracy_confirmation=current_accuracy,
                reference_header=reference_header,
                previous_compile_error=previous_compile_error,
            )

            llm_kwargs = self.llm.get_kwargs()
                                                                                       
            llm_kwargs["temperature"] = float(self.semantic_repair_temperature)

            repair_llm = self._repair_llm_override()
            before_calls = getattr(self.llm, "total_calls", None)
            response = None
            llm_kwargs_eff = dict(llm_kwargs or {})
            try:
                if repair_llm.get("enabled"):
                    logger.info(
                        f"[SemanticRepair] Using repair-only LLM override: "
                        f"model={repair_llm.get('model_name','')}, "
                        f"base_url={repair_llm.get('base_url','')}"
                    )
                    llm_kwargs_eff["model_name"] = str(repair_llm.get("model_name") or "")
                    with _temporary_env({"OPENAI_LLM_BASE_URL": str(repair_llm.get("base_url") or "")}):
                        response = self.llm.query(msg=user_msg, system_msg=sys_msg, llm_kwargs=llm_kwargs_eff)
                else:
                    response = self.llm.query(msg=user_msg, system_msg=sys_msg, llm_kwargs=llm_kwargs_eff)
            except Exception as e:
                logger.warning(
                    f"[SemanticRepair] LLM query failed at attempt {attempt + 1}: {type(e).__name__}: {e}"
                )
                response = None
            after_calls = getattr(self.llm, "total_calls", None)
            if isinstance(before_calls, int) and isinstance(after_calls, int):
                delta = max(after_calls - before_calls, 0)
                calls_used = delta if delta > 0 else 1
            else:
                calls_used = 1
            self.total_semantic_repair_llm_calls += calls_used
            self.total_semantic_repair_cost += float(getattr(response, "cost", 0.0) or 0.0)

            if response is None or response.content is None:
                logger.warning(f"[SemanticRepair] LLM response was None, attempt {attempt + 1}")
                continue

            raw_content = response.content

                                                                                                   
            repaired_code = extract_best_lean_code_block(raw_content) or ""
            if not repaired_code:
                model_name = (
                    str(getattr(response, "model_name", "") or "").strip()
                    or str(repair_llm.get("model_name") or "").strip()
                    or (self.evo_config.llm_models[0] if self.evo_config.llm_models else "")
                )
                adapter = get_model_adapter(model_name)
                parsed_code = adapter.parse_output(raw_content) if adapter else None
                cand = normalize_lean_code(parsed_code or "")
                if normalize_lean_statement(cand):
                    repaired_code = cand

            if not repaired_code:
                logger.warning(f"[SemanticRepair] Empty/invalid Lean code, attempt {attempt + 1}")
                continue

                                                     
            self.total_semantic_repair_evals += 1
            repair_fname = base_job.exec_fname.replace(
                f".{self.lang_ext}", f"_semrepair{attempt + 1}.{self.lang_ext}"
            )
            repaired_program = self._build_repaired_program(repaired_code)
            Path(repair_fname).write_text(repaired_program, encoding="utf-8")

                                                                                                  
            sem_results_dir = Path(base_job.exec_fname).parent / f"results_semrepair_{attempt + 1:02d}"
            sem_results_dir.mkdir(parents=True, exist_ok=True)
            results, rtime = self.scheduler.run(repair_fname, str(sem_results_dir))

            metrics = results.get("metrics", {}) if results else {}
            compile_ok = int(metrics.get("compile_ok", 0) or 0)
            semantic_ok = int(metrics.get("semantic_ok", 0) or 0)

            if compile_ok != 1:
                previous_compile_error = str(metrics.get("compile_error_msg", "") or "")
                current_code = str(metrics.get("code") or repaired_code or "")
                logger.info(
                    f"[SemanticRepair] compile_ok=0 after attempt {attempt + 1}; "
                    "continuing with compiler feedback."
                )
                continue

                                                                 
            current_code = str(metrics.get("code") or repaired_code or "")
            previous_compile_error = ""

            if semantic_ok == 1:
                                                                              
                code_embedding, e_cost = self.get_code_embedding(repair_fname)
                stdout_log = results.get("stdout_log", "") if results else ""
                stderr_log = results.get("stderr_log", "") if results else ""

                sem_job = RunningJob(
                    job_id=f"semantic_repair_{base_program.id[:8]}_{attempt + 1}",
                    exec_fname=repair_fname,
                    results_dir=str(sem_results_dir),
                    start_time=time.time(),
                    generation=base_job.generation,
                    parent_id=base_program.id,
                    archive_insp_ids=getattr(base_job, "archive_insp_ids", []) or [],
                    top_k_insp_ids=getattr(base_job, "top_k_insp_ids", []) or [],
                    code_diff=None,
                    meta_patch_data={
                        **(base_job.meta_patch_data or {}),
                        "patch_type": "semantic_repair",
                        "semantic_repair_attempt": int(attempt + 1),
                        "semantic_repair_source_program_id": base_program.id,
                    },
                )

                db_program = Program(
                    id=str(uuid.uuid4()),
                    code=repaired_program,
                    language=self.evo_config.language,
                    parent_id=base_program.id,
                    generation=base_job.generation,
                    archive_inspiration_ids=sem_job.archive_insp_ids,
                    top_k_inspiration_ids=sem_job.top_k_insp_ids,
                    code_diff=None,
                    embedding=code_embedding,
                    correct=True,
                    combined_score=float(metrics.get("combined_score", 0.0) or 0.0),
                    public_metrics=metrics.get("public", {}) or {},
                    private_metrics=metrics.get("private", {}) or {},
                    text_feedback=metrics.get("text_feedback", "") or "",
                    metadata={
                        "compute_time": rtime,
                        **(base_job.meta_patch_data or {}),
                        "embed_cost": e_cost,
                        "stdout_log": stdout_log,
                        "stderr_log": stderr_log,
                        "patch_type": "semantic_repair",
                        "semantic_repaired": True,
                        "semantic_repair_attempt": int(attempt + 1),
                        "semantic_repair_source_program_id": base_program.id,
                        "semantic_repair_accuracy_sha256": hashlib.sha256(
                            current_accuracy.encode("utf-8")
                        ).hexdigest()
                        if current_accuracy
                        else "",
                        "fitness_tuple": list(metrics.get("fitness_tuple", (1, 0, 0, 0.0))),
                        "cycle_log_prob": metrics.get("cycle_log_prob"),
                        "cycle_normalized_log_prob": metrics.get("cycle_normalized_log_prob"),
                        "cycle_score": metrics.get("cycle_score", 0.0),
                    },
                )

                self.db.add(db_program, verbose=True)
                self.total_semantic_repair_successes += 1
                logger.info(
                    f"[SemanticRepair] SUCCESS: gen={base_job.generation}, "
                    f"base_id={base_program.id[:8]}, new_id={db_program.id[:8]}"
                )
                self._post_process_program(db_program, sem_job)
                return db_program

                                                                                      
            current_accuracy = str(metrics.get("critic_accuracy_confirmation") or current_accuracy or "").strip()

        return None

    def _build_repaired_program(self, statement: str) -> str:
        """
        Build a complete Lean file from a repaired statement.
        """
        statement = statement.strip()
        if "EVOLVE-BLOCK-START" in statement or "EVOLVE-BLOCK-END" in statement:
            return statement
        return (
            "-- EVOLVE-BLOCK-START\n"
            f"{statement}\n"
            "-- EVOLVE-BLOCK-END\n"
        )

    def _add_repaired_to_archive(
        self,
        repair_item: RepairQueueItem,
        repaired_code: str,
        repaired_exec_fname: str,
        metrics: Dict[str, Any],
        compute_time: float,
        original_job: RunningJob,
    ) -> Program:
        """Insert a repaired program into the archive and return the inserted Program."""
        code_embedding, e_cost = self.get_code_embedding(repaired_exec_fname)

        db_program = Program(
            id=str(uuid.uuid4()),
            code=repaired_code,
            language=self.evo_config.language,
            parent_id=repair_item.parent_id,
            generation=repair_item.generation,
            archive_inspiration_ids=original_job.archive_insp_ids,
            top_k_inspiration_ids=original_job.top_k_insp_ids,
            code_diff=None,
            embedding=code_embedding,
            correct=metrics.get("compile_ok", 0) == 1,
            combined_score=metrics.get("combined_score", 0.0),
            public_metrics=metrics.get("public", {}),
            private_metrics=metrics.get("private", {}),
            text_feedback=metrics.get("text_feedback", ""),
            metadata={
                "compute_time": compute_time,
                **(original_job.meta_patch_data or {}),
                "embed_cost": e_cost,
                "repair_attempts": repair_item.repair_attempts,
                "repair_cost": repair_item.total_repair_cost,
                "llm_calls_used": int(repair_item.total_repair_llm_calls_used or 0),
                "original_error_type": repair_item.compile_error_type,
                "fitness_tuple": list(metrics.get("fitness_tuple", (1, 0, 0, 0.0))),
                "repaired": True,                            
                "cycle_log_prob": metrics.get("cycle_log_prob", None),
                "cycle_normalized_log_prob": metrics.get("cycle_normalized_log_prob", None),
                "cycle_score": metrics.get("cycle_score", None),
            },
        )

        self.db.add(db_program, verbose=True)
        self.db.save()
        self._update_best_solution()
        self.meta_summarizer.add_evaluated_program(db_program)
        return db_program

    def _add_to_failure_buffer(
        self,
        program_id: str,
        generation: int,
        compile_error_type: str,
        compile_error_msg: str,
        statement: str,
        repair_attempts: int,
        repair_llm_calls_used: int,
        final_status: str,
    ):
        """Add a record into failure_buffer."""
        record = FailureRecord(
            program_id=program_id,
            generation=generation,
            compile_error_type=compile_error_type,
            compile_error_msg=compile_error_msg,
            statement=statement,
            repair_attempts=repair_attempts,
            repair_llm_calls_used=int(repair_llm_calls_used or 0),
            final_status=final_status,
            timestamp=time.time(),
        )
        self.failure_buffer.add(record)

    def _post_process_program(self, db_program: Program, job: RunningJob):
        """Post-processing after a program is inserted into the DB/archive."""
                               
        self.meta_summarizer.add_evaluated_program(db_program)
                                                                              
                                                                       
        if self.meta_summarizer.should_update_meta(getattr(self.evo_config, "meta_rec_interval", None)):
            logger.info(
                f"[Meta] Updating meta memory after processing "
                f"{len(self.meta_summarizer.evaluated_since_last_meta)} programs..."
            )
            best_program = self.db.get_best_program()
            updated_recs, meta_cost = self.meta_summarizer.update_meta_memory(best_program)
            if updated_recs:
                self.meta_summarizer.write_meta_output(str(self.results_dir))
                if meta_cost and meta_cost > 0:
                    logger.info(f"[Meta] Update cost: ${meta_cost:.4f}")

                               
        if self.llm_selection is not None and db_program.metadata:
            if "model_name" in db_program.metadata:
                parent = (
                    self.db.get(db_program.parent_id) if db_program.parent_id else None
                )
                baseline = parent.combined_score if parent else None
                reward = db_program.combined_score if db_program.correct else None
                model_name = db_program.metadata["model_name"]
                self.llm_selection.update(
                    arm=model_name,
                    reward=reward,
                    baseline=baseline,
                )

        self.db.save()
        self._update_best_solution()
        self._save_meta_memory()

                                                                                  
        try:
            if self.semantic_repair_enabled and bool(self.problem_config.get("use_semantic", False)):
                pm = db_program.public_metrics or {}
                compile_ok = int(pm.get("compile_ok", 0) or 0)
                semantic_ok = int(pm.get("semantic_ok", 0) or 0)
                if compile_ok == 1 and semantic_ok == 0:
                                                                                                    
                    base_job = job
                    base_metrics = {}
                    try:
                        results = self.scheduler.get_job_results(job.job_id, job.results_dir)
                        base_metrics = (results or {}).get("metrics", {}) or {}
                    except Exception:
                        base_metrics = {}

                    evaluated_code = str(db_program.code or "")
                    self._process_semantic_repair(
                        base_program=db_program,
                        base_job=base_job,
                        base_metrics=base_metrics,
                        evaluated_code=evaluated_code,
                    )
        except Exception:
                                                                    
            pass

    def _maybe_update_meta_by_generation(self):
        """Update global meta once every N generations (N = evo_config.meta_rec_interval).

        NOTE: For this Lean task we interpret `meta_rec_interval` as a generation
        interval (per user spec: update every ~10 generations).
        """
        interval = int(getattr(self.evo_config, "meta_rec_interval", 0) or 0)
        if interval <= 0 or self.meta_llm is None:
            return
        if self.completed_generations <= 0:
            return
        if self.completed_generations == self._last_meta_update_generation:
            return
        if (self.completed_generations % interval) != 0:
            return
        if len(self.meta_summarizer.evaluated_since_last_meta) == 0:
            self._last_meta_update_generation = self.completed_generations
            return

        logger.info(
            f"[Meta] Triggered at completed_generations={self.completed_generations} "
            f"(interval={interval}, pending={len(self.meta_summarizer.evaluated_since_last_meta)})"
        )
        best_program = self.db.get_best_program()
        updated_recs, meta_cost = self.meta_summarizer.update_meta_memory(best_program)
        if updated_recs:
            self.meta_summarizer.write_meta_output(str(self.results_dir))
            if meta_cost and meta_cost > 0:
                logger.info(f"[Meta] Update cost: ${meta_cost:.4f}")
        self._last_meta_update_generation = self.completed_generations

    def get_repair_stats(self) -> Dict[str, Any]:
        """Get repair-related statistics."""
        return {
            "total_repair_llm_calls": self.total_repair_llm_calls,
            "total_repair_evals": self.total_repair_evals,
            "total_repair_cost": self.total_repair_cost,
            "total_semantic_repair_llm_calls": self.total_semantic_repair_llm_calls,
            "total_semantic_repair_evals": self.total_semantic_repair_evals,
            "total_semantic_repair_cost": self.total_semantic_repair_cost,
            "total_semantic_repair_successes": self.total_semantic_repair_successes,
            "failure_buffer_stats": self.failure_buffer.get_stats(),
        }

                                                                               
                       
                                                                               

    def _generation_llm_api_calls(self) -> int:
        """Return generation-side raw LLM API calls across all generator clients."""
        calls = int(getattr(self.llm, "total_calls", 0) or 0)
        patch_llm = getattr(self, "patch_llm", None)
        if patch_llm is not None and patch_llm is not self.llm:
            calls += int(getattr(patch_llm, "total_calls", 0) or 0)
        return calls

    def _sync_generation_llm_budget_caps(self) -> None:
        """Dynamically cap generator clients so internal retries can't overshoot `max_llm_calls`.

        Why do this?
        - `LLMClient.query()` increments `total_calls` per retry attempt.
        - With multiple generation-side clients (main + patch override), we must enforce a *global*
          `max_llm_calls` without letting either client independently consume the full budget.
        """
        cap = getattr(self.termination_config, "max_llm_calls", None)
        if cap is None:
            return

        seedbank_debits = int(getattr(self, "seedbank_debited_calls", 0) or 0)
        base_calls = int(getattr(self.llm, "total_calls", 0) or 0)

        patch_llm = getattr(self, "patch_llm", None)
        patch_calls = 0
        if patch_llm is not None and patch_llm is not self.llm:
            patch_calls = int(getattr(patch_llm, "total_calls", 0) or 0)

        used = base_calls + patch_calls + seedbank_debits
        remaining = max(0, int(cap) - int(used))

                                                                                          
        try:
            if hasattr(self.llm, "max_total_calls"):
                self.llm.max_total_calls = int(base_calls) + int(remaining)
        except Exception:
            pass

        if patch_llm is not None and patch_llm is not self.llm:
            try:
                if hasattr(patch_llm, "max_total_calls"):
                    patch_llm.max_total_calls = int(patch_calls) + int(remaining)
            except Exception:
                pass

    def _check_budget_exhausted(self) -> Optional[str]:
        """
        Check whether any budget is exhausted (hard stop).

        Returns the trigger reason if exhausted, otherwise None.

        Budget semantics:
        - max_llm_calls: generation-side raw LLM call count (`self.llm.total_calls`, includes retries)
        - max_evals: evaluator-side "effective eval count" (DB insertions + repair evals)
        """
        tc = self.termination_config

                                                                                                          
        try:
            self._sync_generation_llm_budget_caps()
        except Exception:
            pass

        raw_llm_api_calls = int(self._generation_llm_api_calls())
        budget_calls = raw_llm_api_calls + int(getattr(self, "seedbank_debited_calls", 0) or 0)

                                                                       
        if tc.max_llm_calls is not None:
            if budget_calls >= tc.max_llm_calls:
                return f"max_llm_calls_reached ({budget_calls} >= {tc.max_llm_calls})"

                                                    
        if tc.max_evals is not None:
            total_effective_evals = (
                len(self.db.get_all_programs())
                + int(getattr(self, "total_repair_evals", 0) or 0)
                + int(getattr(self, "total_semantic_repair_evals", 0) or 0)
            )
            if total_effective_evals >= tc.max_evals:
                return f"max_evals_reached ({total_effective_evals} >= {tc.max_evals})"

                                
        if tc.max_time_seconds is not None:
            elapsed = time.time() - self.start_time
            if elapsed >= tc.max_time_seconds:
                return f"max_time_reached ({elapsed:.1f}s >= {tc.max_time_seconds}s)"

        return None

    def _check_exploration_stagnation(self, current_best_tuple: tuple) -> bool:
        """
        Check exploration stagnation (soft stop).

        Returns True if stagnated.
        """
                                                                                      
                                                                   
        if int(self.termination_config.stagnation_generations or 0) <= 0:
            if self.best_fitness_tuple is None or current_best_tuple > self.best_fitness_tuple:
                self.best_fitness_tuple = current_best_tuple
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement += 1
            return False

        if self.best_fitness_tuple is None:
            self.best_fitness_tuple = current_best_tuple
            self.generations_without_improvement = 0
            return False

                                
        if current_best_tuple > self.best_fitness_tuple:
                          
            self.best_fitness_tuple = current_best_tuple
            self.generations_without_improvement = 0
            return False
        else:
                             
            self.generations_without_improvement += 1
            if self.generations_without_improvement >= self.termination_config.stagnation_generations:
                return True
            return False

    def _perform_soft_reset(self):
        """
        Perform a soft reset.

        Current implementation status
        This method currently only records the stagnation event and resets counters. It does
        not yet modify search parameters.

        Intended design (not implemented yet)
        When exploration stagnates, we may "restart" the search by adjusting parameters:
        1) increase temperature → more diversity
        2) increase crossover probability → more combinatorial exploration

        TODO
        A future version can implement actual parameter updates here, e.g.:
        - edit `self.llm.get_kwargs()["temperature"]`
        - edit `self.evo_config.patch_type_probs`
        """
        self.soft_resets_count += 1
        tc = self.termination_config

        logger.info(f"[SoftReset] Performing soft reset #{self.soft_resets_count}")
        logger.info(f"  Temperature boost (config value; not applied yet): +{tc.reset_temperature_boost}")
        logger.info(f"  Crossover boost (config value; not applied yet): +{tc.reset_crossover_boost}")
                                                                    
        try:
            old_alpha = float(getattr(self.db_config, "parent_usage_penalty_alpha", 0.0) or 0.0)
        except Exception:
            old_alpha = 0.0
        new_alpha = min(old_alpha + float(tc.reset_parent_usage_penalty_boost), float(tc.reset_parent_usage_penalty_max))
        setattr(self.db_config, "parent_usage_penalty_alpha", new_alpha)
        logger.info(f"  Parent usage penalty alpha: {old_alpha:.3f} -> {new_alpha:.3f}")

                                   
        self.generations_without_improvement = 0

    def _save_termination_log(self) -> Dict[str, Any]:
        """Save the termination log."""
                                          
                                                                                      
                                                                                
                                                                   
        raw_llm_api_calls = int(getattr(self.llm, "total_calls", 0) or 0)
        seedbank_debited_calls = int(self.seedbank_debited_calls or 0)
        total_budget_calls = raw_llm_api_calls + seedbank_debited_calls
                                                                                              
        total_effective_evals = (
            len(self.db.get_all_programs())
            + int(getattr(self, "total_repair_evals", 0) or 0)
            + int(getattr(self, "total_semantic_repair_evals", 0) or 0)
        )
        no_llm_flag = str(self.problem_config.get("no_llm", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
                                                                                   
        total_embed_cost = 0.0
        total_novelty_checks = 0
        total_novelty_cost = 0.0
        try:
            for p in self.db.get_all_programs():
                meta = p.metadata or {}
                total_embed_cost += float(meta.get("embed_cost", 0.0) or 0.0)
                total_novelty_checks += int(meta.get("novelty_checks_performed", 0) or 0)
                total_novelty_cost += float(meta.get("novelty_cost", 0.0) or 0.0)
        except Exception:
            pass

                                                                                                       
        batchn_expected = None
        batchn_completed = None
        batchn_missing_ranges = None
        if str(getattr(self, "baseline_mode", "") or "").strip() == "batchN":
            try:
                batchn_expected = int(getattr(self.repair_config, "num_init_candidates_gen0", 0) or 0)
                base_dir = Path(self.results_dir) / "baseline_batchN"
                missing: List[int] = []
                completed = 0
                for i in range(max(0, batchn_expected)):
                    m = base_dir / f"sample_{i}" / "results" / "metrics.json"
                    if m.exists() and m.stat().st_size > 0:
                        completed += 1
                    else:
                        missing.append(i)
                batchn_completed = int(completed)
                if missing:
                    ranges: List[List[int]] = []
                    start = prev = missing[0]
                    for x in missing[1:]:
                        if x == prev + 1:
                            prev = x
                            continue
                        ranges.append([int(start), int(prev)])
                        start = prev = x
                    ranges.append([int(start), int(prev)])
                    batchn_missing_ranges = ranges
                else:
                    batchn_missing_ranges = []
            except Exception:
                batchn_expected = None
                batchn_completed = None
                batchn_missing_ranges = None

        log_data = {
            "failure_type": self._classify_failure_type(),
            "trigger_reason": self.termination_reason or "normal_completion",
            "baseline_mode": self.baseline_mode,
            "seed": self.problem_config.get("seed", None),
            "soft_resets_count": self.soft_resets_count,
            "final_best_tuple": list(self.best_fitness_tuple) if self.best_fitness_tuple else None,
            "llm_mode_requested": self.llm_mode_requested,
            "llm_mode_effective": self.llm_mode_effective,
            "llm_unavailable": int(bool(self.llm_unavailable)),
            "llm_unavailable_reason": self.llm_unavailable_reason,
            "no_llm": int(no_llm_flag),
            "openai_llm_base_url": os.environ.get("OPENAI_LLM_BASE_URL", ""),
            "llm_models": list(getattr(self.evo_config, "llm_models", []) or []),
            "parent_selection_strategy": getattr(self.db_config, "parent_selection_strategy", None),
            "cycle_softmax_temperature": getattr(self.db_config, "cycle_softmax_temperature", None),
            "num_archive_inspirations": getattr(self.db_config, "num_archive_inspirations", None),
            "num_top_k_inspirations": getattr(self.db_config, "num_top_k_inspirations", None),
            "patch_types": list(getattr(self.evo_config, "patch_types", []) or []),
            "patch_type_probs": list(getattr(self.evo_config, "patch_type_probs", []) or []),
            "novelty_filter_enabled": int(bool(getattr(self.evo_config, "embedding_model", None))),
            "embedding_model": getattr(self.evo_config, "embedding_model", None),
            "code_embed_sim_threshold": getattr(self.evo_config, "code_embed_sim_threshold", None),
            "openai_embed_base_url": os.environ.get("OPENAI_EMBED_BASE_URL", ""),
            "novelty_llm_base_url": os.environ.get("OPENAI_NOVELTY_LLM_BASE_URL", ""),
                                                                                 
            "total_budget_calls": int(total_budget_calls),
            "total_llm_calls": int(total_budget_calls),
                                                         
            "raw_llm_api_calls": int(raw_llm_api_calls),
            "seedbank_debited_calls": int(seedbank_debited_calls),
            "total_patch_llm_calls": max(
                int(raw_llm_api_calls)
                - int(getattr(self, "total_repair_llm_calls", 0) or 0)
                - int(getattr(self, "total_semantic_repair_llm_calls", 0) or 0),
                0,
            ),
            "total_repair_llm_calls": self.total_repair_llm_calls,
            "total_semantic_repair_llm_calls": self.total_semantic_repair_llm_calls,
            "total_evals": len(self.db.get_all_programs()),
            "total_effective_evals": int(total_effective_evals),
            "total_repair_evals": self.total_repair_evals,
            "total_semantic_repair_evals": self.total_semantic_repair_evals,
            "total_embed_cost": total_embed_cost,
            "total_novelty_checks_performed": total_novelty_checks,
            "total_novelty_cost": total_novelty_cost,
                                                                    
            "batchN_expected_samples": batchn_expected,
            "batchN_completed_samples": batchn_completed,
            "batchN_missing_ranges": batchn_missing_ranges,
            "repair_attempts": sum(r.repair_attempts for r in self.failure_buffer.records),
            "repair_successes": sum(1 for r in self.failure_buffer.records if r.final_status == "repair_success"),
            "semantic_repair_enabled": int(bool(self.semantic_repair_enabled)),
            "semantic_repair_successes": int(self.total_semantic_repair_successes),
            "elapsed_time_seconds": time.time() - self.start_time,
            "generations_completed": self.completed_generations,
        }

        log_path = Path(self.results_dir) / "termination_log.json"
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"[Termination] Log saved to {log_path}")
        return log_data

    def _classify_failure_type(self) -> str:
        """Classify the failure type based on termination reason."""
        if self.termination_reason is None:
            return "normal_completion"
        elif "max_llm_calls" in self.termination_reason or \
             "max_evals" in self.termination_reason or \
             "max_time" in self.termination_reason:
            return "budget_exhausted"
        elif "stagnation" in self.termination_reason:
            return "exploration_stagnation"
        else:
            return "unknown"

    def _finalize_run(self) -> None:
        """Finalize run: write summaries, termination log, and stats (shared by baselines & ours)."""
                                                                                                 
        best_program = self.db.get_best_program()
        self.meta_summarizer.perform_final_summary(str(self.results_dir), best_program)
        self._save_meta_memory()

        self.db.print_summary()
        logger.info(f"Evolution completed! {self.completed_generations} generations")
        logger.info("=" * 80)
        end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Evolution run ended at {end_time}")
        logger.info("=" * 80)

                               
        termination_log = self._save_termination_log()

                      
        repair_stats = self.get_repair_stats()
        logger.info("=" * 60)
        logger.info("Repair Statistics (spec v1.1 constraint C):")
        logger.info(f"  Total repair LLM calls: {repair_stats['total_repair_llm_calls']}")
        logger.info(f"  Total repair evals: {repair_stats['total_repair_evals']}")
        logger.info(f"  Total repair cost: ${repair_stats['total_repair_cost']:.4f}")
        logger.info(f"  Total semantic repair LLM calls: {repair_stats['total_semantic_repair_llm_calls']}")
        logger.info(f"  Total semantic repair evals: {repair_stats['total_semantic_repair_evals']}")
        logger.info(f"  Total semantic repair successes: {repair_stats['total_semantic_repair_successes']}")
        logger.info(f"  Total semantic repair cost: ${repair_stats['total_semantic_repair_cost']:.4f}")
        logger.info(f"  Failure buffer: {repair_stats['failure_buffer_stats']}")
        logger.info("=" * 60)
        logger.info("Termination Log:")
        logger.info(f"  Failure type: {termination_log['failure_type']}")
        logger.info(f"  Trigger reason: {termination_log['trigger_reason']}")
        logger.info(f"  Soft resets: {termination_log['soft_resets_count']}")
        logger.info(f"  Final best tuple: {termination_log['final_best_tuple']}")
        logger.info("=" * 60)

    def _run_baseline_batchN(self) -> None:
        """Baseline: batchN independent initial samples + compile-repair (no archive/parent/inspirations/cross)."""
        num_samples = int(getattr(self.repair_config, "num_init_candidates_gen0", 0) or 0)
        if num_samples <= 0:
            raise ValueError("batchN requires num_init_candidates_gen0 > 0")

        base_dir = Path(self.results_dir) / "baseline_batchN"
        base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[Baseline batchN] Generating {num_samples} independent samples")

        for sample_idx in range(num_samples):
            budget_reason = self._check_budget_exhausted()
            if budget_reason is not None:
                self.termination_reason = budget_reason
                logger.info(f"[Baseline batchN] Budget exhausted: {budget_reason}")
                break

            sample_dir = base_dir / f"sample_{sample_idx}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            exec_fname = str(sample_dir / f"main.{self.lang_ext}")
            results_dir = str(sample_dir / "results")
            Path(results_dir).mkdir(parents=True, exist_ok=True)

                                              
                                                                                                     
                                                                                                
                                         
            metrics_path = Path(results_dir) / "metrics.json"
            if metrics_path.exists() and metrics_path.stat().st_size > 0:
                continue

                                                              
                                                                                                       
                                                                                                    
                                                                                                   
                                       
            before_calls = getattr(self.llm, "total_calls", None)
            api_costs = 0.0
            max_gen_retries = _env_int("AUTOFORMAL_BASELINE_BATCHN_GEN_MAX_RETRIES")
            retry_backoff_s = _env_float("AUTOFORMAL_BASELINE_BATCHN_GEN_RETRY_BACKOFF_S", 0.8)
            retry_backoff_cap_s = _env_float("AUTOFORMAL_BASELINE_BATCHN_GEN_RETRY_BACKOFF_CAP_S", 20.0)
            gen_attempt = 0
            initial_code = None
            patch_name = ""
            patch_description = ""
            while True:
                budget_reason = self._check_budget_exhausted()
                if budget_reason is not None:
                    self.termination_reason = budget_reason
                    logger.info(f"[Baseline batchN] Budget exhausted: {budget_reason}")
                    break
                try:
                    initial_code, patch_name, patch_description, api_costs_one = (
                        self.generate_initial_program()
                    )
                    try:
                        api_costs += float(api_costs_one or 0.0)
                    except Exception:
                        pass
                    break
                except Exception as e:
                    gen_attempt += 1
                    logger.warning(
                        f"[Baseline batchN] Sample {sample_idx} generation failed "
                        f"(attempt {gen_attempt}{'' if max_gen_retries is None else f'/{max_gen_retries}'}): {e}"
                    )
                    if max_gen_retries is not None and gen_attempt >= int(max_gen_retries):
                                                                                    
                        self.termination_reason = (
                            f"baseline_generation_failed (sample={sample_idx}, attempts={gen_attempt})"
                        )
                        self._add_to_failure_buffer(
                            program_id=str(uuid.uuid4()),
                            generation=0,
                            compile_error_type="baseline_generation_failed",
                            compile_error_msg=str(e),
                            statement="",
                            repair_attempts=0,
                            repair_llm_calls_used=0,
                            final_status="patch_failed",
                        )
                        logger.error(f"[Baseline batchN] {self.termination_reason}")
                        break
                                                                             
                    sleep_s = min(float(retry_backoff_cap_s), float(retry_backoff_s) * (2 ** min(gen_attempt - 1, 6)))
                    time.sleep(max(0.0, sleep_s))
                    continue

            if initial_code is None:
                                                                                                                
                break

            after_calls = getattr(self.llm, "total_calls", None)
            gen_calls_used = None
            if isinstance(before_calls, int) and isinstance(after_calls, int):
                gen_calls_used = max(after_calls - before_calls, 0)

            Path(exec_fname).write_text(initial_code, encoding="utf-8")

                              
            results, rtime = self.scheduler.run(exec_fname, results_dir)

            try:
                evaluated_code = Path(exec_fname).read_text(encoding="utf-8")
            except Exception:
                evaluated_code = ""

            metrics_val = results.get("metrics", {}) if results else {}
            compile_ok = int(metrics_val.get("compile_ok", 0) or 0)
            compile_error_type = str(metrics_val.get("compile_error_type", "") or "")
            compile_error_msg = str(metrics_val.get("compile_error_msg", "") or "")
            statement = str(metrics_val.get("statement", "") or "")
            stdout_log = results.get("stdout_log", "") if results else ""
            stderr_log = results.get("stderr_log", "") if results else ""

            combined_score = float(metrics_val.get("combined_score", 0.0) or 0.0)
            public_metrics = metrics_val.get("public", {}) or {}
            private_metrics = metrics_val.get("private", {}) or {}
            text_feedback = metrics_val.get("text_feedback", "") or ""

            pre_program_id = str(uuid.uuid4())
            code_embedding, e_cost = self.get_code_embedding(exec_fname)

            pre_state_role = "pre_post" if compile_ok == 1 else "pre"
            pre_program = Program(
                id=pre_program_id,
                code=evaluated_code,
                language=self.evo_config.language,
                parent_id=None,
                generation=0,
                archive_inspiration_ids=[],
                top_k_inspiration_ids=[],
                code_diff=None,
                embedding=code_embedding,
                correct=(compile_ok == 1),
                combined_score=combined_score,
                public_metrics=public_metrics,
                private_metrics=private_metrics,
                text_feedback=text_feedback,
                metadata={
                    "compute_time": rtime,
                    "api_costs": api_costs,
                    "embed_cost": e_cost,
                    "novelty_cost": 0.0,
                    "patch_type": "baseline_init",
                    "patch_name": patch_name,
                    "patch_description": patch_description,
                    "stdout_log": stdout_log,
                    "stderr_log": stderr_log,
                                                                                     
                "fitness_tuple": list(metrics_val.get("fitness_tuple", (compile_ok, 0, 0, 0.0))),
                    "cycle_log_prob": metrics_val.get("cycle_log_prob", None),
                    "cycle_normalized_log_prob": metrics_val.get("cycle_normalized_log_prob", None),
                    "cycle_score": metrics_val.get("cycle_score", None),
                                                   
                    "baseline_mode": "batchN",
                    "sample_id": int(sample_idx),
                    "step_idx": 0,
                    "prev_step_id": None,
                    "state_role": pre_state_role,
                    "llm_calls_used": gen_calls_used,
                    "compile_error_type": compile_error_type,
                    "compile_error_msg": compile_error_msg,
                },
            )
            self.db.add(pre_program, verbose=True)
            self.db.save()

            if compile_ok == 1:
                                                                                                
                                                                      
                if self.semantic_repair_enabled:
                    try:
                        sem_ok = int(public_metrics.get("semantic_ok", 0) or 0)
                    except Exception:
                        sem_ok = 0
                    if sem_ok == 0:
                        base_job = RunningJob(
                            job_id=f"baseline_batchN_pre_{sample_idx}",
                            exec_fname=exec_fname,
                            results_dir=results_dir,
                            start_time=time.time(),
                            generation=0,
                            parent_id=None,
                            archive_insp_ids=[],
                            top_k_insp_ids=[],
                            code_diff=None,
                            meta_patch_data={
                                "baseline_mode": "batchN",
                                "sample_id": int(sample_idx),
                                "pre_program_id": pre_program_id,
                                "step_idx": 0,
                                "prev_step_id": None,
                                "state_role": pre_state_role,
                            },
                        )
                        self._process_semantic_repair(
                            base_program=pre_program,
                            base_job=base_job,
                            base_metrics=metrics_val,
                            evaluated_code=evaluated_code,
                        )
                continue

                                               
            if not self.repair_config.enabled:
                self._add_to_failure_buffer(
                    program_id=pre_program_id,
                    generation=0,
                    compile_error_type=compile_error_type,
                    compile_error_msg=compile_error_msg,
                    statement=statement,
                    repair_attempts=0,
                    repair_llm_calls_used=0,
                    final_status="no_repair",
                )
                continue

            dummy_job = RunningJob(
                job_id=f"baseline_batchN_{sample_idx}",
                exec_fname=exec_fname,
                results_dir=results_dir,
                start_time=time.time(),
                generation=0,
                parent_id=None,
                archive_insp_ids=[],
                top_k_insp_ids=[],
                code_diff=None,
                meta_patch_data={
                    "baseline_mode": "batchN",
                    "sample_id": int(sample_idx),
                    "pre_program_id": pre_program_id,
                    "step_idx": 1,
                    "prev_step_id": pre_program_id,
                    "state_role": "post",
                },
            )
            repair_item = RepairQueueItem(
                program_id=pre_program_id,
                exec_fname=exec_fname,
                results_dir=results_dir,
                generation=0,
                parent_id=pre_program_id,
                compile_error_type=compile_error_type,
                compile_error_msg=compile_error_msg,
                original_code=str(metrics_val.get("code") or evaluated_code or statement or ""),
                repair_attempts=0,
            )
            repaired_program = self._process_repair(repair_item, dummy_job)

                                                                       
            if repaired_program is not None and self.semantic_repair_enabled:
                try:
                    pm = repaired_program.public_metrics or {}
                    sem_ok = int(pm.get("semantic_ok", 0) or 0)
                    comp_ok = int(pm.get("compile_ok", 0) or 0)
                except Exception:
                    sem_ok = 0
                    comp_ok = 0
                if comp_ok == 1 and sem_ok == 0:
                    base_metrics = _read_metrics_json(results_dir)
                    self._process_semantic_repair(
                        base_program=repaired_program,
                        base_job=dummy_job,
                        base_metrics=base_metrics,
                        evaluated_code=str(repaired_program.code or ""),
                    )

    def _run_baseline_repairloop1(self) -> None:
        """Baseline: single repair trajectory (compile-only correction, no archive/selection/inspirations)."""
        trajectory_id = str(uuid.uuid4())
        base_dir = Path(self.results_dir) / "baseline_repairloop1" / f"trajectory_{trajectory_id}"
        base_dir.mkdir(parents=True, exist_ok=True)

        max_attempts = int(getattr(self.repair_config, "max_repair_attempts_gen0", 0) or 0)
        max_attempts = max(max_attempts, 0)
        logger.info(f"[Baseline repairloop1] trajectory={trajectory_id}, max_attempts={max_attempts}")

                                                   
        step_idx = 0
        step_dir = base_dir / f"step_{step_idx}"
        step_dir.mkdir(parents=True, exist_ok=True)
        exec_fname = str(step_dir / f"main.{self.lang_ext}")
        results_dir = str(step_dir / "results")
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        before_calls = getattr(self.llm, "total_calls", None)
        api_costs = 0.0
        patch_name = "baseline_initial"
        patch_description = "Baseline repairloop1 initial sample."

        if self.evo_config.init_program_path:
            try:
                shutil.copy(self.evo_config.init_program_path, exec_fname)
                patch_description = f"Baseline initial from file: {self.evo_config.init_program_path}"
            except Exception:
                self.evo_config.init_program_path = None

        if not self.evo_config.init_program_path:
            try:
                initial_code, patch_name, patch_description, api_costs = self.generate_initial_program()
                Path(exec_fname).write_text(initial_code, encoding="utf-8")
            except Exception as e:
                logger.warning(f"[Baseline repairloop1] Initial generation failed: {e}")
                self._add_to_failure_buffer(
                    program_id=str(uuid.uuid4()),
                    generation=0,
                    compile_error_type="baseline_generation_failed",
                    compile_error_msg=str(e),
                    statement="",
                    repair_attempts=0,
                    repair_llm_calls_used=0,
                    final_status="patch_failed",
                )
                self.termination_reason = "baseline_generation_failed"
                return

        after_calls = getattr(self.llm, "total_calls", None)
        gen_calls_used = None
        if isinstance(before_calls, int) and isinstance(after_calls, int):
            gen_calls_used = max(after_calls - before_calls, 0)

        results, rtime = self.scheduler.run(exec_fname, results_dir)
        try:
            evaluated_code = Path(exec_fname).read_text(encoding="utf-8")
        except Exception:
            evaluated_code = ""

        metrics_val = results.get("metrics", {}) if results else {}
        compile_ok = int(metrics_val.get("compile_ok", 0) or 0)
        compile_error_type = str(metrics_val.get("compile_error_type", "") or "")
        compile_error_msg = str(metrics_val.get("compile_error_msg", "") or "")
        statement = str(metrics_val.get("statement", "") or "")

        combined_score = float(metrics_val.get("combined_score", 0.0) or 0.0)
        public_metrics = metrics_val.get("public", {}) or {}
        private_metrics = metrics_val.get("private", {}) or {}
        text_feedback = metrics_val.get("text_feedback", "") or ""
        stdout_log = results.get("stdout_log", "") if results else ""
        stderr_log = results.get("stderr_log", "") if results else ""

        code_embedding, e_cost = self.get_code_embedding(exec_fname)
        prev_program_id: Optional[str] = None
        program_id = str(uuid.uuid4())
        prev_program_id = program_id

        pre_state_role = "pre_post" if compile_ok == 1 else "pre"
        self.db.add(
            Program(
                id=program_id,
                code=evaluated_code,
                language=self.evo_config.language,
                parent_id=None,
                generation=0,
                archive_inspiration_ids=[],
                top_k_inspiration_ids=[],
                code_diff=None,
                embedding=code_embedding,
                correct=(compile_ok == 1),
                combined_score=combined_score,
                public_metrics=public_metrics,
                private_metrics=private_metrics,
                text_feedback=text_feedback,
                metadata={
                    "compute_time": rtime,
                    "api_costs": api_costs,
                    "embed_cost": e_cost,
                    "novelty_cost": 0.0,
                    "patch_type": "baseline_init",
                    "patch_name": patch_name,
                    "patch_description": patch_description,
                    "stdout_log": stdout_log,
                    "stderr_log": stderr_log,
                                                                                     
                "fitness_tuple": list(metrics_val.get("fitness_tuple", (compile_ok, 0, 0, 0.0))),
                    "cycle_log_prob": metrics_val.get("cycle_log_prob", None),
                    "cycle_normalized_log_prob": metrics_val.get("cycle_normalized_log_prob", None),
                    "cycle_score": metrics_val.get("cycle_score", None),
                                                   
                    "baseline_mode": "repairloop1",
                    "trajectory_id": trajectory_id,
                    "step_idx": 0,
                    "prev_step_id": None,
                    "state_role": pre_state_role,
                    "llm_calls_used": gen_calls_used,
                    "compile_error_type": compile_error_type,
                    "compile_error_msg": compile_error_msg,
                },
            ),
            verbose=True,
        )
        self.db.save()

        if compile_ok == 1 or not self.repair_config.enabled:
            if compile_ok == 0 and not self.repair_config.enabled:
                self._add_to_failure_buffer(
                    program_id=program_id,
                    generation=0,
                    compile_error_type=compile_error_type,
                    compile_error_msg=compile_error_msg,
                    statement=statement,
                    repair_attempts=0,
                    repair_llm_calls_used=0,
                    final_status="no_repair",
                )
            return

                                                     
        current_code = str(metrics_val.get("code") or normalize_lean_code(evaluated_code) or evaluated_code or "")
        current_error_type = compile_error_type
        current_error_msg = compile_error_msg

        for attempt in range(max_attempts):
            budget_reason = self._check_budget_exhausted()
            if budget_reason is not None:
                self.termination_reason = budget_reason
                logger.info(f"[Baseline repairloop1] Budget exhausted: {budget_reason}")
                break

            step_idx = attempt + 1
            step_dir = base_dir / f"step_{step_idx}"
            step_dir.mkdir(parents=True, exist_ok=True)
            exec_fname = str(step_dir / f"main.{self.lang_ext}")
            results_dir = str(step_dir / "results")
            Path(results_dir).mkdir(parents=True, exist_ok=True)

            sys_msg, user_msg = build_repair_prompt(
                original_code=current_code,
                compile_error_type=current_error_type,
                compile_error_msg=current_error_msg,
                informal=self.problem_config.get("informal", ""),
                reference_header=self.problem_config.get("header", ""),
            )
            llm_kwargs = self.llm.get_kwargs()
            llm_kwargs["temperature"] = self.repair_config.repair_temperature

            repair_llm = self._repair_llm_override()
            before_calls = getattr(self.llm, "total_calls", None)
            response = None
            llm_kwargs_eff = dict(llm_kwargs or {})
            try:
                if repair_llm.get("enabled"):
                    logger.info(
                        f"[Baseline repairloop1] Using repair-only LLM override: "
                        f"model={repair_llm.get('model_name','')}, "
                        f"base_url={repair_llm.get('base_url','')}"
                    )
                    llm_kwargs_eff["model_name"] = str(repair_llm.get("model_name") or "")
                    with _temporary_env({"OPENAI_LLM_BASE_URL": str(repair_llm.get("base_url") or "")}):
                        response = self.llm.query(msg=user_msg, system_msg=sys_msg, llm_kwargs=llm_kwargs_eff)
                else:
                    response = self.llm.query(msg=user_msg, system_msg=sys_msg, llm_kwargs=llm_kwargs_eff)
            except Exception as e:
                logger.warning(
                    f"[Baseline repairloop1] LLM query failed at attempt {attempt + 1}: {type(e).__name__}: {e}"
                )
                response = None
            after_calls = getattr(self.llm, "total_calls", None)
            calls_used = None
            if isinstance(before_calls, int) and isinstance(after_calls, int):
                calls_used = max(after_calls - before_calls, 0)

                                                       
            self.total_repair_llm_calls += (calls_used if calls_used is not None and calls_used > 0 else 1)
            resp_cost = float(getattr(response, "cost", 0.0) or 0.0)
            self.total_repair_cost += resp_cost

            repaired_code = extract_between(
                (getattr(response, "content", "") or ""),
                f"```{self.evo_config.language}",
                "```",
                False,
            )
            INVALID_EXTRACT_RESULTS = {None, "none", "null", "nil", ""}
            if repaired_code in INVALID_EXTRACT_RESULTS or (
                isinstance(repaired_code, str)
                and repaired_code.strip().lower() in {"none", "null", "nil", ""}
            ):
                repaired_code = None
            if not repaired_code:
                model_name = (
                    str(getattr(response, "model_name", "") or "").strip()
                    or str(repair_llm.get("model_name") or "").strip()
                    or (self.evo_config.llm_models[0] if self.evo_config.llm_models else "")
                )
                adapter = get_model_adapter(model_name)
                if adapter:
                    parsed_stmt = adapter.parse_output((getattr(response, "content", "") or ""))
                    if parsed_stmt:
                        repaired_code = parsed_stmt

            repaired_code_norm = normalize_lean_code(repaired_code or "")
            if not repaired_code_norm:
                continue

            repaired_program = self._build_repaired_program(repaired_code_norm)
            Path(exec_fname).write_text(repaired_program, encoding="utf-8")

                                                                   
            self.total_repair_evals += 1
            results, rtime = self.scheduler.run(exec_fname, results_dir)

            metrics_val = results.get("metrics", {}) if results else {}
            compile_ok = int(metrics_val.get("compile_ok", 0) or 0)
            current_error_type = str(metrics_val.get("compile_error_type", "") or "")
            current_error_msg = str(metrics_val.get("compile_error_msg", "") or "")
            current_code = str(metrics_val.get("code") or repaired_code_norm or "")

            combined_score = float(metrics_val.get("combined_score", 0.0) or 0.0)
            public_metrics = metrics_val.get("public", {}) or {}
            private_metrics = metrics_val.get("private", {}) or {}
            text_feedback = metrics_val.get("text_feedback", "") or ""
            stdout_log = results.get("stdout_log", "") if results else ""
            stderr_log = results.get("stderr_log", "") if results else ""

            code_embedding, e_cost = self.get_code_embedding(exec_fname)
            new_program_id = str(uuid.uuid4())

            state_role = "post" if compile_ok == 1 else "intermediate"
            self.db.add(
                Program(
                    id=new_program_id,
                    code=repaired_program,
                    language=self.evo_config.language,
                    parent_id=prev_program_id,
                    generation=0,
                    archive_inspiration_ids=[],
                    top_k_inspiration_ids=[],
                    code_diff=None,
                    embedding=code_embedding,
                    correct=(compile_ok == 1),
                    combined_score=combined_score,
                    public_metrics=public_metrics,
                    private_metrics=private_metrics,
                    text_feedback=text_feedback,
                    metadata={
                        "compute_time": rtime,
                        "embed_cost": e_cost,
                        "repair_attempt": step_idx,
                        "repair_cost": resp_cost,
                        "stdout_log": stdout_log,
                        "stderr_log": stderr_log,
                                                                                             
                        "fitness_tuple": list(metrics_val.get("fitness_tuple", (compile_ok, 0, 0, 0.0))),
                        "cycle_log_prob": metrics_val.get("cycle_log_prob", None),
                        "cycle_normalized_log_prob": metrics_val.get("cycle_normalized_log_prob", None),
                        "cycle_score": metrics_val.get("cycle_score", None),
                                                       
                        "baseline_mode": "repairloop1",
                        "trajectory_id": trajectory_id,
                        "step_idx": step_idx,
                        "prev_step_id": prev_program_id,
                        "state_role": state_role,
                        "llm_calls_used": calls_used,
                        "compile_error_type": current_error_type,
                        "compile_error_msg": current_error_msg,
                        "repaired": True,
                    },
                ),
                verbose=True,
            )
            self.db.save()
            prev_program_id = new_program_id

            if compile_ok == 1:
                self._add_to_failure_buffer(
                    program_id=program_id,
                    generation=0,
                    compile_error_type=compile_error_type,
                    compile_error_msg=compile_error_msg,
                    statement=statement,
                    repair_attempts=step_idx,
                    repair_llm_calls_used=0,
                    final_status="repair_success",
                )
                return

                   
        self._add_to_failure_buffer(
            program_id=program_id,
            generation=0,
            compile_error_type=current_error_type,
            compile_error_msg=current_error_msg,
            statement=normalize_lean_statement(current_code) or statement,
            repair_attempts=max_attempts,
            repair_llm_calls_used=0,
            final_status="repair_exhausted",
        )

    def run(self):
        """
        Run evolution with repair + budget/stagnation checks + termination logging.

        Compared to the parent `run()`, this adds:
        - hard-stop: _check_budget_exhausted
        - soft-stop: _check_exploration_stagnation → _perform_soft_reset
        - record termination reason to termination_log
        """
        mode_norm = str(self.baseline_mode or "ours").strip().lower()
        if mode_norm == "batchn":
            self._run_baseline_batchN()
            self.best_fitness_tuple = self._get_current_best_tuple()
            self._finalize_run()
            return
        if mode_norm == "repairloop1":
            self._run_baseline_repairloop1()
            self.best_fitness_tuple = self._get_current_best_tuple()
            self._finalize_run()
            return

        max_jobs = self.evo_config.max_parallel_jobs
        target_gens = self.evo_config.num_generations
        logger.info(
            f"Starting evolution with {max_jobs} parallel jobs, "
            f"target: {target_gens} generations (with budget/stagnation checks)"
        )

                                                           
        if self.completed_generations == 0 and target_gens > 0:
            logger.info("Running generation 0 sequentially to initialize database...")
            self._run_generation_0()
            if len(self.db.get_all_programs()) == 0:
                if self.termination_reason is None:
                    self.termination_reason = "gen0_no_compile_ok_seed"
                logger.info(
                    f"[Termination] Gen0 produced no compile_ok=1 seeds, stopping: {self.termination_reason}"
                )
                self._finalize_run()
                return
            self.completed_generations = 1
            self.next_generation_to_submit = 1
            logger.info(f"Completed generation 0, total: 1/{target_gens}")

                                                    
        if self.completed_generations < target_gens:
            logger.info("Starting parallel execution for remaining generations...")

            while (
                self.completed_generations < target_gens or len(self.running_jobs) > 0
            ):
                                          
                completed_jobs = self._check_completed_jobs()

                                            
                if completed_jobs:
                    for job in completed_jobs:
                        self._process_completed_job(job)

                                                        
                    self._update_completed_generations()

                    if self.verbose:
                        logger.info(
                            f"Processed {len(completed_jobs)} jobs. "
                            f"Total completed generations: "
                            f"{self.completed_generations}/{target_gens}"
                        )

                                                                                     
                    current_best_tuple = self._get_current_best_tuple()
                    if current_best_tuple is not None:
                        if self._check_exploration_stagnation(current_best_tuple):
                            self._perform_soft_reset()
                            if (
                                self.soft_resets_count
                                >= self.termination_config.max_soft_resets
                            ):
                                self.termination_reason = (
                                    "exploration_stagnation_max_soft_resets"
                                )
                                logger.info(
                                    "[Termination] Soft reset limit reached, stopping..."
                                )
                                break

                                              
                budget_reason = self._check_budget_exhausted()
                if budget_reason is not None:
                    self.termination_reason = budget_reason
                    logger.info(f"[Termination] Budget exhausted: {budget_reason}")
                    break

                                                             
                if self.completed_generations >= target_gens:
                    logger.info("All generations completed, exiting...")
                    break

                                                         
                if (
                    len(self.running_jobs) < max_jobs
                    and self.next_generation_to_submit < target_gens
                ):
                    self._submit_new_job()

                time.sleep(2)
        self._finalize_run()

    def _get_current_best_tuple(self) -> Optional[tuple]:
        """
        Get current best fitness_tuple from the DB (compile_ok=1 programs only).

        Returns None if no such program exists.
        """
        best_tuple = None
        programs = self.db.get_all_programs()
        for p in programs:
            meta = p.metadata or {}
            ft = meta.get("fitness_tuple")
            if ft is None:
                continue
            try:
                ft_tuple = tuple(ft)
            except Exception:
                continue
            if best_tuple is None or ft_tuple > best_tuple:
                best_tuple = ft_tuple
        return best_tuple
