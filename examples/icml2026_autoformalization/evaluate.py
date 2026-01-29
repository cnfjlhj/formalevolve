"""
Evaluator for Lean4 Autoformalization.

================================================================================
                               Design Overview
================================================================================

This file is the evaluator used by the autoformalization system. It scores a Lean 4 candidate
statement/file and returns metrics that act as the evolutionary fitness signal.

Core metrics (ordered by importance)
1. compile_ok: compiles successfully (hard gate)
2. beq_ok: formal equivalence via BEq+ (if enabled; compared to ground truth)
3. semantic_ok: semantic agreement via LLM-as-a-judge (optional; can be noisy)
4. cycle_score: cycle-consistency soft signal (optional; typically weaker than semantic)
5. potential: reserved field (currently fixed to 0)

Why use "gated" scoring?
A naive weighted sum can rank a non-compiling candidate above a compiling one if the soft
signals are high. That violates the basic principle: invalid programs should not be competitive.
We therefore gate all soft signals behind compile_ok.

Spec constraints
- [D] Compile checks must run on the full file `{header} + {body}`.
- [F] Candidates with compile_ok=0 must always rank below compile_ok=1.

================================================================================
                              Spec Version: v1.1
================================================================================
"""

                                                                               
                          
                                                                               
import argparse
import asyncio
import importlib.util                                                 
import json
import math
import os
import re
import sys
import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests                                   

                                                                            
                                       
 
               
                                                                                 
                                            
                              
_LOCAL_CYCLE_CC_SRC = Path(__file__).parent / "autoformalization-cycle-consistency" / "src"
_ENV_CYCLE_CC_SRC = Path(os.environ.get("CYCLE_CC_SRC_PATH", "")) if os.environ.get("CYCLE_CC_SRC_PATH") else None
_CYCLE_CC_SRC = (
    _LOCAL_CYCLE_CC_SRC if _LOCAL_CYCLE_CC_SRC.exists()
    else _ENV_CYCLE_CC_SRC if _ENV_CYCLE_CC_SRC and _ENV_CYCLE_CC_SRC.exists()
    else None
)

if _CYCLE_CC_SRC and _CYCLE_CC_SRC.exists():
    sys.path.insert(0, str(_CYCLE_CC_SRC))
try:
    from model_interface import OpenAICompatibleLLM                
except Exception:
    OpenAICompatibleLLM = None                
                                                                               
                                                     
                                                                               
                                                                                         
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

                                                                              
                                                                                          
PROJECT_CACHE_DIR = Path(__file__).parent / ".lean_interact_cache"
TMP_CACHE_DIR = Path("/tmp/lean_interact_cache_autoformal")
MATHLIB_CACHE = Path(os.environ.get("MATHLIB_CACHE_DIR", "~/.cache/mathlib")).expanduser()

def _pick_writable_cache_dir() -> Path:
    candidates: List[Path] = []
    env_cache = os.environ.get("LEAN_INTERACT_CACHE_DIR", "").strip()
    if env_cache:
        candidates.append(Path(env_cache))
    if MATHLIB_CACHE.exists():
        candidates.append(MATHLIB_CACHE)
    candidates.append(PROJECT_CACHE_DIR)
    candidates.append(TMP_CACHE_DIR)

    for p in candidates:
        try:
            p.mkdir(parents=True, exist_ok=True)
            test_file = p / ".write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            return p
        except Exception:
            continue

    PROJECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return PROJECT_CACHE_DIR

cache_dir = _pick_writable_cache_dir()
os.environ["LEAN_INTERACT_CACHE_DIR"] = str(cache_dir)

                                            
try:
    import lean_interact.utils as li_utils
    import lean_interact.project as li_project
    import lean_interact.config as li_config

    li_utils.DEFAULT_CACHE_DIR = cache_dir
    li_project.DEFAULT_CACHE_DIR = cache_dir
    li_config.DEFAULT_CACHE_DIR = cache_dir
except Exception:
    pass

                                                                                
                                                                                       
                      
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
                                                                         
                                                                         
BEQ_PLUS_PATH = os.environ.get("BEQ_PLUS_PATH", "").strip()
if BEQ_PLUS_PATH:
    sys.path.insert(0, BEQ_PLUS_PATH)

                                                                               
                      
                                                                               
                                                                   
                                                                                                       
from shinka.core import run_shinka_eval
from shinka.core.wrap_eval import save_json_results

                                                                               
                           
                                                                               
                                                
                                                           
                                                         
from autoformalization.lean_env import get_lean_server, is_lean_server_available
from autoformalization.models import Candidate, Problem
from autoformalization.critic_wrapper import critic_eval, close_session


                                                                               
                
                                                                               

_CONFIG_CONTEXT_RESULTS_DIR: Optional[str] = None


def _set_config_context_results_dir(results_dir: str) -> None:
    """Set per-process config resolution context for validate_fn.

    `run_shinka_eval` calls validate_fn before aggregate_metrics_fn, so we
    cannot pass results_dir through the validate_fn signature. We instead set
    a module-level context in main().
    """
    global _CONFIG_CONTEXT_RESULTS_DIR
    _CONFIG_CONTEXT_RESULTS_DIR = results_dir


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _apply_semantic_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply global semantic kill-switches to evaluator config.

    - `AUTOFORMAL_DISABLE_SEMANTIC=1` forces `use_semantic=False` (no CriticLean calls).
    """
    if _env_truthy("AUTOFORMAL_DISABLE_SEMANTIC"):
        cfg = dict(cfg)
        cfg["use_semantic"] = False
    return cfg


def _find_problem_config_path(results_dir: Optional[str] = None) -> Optional[Path]:
    """Find the nearest `problem_config.json` for this run.

    Search order:
    1) `AUTOFORMAL_PROBLEM_CONFIG_PATH` env var (explicit override)
    2) Walk up from `results_dir` (or context results_dir) to filesystem root
    3) Legacy: `examples/autoformalization_v1/problem_config.json`
    """
    explicit = os.environ.get("AUTOFORMAL_PROBLEM_CONFIG_PATH")
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p

    anchor = results_dir or _CONFIG_CONTEXT_RESULTS_DIR
    if anchor:
        start = Path(anchor)
        if start.suffix:             
            start = start.parent
        for parent in (start, *start.parents):
            cand = parent / "problem_config.json"
            if cand.exists():
                return cand

    legacy = Path(__file__).parent / "problem_config.json"
    return legacy if legacy.exists() else None


def get_config(results_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Load evaluator config.

    Config sources (in priority order):
    1. `problem_config.json` (created by `run_evo.py` at run start)
    2. Environment variables (fallback)

    Key fields:
    - informal: natural-language statement to formalize
    - header: Lean 4 header (imports/opens/options)
    - ground_truth: optional reference statement (for BEq+)
    - use_beq: enable BEq+ equivalence checking
    - compile_timeout: compile timeout (seconds)
    """
    config_path = _find_problem_config_path(results_dir)

                                                                           
    if config_path and config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        return _apply_semantic_overrides({
            "informal": config.get("informal", ""),
            "header": config.get("header", "import Mathlib"),
            "ground_truth": config.get("ground_truth", ""),
            "use_beq": config.get("use_beq", False),
            "no_llm": bool(config.get("no_llm", False)),
                                                                
            "use_semantic": config.get("use_semantic", False),
                                                                                         
            "use_cycle_consistency": config.get("use_cycle_consistency", False),
            "compile_timeout": config.get("compile_timeout", 600),
                                                                                               
            "score_base": config.get(
                "score_base",
                float(os.environ.get("AUTOFORMAL_SCORE_BASE", "100")),
            ),
            "score_cycle_weight": config.get(
                "score_cycle_weight",
                float(os.environ.get("AUTOFORMAL_SCORE_CYCLE_WEIGHT", "50")),
            ),
            "score_semantic_bonus": config.get(
                "score_semantic_bonus",
                float(os.environ.get("AUTOFORMAL_SCORE_SEMANTIC_BONUS", "100")),
            ),
            "score_beq_bonus": config.get(
                "score_beq_bonus",
                float(os.environ.get("AUTOFORMAL_SCORE_BEQ_BONUS", "200")),
            ),
                                                                                
            "cycle_api_base_url": config.get(
                "cycle_api_base_url",
                os.environ.get("CYCLE_API_BASE_URL", "http://127.0.0.1:8090/v1"),
            ),
            "cycle_model_name": config.get(
                "cycle_model_name",
                os.environ.get("CYCLE_MODEL_NAME", "Qwen2.5-32B-Instruct"),
            ),
            "informalize_prompt_template": config.get(
                "informalize_prompt_template",
                os.environ.get(
                    "INFORMALIZE_PROMPT_TEMPLATE",
                    "Informalize: {formal_statement}",
                ),
            ),
                                                                               
            "cycle_normalize_decl_name": bool(
                config.get(
                    "cycle_normalize_decl_name",
                    os.environ.get("AUTOFORMAL_CYCLE_NORMALIZE_DECL_NAME", "true").lower()
                    in {"1", "true", "yes", "y", "on"},
                )
            ),
            "cycle_normalized_decl_name": str(
                config.get(
                    "cycle_normalized_decl_name",
                    os.environ.get("AUTOFORMAL_CYCLE_NORMALIZED_DECL_NAME", _CYCLE_CANONICAL_DECL_NAME),
                )
                or _CYCLE_CANONICAL_DECL_NAME
            ),
            "semantic_normalize_decl_name": bool(
                config.get(
                    "semantic_normalize_decl_name",
                    os.environ.get("AUTOFORMAL_SEMANTIC_NORMALIZE_DECL_NAME", "true").lower()
                    in {"1", "true", "yes", "y", "on"},
                )
            ),
            "semantic_normalized_decl_name": str(
                config.get(
                    "semantic_normalized_decl_name",
                    os.environ.get("AUTOFORMAL_SEMANTIC_NORMALIZED_DECL_NAME", _CYCLE_CANONICAL_DECL_NAME),
                )
                or _CYCLE_CANONICAL_DECL_NAME
            ),
            "beq_normalize_decl_name": bool(
                config.get(
                    "beq_normalize_decl_name",
                    os.environ.get("AUTOFORMAL_BEQ_NORMALIZE_DECL_NAME", "true").lower()
                    in {"1", "true", "yes", "y", "on"},
                )
            ),
            "beq_candidate_decl_name": str(
                config.get(
                    "beq_candidate_decl_name",
                    os.environ.get("AUTOFORMAL_BEQ_CANDIDATE_DECL_NAME", "my_cand"),
                )
                or "my_cand"
            ),
            "beq_ground_truth_decl_name": str(
                config.get(
                    "beq_ground_truth_decl_name",
                    os.environ.get("AUTOFORMAL_BEQ_GROUND_TRUTH_DECL_NAME", "my_gt"),
                )
                or "my_gt"
            ),
            "cycle_softmax_temperature": float(
                config.get(
                    "cycle_softmax_temperature",
                    os.environ.get("SOFTMAX_TEMPERATURE", "3.5"),
                )
            ),
            "cycle_fallback_normalized_log_prob": float(
                config.get(
                    "cycle_fallback_normalized_log_prob",
                    os.environ.get("CYCLE_FALLBACK_NORMALIZED_LOG_PROB", "-100.0"),
                )
            ),
            "lean_server_url": config.get(
                "lean_server_url",
                os.environ.get("LEAN_SERVER_URL", "http://127.0.0.1:8001/api/check"),
            ),
        })

                                   
    return _apply_semantic_overrides({
        "informal": os.environ.get("AUTOFORMAL_INFORMAL", ""),
        "header": os.environ.get(
            "AUTOFORMAL_HEADER",
            "import Mathlib\nopen Function Fintype Subgroup Ideal Polynomial"
        ),
        "ground_truth": os.environ.get("AUTOFORMAL_GT", ""),
        "use_beq": os.environ.get("AUTOFORMAL_USE_BEQ", "false").lower() == "true",
        "no_llm": os.environ.get("AUTOFORMAL_NO_LLM", "").lower() in {"1", "true", "yes"},
        "use_semantic": os.environ.get("AUTOFORMAL_USE_SEMANTIC", "false").lower() == "true",
        "use_cycle_consistency": os.environ.get("AUTOFORMAL_USE_CYCLE", "false").lower() == "true",
        "compile_timeout": int(os.environ.get("AUTOFORMAL_COMPILE_TIMEOUT", "600")),
        "score_base": float(os.environ.get("AUTOFORMAL_SCORE_BASE", "100")),
        "score_cycle_weight": float(os.environ.get("AUTOFORMAL_SCORE_CYCLE_WEIGHT", "50")),
        "score_semantic_bonus": float(os.environ.get("AUTOFORMAL_SCORE_SEMANTIC_BONUS", "100")),
        "score_beq_bonus": float(os.environ.get("AUTOFORMAL_SCORE_BEQ_BONUS", "200")),
        "cycle_api_base_url": os.environ.get("CYCLE_API_BASE_URL", "http://127.0.0.1:8090/v1"),
        "cycle_model_name": os.environ.get("CYCLE_MODEL_NAME", "Qwen2.5-32B-Instruct"),
        "informalize_prompt_template": os.environ.get(
            "INFORMALIZE_PROMPT_TEMPLATE", "Informalize: {formal_statement}"
        ),
        "cycle_normalize_decl_name": os.environ.get(
            "AUTOFORMAL_CYCLE_NORMALIZE_DECL_NAME", "true"
        ).lower()
        in {"1", "true", "yes", "y", "on"},
        "cycle_normalized_decl_name": os.environ.get(
            "AUTOFORMAL_CYCLE_NORMALIZED_DECL_NAME", _CYCLE_CANONICAL_DECL_NAME
        ),
        "semantic_normalize_decl_name": os.environ.get(
            "AUTOFORMAL_SEMANTIC_NORMALIZE_DECL_NAME", "true"
        ).lower()
        in {"1", "true", "yes", "y", "on"},
        "semantic_normalized_decl_name": os.environ.get(
            "AUTOFORMAL_SEMANTIC_NORMALIZED_DECL_NAME", _CYCLE_CANONICAL_DECL_NAME
        ),
        "beq_normalize_decl_name": os.environ.get(
            "AUTOFORMAL_BEQ_NORMALIZE_DECL_NAME", "true"
        ).lower()
        in {"1", "true", "yes", "y", "on"},
        "beq_candidate_decl_name": os.environ.get(
            "AUTOFORMAL_BEQ_CANDIDATE_DECL_NAME", "my_cand"
        ),
        "beq_ground_truth_decl_name": os.environ.get(
            "AUTOFORMAL_BEQ_GROUND_TRUTH_DECL_NAME", "my_gt"
        ),
        "cycle_softmax_temperature": float(os.environ.get("SOFTMAX_TEMPERATURE", "3.5")),
        "cycle_fallback_normalized_log_prob": float(
            os.environ.get("CYCLE_FALLBACK_NORMALIZED_LOG_PROB", "-100.0")
        ),
        "lean_server_url": os.environ.get(
            "LEAN_SERVER_URL", "http://127.0.0.1:8001/api/check"
        ),
    })
def cycle_consistency_score(
    target_informal: str,
    formal_statement: str,
    cfg: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """
    Continuous score derived from cycle-consistency log-probability.

    Paper uses softmax over candidate log-probs (temperature ~3.5). In Shinka's
    per-candidate evaluation, we use an *unnormalized* softmax weight:
      score = exp(normalized_log_prob / T)
    which stays in (0, 1] and preserves ordering.
    """
    meta: Dict[str, Any] = {}
    t = float(cfg.get("cycle_softmax_temperature", 3.5))
    t = max(t, 1e-6)
    fallback_nlp = float(cfg.get("cycle_fallback_normalized_log_prob", -100.0))

    def _stub(method: str, err: str) -> Tuple[float, Dict[str, Any]]:
        meta["cycle_method"] = method
        meta["cycle_error"] = err
        meta["cycle_log_prob"] = None
        meta["cycle_num_tokens"] = 0
        meta["cycle_normalized_log_prob"] = float(fallback_nlp)
                                    
        score = math.exp(float(fallback_nlp) / t) if math.isfinite(float(fallback_nlp)) else 0.0
        meta["cycle_score_raw"] = float(score)
        return float(score), meta

    a = (target_informal or "").strip()
                                                                           
    if cfg.get("no_llm", False) or os.environ.get("AUTOFORMAL_NO_LLM", "").lower() in {"1", "true", "yes"}:
        meta["cycle_prompt_preview"] = ""
        return _stub("stub_no_llm", "no_llm=1")

    if not a:
        meta["cycle_prompt_preview"] = ""
        return _stub("stub_empty_informal", "Empty informal statement")

    formal_for_prompt = formal_statement.strip()
    normalize_decl_name = bool(cfg.get("cycle_normalize_decl_name", True))
    if normalize_decl_name:
        normalized_name = str(
            cfg.get("cycle_normalized_decl_name", _CYCLE_CANONICAL_DECL_NAME)
            or _CYCLE_CANONICAL_DECL_NAME
        )
        formal_for_prompt, original_name = normalize_decl_name_for_cycle_prompt(
            formal_for_prompt, normalized_name=normalized_name
        )
        meta["cycle_normalize_decl_name"] = True
        meta["cycle_normalized_decl_name"] = normalized_name
        meta["cycle_original_decl_name"] = original_name
    else:
        meta["cycle_normalize_decl_name"] = False

                                                                          
    formal_for_prompt = strip_theorem_keyword_for_cycle_prompt(formal_for_prompt)
    meta["cycle_strip_theorem_keyword"] = True
    meta["cycle_strip_decl_name"] = True

    prompt = cfg["informalize_prompt_template"].format(
        formal_statement=formal_for_prompt
    )
    meta["cycle_prompt_preview"] = prompt[:200]

    try:
        if OpenAICompatibleLLM is None:
            return _stub(
                "stub_cycle_missing_impl",
                "autoformalization-cycle-consistency not importable",
            )
        lm = OpenAICompatibleLLM(
            model=cfg["cycle_model_name"],
            base_url=cfg["cycle_api_base_url"],
            api_key=os.environ.get("CYCLE_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY")),
        )
        res = lm.compute_log_prob(prompt=prompt, completion=a, system_prompt=None)
        meta["cycle_log_prob"] = float(res.log_prob)
        meta["cycle_num_tokens"] = int(res.num_tokens)
        meta["cycle_normalized_log_prob"] = float(res.normalized_log_prob)
        meta["cycle_method"] = "openai_compatible_llm"
    except Exception as e:
        return _stub(
            "stub_cycle_exception",
            f"cycle logprob exception: {type(e).__name__}: {e}",
        )

    nlp = float(meta.get("cycle_normalized_log_prob", float("-inf")))
    if not math.isfinite(nlp):
        return _stub("stub_cycle_non_finite", "cycle_normalized_log_prob not finite")

                                
    score = math.exp(nlp / t)
    meta["cycle_score_raw"] = score
    return float(score), meta


                                                                               
                                                    
                                                                               

def extract_compile_error_type(error_msg: str) -> str:
    """
    Extract a coarse error type from a Lean compile error message.

    Design goal
    When compilation fails and triggers repair, we want to tell the LLM what kind of
    error it is, so the repair prompt can be more targeted.

    Error categories
    - type_mismatch: type mismatch (often wrong argument types)
    - unknown_identifier: unknown identifier/constant (typo or missing import/open)
    - unbound_variable: unbound variable / not in scope (scoping issue)
    - syntax_error: syntax error (e.g., missing `:=`)
    - typeclass_error: typeclass synthesis failure
    - ambiguous_identifier: ambiguous identifier (needs more type annotations)
    - invalid_syntax: invalid syntax
    - timeout: compile timeout
    - import_error: import error
    - other: uncategorized

    Why categorize?
    Different errors call for different repair strategies:
    - type_mismatch → revisit type annotations / arguments
    - unknown_identifier → check spelling or add imports/opens
    - syntax_error → fix syntax structure
    """
    if not error_msg:
        return "unknown"

    error_lower = error_msg.lower()

                                                            
    if "type mismatch" in error_lower:
        return "type_mismatch"
    elif "unknown identifier" in error_lower or "unknown constant" in error_lower:
        return "unknown_identifier"
    elif "unbound" in error_lower or "not in scope" in error_lower:
        return "unbound_variable"
    elif "expected" in error_lower and "got" in error_lower:
        return "syntax_error"
    elif "failed to synthesize" in error_lower:
        return "typeclass_error"
    elif "ambiguous" in error_lower:
        return "ambiguous_identifier"
    elif "invalid" in error_lower:
        return "invalid_syntax"
    elif "timeout" in error_lower:
        return "timeout"
    elif "import" in error_lower:
        return "import_error"
    else:
        return "other"


def truncate_error_msg(error_msg: str, max_len: int = 500) -> str:
    """
    Truncate error messages to keep prompts short.

    Lean error messages can be very long (e.g., full type inference traces), but for repair,
    the first ~500 characters usually contain the crucial signal. Extremely long errors waste
    tokens and can degrade LLM understanding.
    """
    if not error_msg:
        return ""
    if len(error_msg) <= max_len:
        return error_msg
    return error_msg[:max_len] + "... [truncated]"


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

    Semantic-eval preference:
    For semantic_ok (CriticLean), we want to judge consistency on a *theorem/lemma*.
    In practice, models may emit helper `def` / `abbrev` declarations before the final theorem.
    If we always take "the first declaration", we might accidentally judge a helper `def` as
    the formal statement, which leads to systematic semantic_ok errors.
    We therefore prefer the first theorem/lemma; if none exists, we fall back to the first
    declaration of any kind.
    """
    if not text:
        return ""
    lines = text.splitlines()
    keyword_pattern_any = re.compile(
        rf"^\s*(?:noncomputable\s+)?(?:{'|'.join(LEAN_DECL_KEYWORDS)})\b"
    )
    keyword_pattern_thm = re.compile(r"^\s*(?:noncomputable\s+)?(?:theorem|lemma)\b")
    decl_idx = None
                                      
    for i, line in enumerate(lines):
        if keyword_pattern_thm.search(line):
            decl_idx = i
            break
                                                        
    if decl_idx is None:
        for i, line in enumerate(lines):
            if keyword_pattern_any.search(line):
                decl_idx = i
                break
    if decl_idx is None:
        return ""                                                          
    start = decl_idx
    while start > 0 and lines[start - 1].lstrip().startswith("@["):
        start -= 1
    next_idx = None
    for j in range(decl_idx + 1, len(lines)):
        if keyword_pattern_any.search(lines[j]):
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
    """Normalize a model-produced Lean snippet into a compile-ready Lean file.

    This keeps imports/opens/options and removes only:
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


def extract_lean_preamble(code: str) -> str:
    """Extract the preamble (imports/opens/options) before the first top-level declaration.

    Note: we intentionally drop standalone attribute lines (e.g. `@[simp]`) that
    syntactically belong to the following declaration, to avoid producing a
    header that ends with a dangling attribute (which can break downstream tools
    like BEq+ that expect a pure header).
    """
    s = (code or "").strip()
    if not s:
        return ""

    keyword_pattern = re.compile(
        rf"^\s*(?:noncomputable\s+)?(?:{'|'.join(LEAN_DECL_KEYWORDS)})\b"
    )
    preamble_lines: List[str] = []
    for line in s.splitlines():
        if keyword_pattern.search(line):
            break
        if line.lstrip().startswith("@["):
            continue
        preamble_lines.append(line)
    return "\n".join(preamble_lines).strip()

                                                                               
                                        
                                                                               
 
                                                                                     
                                                                               
                                                                     
 
                                                                       
                                                                              
         
 
            
                                                                          
                                                                       
 

_CYCLE_CANONICAL_DECL_NAME = "my_theorem"


def normalize_decl_name_for_cycle_prompt(
    formal_statement: str,
    normalized_name: str = _CYCLE_CANONICAL_DECL_NAME,
) -> Tuple[str, Optional[str]]:
    """Normalize the first top-level Lean declaration name.

    Example:
        `theorem my_874887 (n : ℕ) : P n := by sorry`
        → `theorem my_theorem (n : ℕ) : P n := by sorry`

    Returns:
        (normalized_statement, original_name_if_replaced)
    """
    stmt = (formal_statement or "").strip()
    if not stmt:
        return stmt, None

                                                       
     
                                                                    
                                                                          
    name_pat = re.compile(
        r"(?m)^(\s*(?:noncomputable\s+)?(?:theorem|lemma|def|instance|axiom|abbrev|opaque)\s+)([^\s:(]+)"
    )
    m = name_pat.search(stmt)
    if not m:
        return stmt, None

    original = m.group(2)
    if original == normalized_name:
        return stmt, None

    normalized = name_pat.sub(rf"\1{normalized_name}", stmt, count=1)
    return normalized, original


def strip_theorem_keyword_for_cycle_prompt(formal_statement: str) -> str:
    """Strip `theorem` and its declaration name for cycle-consistency prompting.

    Requirement:
    - When constructing the cycle-consistency prompt, remove:
      1) the leading `theorem` keyword (and optional leading `noncomputable`)
      2) the declaration name immediately following it (e.g. `my_theorem`)
    - Keep the rest of the declaration (binders, type, `:= by sorry`, etc.).
    """
    s = (formal_statement or "").strip()
    if not s:
        return ""

                                                                                   
    s = re.sub(r"(?m)^\s*@\[.*\]\s*\n?", "", s).strip()

                                                             
    s2 = re.sub(r"(?m)^\s*(?:noncomputable\s+)?theorem\s+", "", s, count=1).strip()

                                        
              
                                              
                                   
    m_name = re.match(r"^\s*([^\s:(]+)\s*(.*)\Z", s2, flags=re.DOTALL)
    if not m_name:
        return s2.strip()
    return (m_name.group(2) or "").strip()


                                                                               
                        
                                                                               

def check_lean_compile(
    code: str,
    timeout: int = 600,
    lean_server_url: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Check whether Lean code compiles successfully.

    Notes:
    - Candidates are expected to output a *complete Lean file* (imports + theorem).
    - The compile check runs on the full code directly (no external header+body concatenation).

    Returns:
    - (True, "") on success
    - (False, error_msg) on failure
    """
                                                                             
                                                                          
     
            
                                                                                                
                                                                              
    raw_server_url = (lean_server_url or os.environ.get("LEAN_SERVER_URL", "")).strip()
    use_http = raw_server_url.startswith("http://") or raw_server_url.startswith("https://")
    LEAN_SERVER_URL = raw_server_url if use_http else ""

    full_code = normalize_lean_code(code)
    if not full_code:
        return False, "Empty Lean code"

                                                                                      
    if not any(kw in full_code for kw in LEAN_DECL_KEYWORDS):
        return False, "Missing Lean declaration keyword"

    def _fallback_lean_interact() -> Tuple[bool, str]:
        """Fallback compile check via lean_interact (no HTTP server required)."""
        try:
            from autoformalization.lean_env import get_lean_server
            from lean_interact import Command
            from lean_interact.interface import CommandResponse, LeanError
        except Exception as e:
            return False, f"[LeanInteractError] {type(e).__name__}: {e}"

        try:
            server = get_lean_server()
            res = server.run(Command(cmd=full_code), timeout=timeout)
        except Exception as e:
            return False, f"[LeanInteractRunError] {type(e).__name__}: {e}"

        if isinstance(res, LeanError):
            return False, str(res)
        if isinstance(res, CommandResponse):
            if res.lean_code_is_valid():
                return True, ""
            errors = [m.data for m in res.messages if m.severity == "error"]
            return False, "\n".join(errors) if errors else "Unknown Lean error"

        return False, f"Unknown LeanInteract response type: {type(res).__name__}"

    if not use_http:
        return _fallback_lean_interact()

    try:
        snippet_id = f"compile-check-{uuid.uuid4().hex}"
        response = requests.post(
            LEAN_SERVER_URL,
            json={
                "snippets": [{"id": snippet_id, "code": full_code}],
                "timeout": timeout,
            },
            timeout=timeout + 30,                                                  
        )
        response.raise_for_status()
        result = response.json()

                       
        if "results" not in result or len(result["results"]) == 0:
            return False, "No results from Lean server"

        check_result = result["results"][0]
        messages = check_result.get("response", {}).get("messages", [])

                                         
        errors = [m for m in messages if m.get("severity") == "error"]

        if errors:
                                     
            error_msgs = [e.get("data", "Unknown error") for e in errors]
            return False, "\n".join(error_msgs)

                                       
        return True, ""

    except requests.exceptions.Timeout:
        return _fallback_lean_interact()
    except requests.exceptions.ConnectionError:
        return _fallback_lean_interact()
    except requests.exceptions.RequestException as e:
        return False, f"[RequestError] {type(e).__name__}: {e}"
    except Exception as e:
        return False, f"[Exception] {type(e).__name__}: {e}"


                                                                               
                           
                                                                               

async def check_critic_lean(informal: str, formal: str) -> Tuple[int, str]:
    """
    Use CriticLean to check semantic consistency.

    Motivation:
    Compiling successfully only guarantees syntactic correctness, not mathematical correctness.
    CriticLean is an LLM-based semantic judge that:
    1) interprets the informal statement
    2) interprets the formal statement
    3) decides whether they match

    Why only call it when compile_ok=1?
    1) If compilation fails, the statement may be incomplete and semantic checking is meaningless
    2) Saves API cost
    3) Matches the lexicographic / gated design

    Returns:
    - (1, reason) if semantically consistent
    - (0, reason) if inconsistent or if the check fails
    """
    if not informal:
        return 0, "[Error] No informal statement provided"

    try:
        score, reason = await critic_eval(informal, formal)
        return score, reason
    except Exception as e:
        return 0, f"[Exception] {type(e).__name__}: {e}"


def extract_critic_accuracy_confirmation(reasons: str) -> str:
    """Extract the '5. Accuracy Confirmation' section from CriticLean reasons."""
    s = (reasons or "").strip()
    if not s:
        return ""

    m = re.search(
        r"(?is)\b5\s*\.?\s*Accuracy\s*Confirmation\s*:?\s*(.*)\Z",
        s,
    )
    if m:
        return m.group(1).strip()

                                                              
    m2 = re.search(r"(?is)\b5\s*\.?\s*Accuracy\s*Confirmation\b", s)
    if not m2:
        return ""
    return s[m2.end() :].lstrip(" :\n\t").strip()


                                                                               
                                   
                                                                               

def check_beq_plus(statement: str, ground_truth: str, header: str, timeout: int = 600) -> int:
    """
    Use BEq+ to check formal equivalence against the ground truth.

    Motivation:
    BEq+ is a high-precision, low-recall equivalence checker:
    - High precision: if it returns 1, the two statements are very likely equivalent.
    - Low recall: even if statements are equivalent, it may still return 0.

    Why optional?
    1) Requires a ground truth reference (many tasks do not have one)
    2) Higher computational cost
    3) Low recall means it can miss correct answers

    Returns:
    - 1 for "equivalent"
    - 0 for "not proven equivalent" / failed check
    """
    if not ground_truth:
        return 0

    try:
        from beq_plus import beq_plus
    except ImportError:
        return 0

    server = get_lean_server()
    cand = statement.strip()
    gt = ground_truth.strip()
    hdr = header.strip()

    if not cand or not gt:
        return 0

    try:
        ok = beq_plus(cand, gt, hdr, server, timeout_per_proof=timeout, verbose=False)
        return 1 if ok else 0
    except Exception as e:
        print(f"[BEq+] Error: {e}")
        return 0


                                                                               
                  
                                                                               

def load_code_from_program(program_path: str) -> str:
    """
    Load Lean code from a candidate program file.

    Behavior:
    - For `.lean` files: read the full file (imports + theorem), stripping EVOLVE markers.
    - For Python candidate files: execute `run_formalization()` or `generate_statement()`.
    """
    if program_path.endswith(".lean"):
        raw = Path(program_path).read_text(encoding="utf-8")
        return normalize_lean_code(raw) or raw.strip()

    spec = importlib.util.spec_from_file_location("candidate", program_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {program_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "run_formalization"):
        return module.run_formalization()
    if hasattr(module, "generate_statement"):
        return module.generate_statement()
    raise RuntimeError("Module must have run_formalization() or generate_statement()")


_COMPILE_CACHE: Dict[Tuple[str, str, int], Tuple[bool, str]] = {}


def _compile_cached(
    code: str,
    timeout: int = 600,
    lean_server_url: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Cached wrapper for Lean compilation.

    Shinka's `run_shinka_eval` runs `validate_fn` before `aggregate_metrics_fn`.
    For autoformalization we want `correct == compile_ok`, so we compile in
    `validate_formalization` and reuse the result in `aggregate_metrics`.
    """
    code_norm = normalize_lean_code(code)
    key = ((lean_server_url or "").strip(), code_norm, int(timeout))
    if key in _COMPILE_CACHE:
        return _COMPILE_CACHE[key]
    res = check_lean_compile(
        code=code_norm,
        timeout=timeout,
        lean_server_url=lean_server_url,
    )
    _COMPILE_CACHE[key] = res
    return res


def validate_formalization(run_output: str, results_dir: Optional[str] = None, atol: float = 1e-6) -> Tuple[bool, Optional[str]]:
    """
    Validate the run output for Shinka's `correct` flag.

    Autoformalization semantics:
    - `correct` MUST be equivalent to `compile_ok`.
    - This removes heuristic correctness checks based on string patterns.
    """
    if not run_output or not isinstance(run_output, str):
        return False, "Empty or invalid output"

    config = get_config(results_dir=results_dir)
    lean_server_url = config.get("lean_server_url")
    timeout = int(config.get("compile_timeout", 600))
    ok, err = _compile_cached(
        code=run_output,
        timeout=timeout,
        lean_server_url=lean_server_url,
    )
    return (ok, None if ok else err)


def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:
    """Provide experiment kwargs for a run."""
    config = get_config()
    return {
        "informal": config["informal"],
        "header": config["header"],
        "ground_truth": config["ground_truth"],
        "use_beq": config["use_beq"],
    }


                                                                               
                              
                                                                               

def compute_gated_score(
    compile_ok: int,
    cycle_score: float,
    semantic_ok: int,
    beq_ok: int,
    potential: float,
    *,
    score_base: float = 100.0,
    score_cycle_weight: float = 50.0,
    score_semantic_bonus: float = 100.0,
    score_beq_bonus: float = 200.0,
) -> float:
    """
    Compute the gated fitness score.

    ================================================================================
                       This is the "heart" of the evaluator
    ================================================================================

    Constraint F
    compile_ok=0 candidates must always rank below compile_ok=1 candidates.

    Formula (aligned with SPEC.md; additive, and ensures beq > semantic > cycle):
    combined_score = compile_ok * (
        score_base
        + score_cycle_weight * cycle_score
        + score_semantic_bonus * semantic_ok
        + score_beq_bonus * beq_ok
    )

    Why does this satisfy Constraint F?
    - If compile_ok = 0, the product is 0.
    - If compile_ok = 1, the minimum score is score_base (default 100).
    - Therefore any compile_ok=1 candidate (>=100) beats any compile_ok=0 candidate (=0).

    Intuition (default weights)
    - compile_ok=0: score = 0
    - compile_ok=1, cycle=1.0, semantic=0, beq=0: score = 150
    - compile_ok=1, cycle=0.0, semantic=1, beq=0: score = 200
    - compile_ok=1, cycle=1.0, semantic=1, beq=0: score = 250
    - compile_ok=1, cycle=0.0, semantic=0, beq=1: score = 300
    - compile_ok=1, cycle=1.0, semantic=1, beq=1: score = 450

    Weight design
    - score_base=100: base score for compile_ok=1
    - score_cycle_weight * cycle_score: continuous signal (cycle_score∈[0,1])
    - score_semantic_bonus * semantic_ok: discrete bonus (semantic_ok∈{0,1})
    - score_beq_bonus * beq_ok: discrete bonus (beq_ok∈{0,1})

    Constraint: to preserve ordering semantics (beq > semantic > cycle), defaults should satisfy:
    - score_semantic_bonus > score_cycle_weight
    - score_beq_bonus > score_semantic_bonus + score_cycle_weight
    """
                                                                                          
    return float(
        int(compile_ok)
        * (
            float(score_base)
            + float(score_cycle_weight) * float(cycle_score)
            + float(score_semantic_bonus) * int(semantic_ok)
            + float(score_beq_bonus) * int(beq_ok)
        )
    )


                                                                               
                                    
                                                                               

def aggregate_metrics(
    results: List[str],
    results_dir: str,
) -> Dict[str, Any]:
    """
    Aggregate evaluation results and return a full `metrics` dict.

    ================================================================================
                     This is the evaluator "main function"
    ================================================================================

    Design: lexicographic gating
    Evaluation happens in strict priority order:
    1) Check compile_ok (most important)
    2) Only if compile_ok=1, compute soft signals (cycle-consistency; optional Critic semantics)
    3) Only if compile_ok=1, check beq_ok

    "Lexicographic" means we compare the first dimension first; only ties compare the next, etc.

    Spec constraints
    - [D] Compile checks run on the full file (see check_lean_compile)
    - [F] If compile_ok=0, combined_score is always 0 (see compute_gated_score)

    Returned metrics schema (subset)
    {
        "compile_ok": 0 or 1,
        "compile_error_type": error type (only when compile_ok=0),
        "compile_error_msg": error message (only when compile_ok=0),
        "semantic_ok": 0 or 1,
        "critic_raw": raw CriticLean output,
        "beq_ok": 0 or 1,
        "potential": 0.0 (fixed in v1),
        "fitness_tuple": (compile_ok, beq_ok, semantic_ok, cycle_score),
        "combined_score": gated score,
        "statement": extracted Lean statement,
    }
    """
                                         
    if not results:
        return {
            "compile_ok": 0,
            "compile_error_type": "no_results",
            "compile_error_msg": "No results",
            "semantic_ok": 0,
            "beq_ok": 0,
            "potential": 0.0,
            "fitness_tuple": (0, 0, 0, 0.0),
            "combined_score": 0.0,
            "error": "No results",
        }

    raw_output = results[0]
    lean_code = normalize_lean_code(raw_output)
    if not lean_code:
        lean_code = str(raw_output or "").strip()

                                                                                
    statement = normalize_lean_statement(lean_code)

    config = get_config()
    lean_server_url = config.get("lean_server_url")               

                                                                               
                                
                                                                               
                                                              
    compile_ok_bool, compile_error = _compile_cached(
        code=lean_code,
        timeout=config["compile_timeout"],
        lean_server_url=lean_server_url,
    )
    compile_ok = 1 if compile_ok_bool else 0

                                                       
    compile_error_type = extract_compile_error_type(compile_error) if compile_error else ""
    compile_error_msg = truncate_error_msg(compile_error)

                                                                   
    potential = 0.0

                              
    metrics = {
                                 
        "compile_ok": compile_ok,
        "compile_error_type": compile_error_type,
        "compile_error_msg": compile_error_msg,
        "semantic_ok": 0,                              
        "critic_raw": "",
        "beq_ok": 0,
        "potential": potential,
        "cycle_score": 0.0,
                        
        "fitness_tuple": (compile_ok, 0, 0, 0.0),
        "combined_score": 0.0,
                                            
        "code": lean_code,
                                                                  
        "statement": statement,
    }

                                                                               
                                                          
                                                                               
                                                              
                                                               
    if compile_ok == 0:
        metrics["combined_score"] = 0.0
        metrics["fitness_tuple"] = (0, 0, 0, 0.0)
                                                                           
        metrics["public"] = {
            "compile_ok": 0,
            "cycle_score": 0.0,
            "semantic_ok": 0,
            "beq_ok": 0,
        }
        metrics["private"] = {
            "compile_error_type": compile_error_type,
            "compile_error_msg": compile_error_msg,
        }
        return metrics

                                                                               
                                                                                            
                                                                               
                                                                                         
    semantic_ok = 0
    cycle_score = 0.0

    use_semantic = bool(config.get("use_semantic", False))
    use_cycle = bool(config.get("use_cycle_consistency", False))

                                                                             
                                                                            
                                                                                
                                               
    formal_for_semantic = lean_code
    if use_semantic:
        metrics["semantic_input_kind"] = "lean_code_full_file"
        semantic_normalize = bool(config.get("semantic_normalize_decl_name", False))
        metrics["semantic_normalize_decl_name"] = semantic_normalize
        if semantic_normalize:
            normalized_name = str(
                config.get("semantic_normalized_decl_name", _CYCLE_CANONICAL_DECL_NAME)
                or _CYCLE_CANONICAL_DECL_NAME
            )
            formal_for_semantic, original_name = normalize_decl_name_for_cycle_prompt(
                formal_for_semantic, normalized_name=normalized_name
            )
            metrics["semantic_normalized_decl_name"] = normalized_name
            if original_name:
                metrics["semantic_original_decl_name"] = original_name

    cycle_meta = {}
    if use_semantic and use_cycle:
        async def _run_post_compile():
            sem_task = asyncio.create_task(
                check_critic_lean(config.get("informal", ""), formal_for_semantic)
            )
            cycle_task = asyncio.to_thread(
                cycle_consistency_score,
                target_informal=config.get("informal", ""),
                formal_statement=statement,
                cfg=config,
            )
            return await asyncio.gather(sem_task, cycle_task, return_exceptions=True)

        sem_res, cycle_res = asyncio.run(_run_post_compile())
        if isinstance(sem_res, Exception):
            metrics["critic_raw"] = f"[Error] {type(sem_res).__name__}: {sem_res}"
            semantic_ok = 0
        else:
            sem_score, critic_raw = sem_res
            semantic_ok = 1 if sem_score else 0
            metrics["semantic_ok"] = semantic_ok
            metrics["critic_raw"] = critic_raw
            metrics["critic_accuracy_confirmation"] = extract_critic_accuracy_confirmation(critic_raw)
            try:
                raw = str(critic_raw or "")
                sha = hashlib.sha256(raw.encode("utf-8")).hexdigest() if raw else ""
                metrics["critic_raw_sha256"] = sha
                out_dir = Path(results_dir or _CONFIG_CONTEXT_RESULTS_DIR or ".")
                out_path = out_dir / "critic_lean_reason.txt"
                out_path.write_text(raw, encoding="utf-8")
                metrics["critic_raw_path"] = str(out_path.name)
            except Exception as e:
                metrics["critic_raw_write_error"] = f"{type(e).__name__}: {e}"

        if isinstance(cycle_res, Exception):
            cycle_score = 0.0
            cycle_meta = {"cycle_method": "stub_cycle_exception", "cycle_error": f"{type(cycle_res).__name__}: {cycle_res}"}
        else:
            cycle_score, cycle_meta = cycle_res
    else:
                        
        if use_semantic:
            try:
                sem_score, critic_raw = asyncio.run(
                    check_critic_lean(config.get("informal", ""), formal_for_semantic)
                )
                semantic_ok = 1 if sem_score else 0
                metrics["semantic_ok"] = semantic_ok
                metrics["critic_raw"] = critic_raw
                metrics["critic_accuracy_confirmation"] = extract_critic_accuracy_confirmation(critic_raw)
                try:
                    raw = str(critic_raw or "")
                    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest() if raw else ""
                    metrics["critic_raw_sha256"] = sha
                    out_dir = Path(results_dir or _CONFIG_CONTEXT_RESULTS_DIR or ".")
                    out_path = out_dir / "critic_lean_reason.txt"
                    out_path.write_text(raw, encoding="utf-8")
                    metrics["critic_raw_path"] = str(out_path.name)
                except Exception as e:
                    metrics["critic_raw_write_error"] = f"{type(e).__name__}: {e}"
            except Exception as e:
                metrics["critic_raw"] = f"[Error] {e}"

                                  
        if use_cycle:
            cycle_score, cycle_meta = cycle_consistency_score(
                target_informal=config.get("informal", ""),
                formal_statement=statement,
                cfg=config,
            )

    if use_cycle:
        metrics.update(cycle_meta)
    metrics["cycle_score"] = float(cycle_score)

                                                                               
                                                                       
                                                                               
    beq_ok = 0
    if config["use_beq"] and config["ground_truth"]:
        cand_stmt = statement
        gt_stmt = config["ground_truth"]
        beq_header = extract_lean_preamble(lean_code) or str(config.get("header", "") or "")
        if bool(config.get("beq_normalize_decl_name", True)):
            cand_name = str(config.get("beq_candidate_decl_name", "my_cand") or "my_cand")
            gt_name = str(config.get("beq_ground_truth_decl_name", "my_gt") or "my_gt")
            cand_stmt, cand_orig = normalize_decl_name_for_cycle_prompt(
                cand_stmt, normalized_name=cand_name
            )
            gt_stmt, gt_orig = normalize_decl_name_for_cycle_prompt(
                gt_stmt, normalized_name=gt_name
            )
            metrics["beq_normalize_decl_name"] = True
            metrics["beq_candidate_decl_name"] = cand_name
            metrics["beq_ground_truth_decl_name"] = gt_name
            metrics["beq_candidate_original_decl_name"] = cand_orig
            metrics["beq_ground_truth_original_decl_name"] = gt_orig
        else:
            metrics["beq_normalize_decl_name"] = False
        beq_ok = check_beq_plus(
            cand_stmt,
            gt_stmt,
            beq_header,
            config["compile_timeout"],
        )
        metrics["beq_ok"] = beq_ok

                                                                       

                                                                               
                                                                             
                                                                               
                                                                                              
                                                     
    metrics["fitness_tuple"] = (compile_ok, int(beq_ok), int(semantic_ok), float(cycle_score))

    def _cfg_float(key: str, default: float) -> float:
        """Read a float from config, treating explicit 0 as a valid value."""
        if key not in config:
            return float(default)
        v = config.get(key)
        if v is None:
            return float(default)
        try:
            return float(v)
        except Exception:
            return float(default)

    metrics["combined_score"] = compute_gated_score(
        compile_ok=compile_ok,
        cycle_score=float(cycle_score),
        semantic_ok=int(semantic_ok),
        beq_ok=int(beq_ok),
        potential=potential,
        score_base=_cfg_float("score_base", 100.0),
        score_cycle_weight=_cfg_float("score_cycle_weight", 50.0),
        score_semantic_bonus=_cfg_float("score_semantic_bonus", 100.0),
        score_beq_bonus=_cfg_float("score_beq_bonus", 200.0),
    )

                                                                               
                                                   
                                                                               
                                                                                           
                                                                  
    metrics["public"] = {
        "compile_ok": compile_ok,
        "cycle_score": round(cycle_score, 4),
        "semantic_ok": int(semantic_ok),
        "beq_ok": beq_ok,
    }
                                                        
    metrics["private"] = {
        "compile_error_type": compile_error_type,
        "compile_error_msg": compile_error_msg,
        "cycle_normalized_log_prob": metrics.get("cycle_normalized_log_prob"),
    }

                                                                               
                                                            
                                                                               
    try:
                                                       
        metrics_for_json = metrics.copy()
        metrics_for_json["fitness_tuple"] = list(metrics["fitness_tuple"])
        with open(Path(results_dir) / "eval_details.json", "w") as f:
            json.dump(metrics_for_json, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Evaluator] Failed to save details: {e}")

    return metrics


                                                                               
                            
                                                                               

def main(program_path: str, results_dir: str):
    """
    Evaluator entrypoint.

    Called by the engine to evaluate a single candidate program.

    Call chain
    run_evo.py → AutoformalizationRunner → scheduler → evaluate.py:main()

    Execution flow (diagram)
    ┌──────────────────┐
    │ Receive args      │  program_path: candidate program file
    │ (program_path,    │  results_dir: output directory
    │  results_dir)    │
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │ run_shinka_eval  │  evaluation harness
    │   ├─ load program │
    │   ├─ execute      │
    │   └─ aggregate    │
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │ aggregate_metrics│  our core evaluation logic
    │   ├─ Lean compile │  check_lean_compile()
    │   ├─ semantics    │  check_critic_lean()
    │   ├─ BEq+         │  check_beq_plus()
    │   └─ gated score  │  compute_gated_score()
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │ Return metrics    │  includes fitness_tuple, combined_score, etc.
    └──────────────────┘
    """
    print(f"[Evaluator] Program: {program_path}")
    print(f"[Evaluator] Results dir: {results_dir}")

    os.makedirs(results_dir, exist_ok=True)
    _set_config_context_results_dir(results_dir)

    config = get_config(results_dir=results_dir)
    print(f"[Evaluator] Informal: {config['informal'][:100]}...")
    print(f"[Evaluator] Use BEq+: {config['use_beq']}")

    def _aggregator(r: List[str]) -> Dict[str, Any]:
        return aggregate_metrics(r, results_dir)

    def _validator(run_output: str, atol: float = 1e-6) -> Tuple[bool, Optional[str]]:
        return validate_formalization(run_output, results_dir=results_dir, atol=atol)

    if program_path.endswith(".lean"):
        code = load_code_from_program(program_path)
        metrics = aggregate_metrics([code], results_dir)
        correct, error_msg = validate_formalization(code, results_dir=results_dir)
        save_json_results(results_dir, metrics, correct, error_msg)
    else:
                                                   
        metrics, correct, error_msg = run_shinka_eval(
            program_path=program_path,
            results_dir=results_dir,
            experiment_fn_name="run_formalization",
            num_runs=1,
            get_experiment_kwargs=get_experiment_kwargs,
            validate_fn=_validator,
            aggregate_metrics_fn=_aggregator,
        )

                                       
    print("\n[Evaluator] Results (spec v1.1):")
    print(f"  compile_ok: {metrics.get('compile_ok', 0)}")
    if metrics.get('compile_ok', 0) == 0:
        print(f"  compile_error_type: {metrics.get('compile_error_type', 'unknown')}")
    print(f"  semantic_ok: {metrics.get('semantic_ok', 0)}")
    print(f"  beq_ok: {metrics.get('beq_ok', 0)}")
    print(f"  potential: {metrics.get('potential', 0.0)}")
    print(f"  fitness_tuple: {metrics.get('fitness_tuple', (0, 0, 0, 0.0))}")
    print(f"  combined_score: {metrics.get('combined_score', 0.0)}")
    print(f"  correct: {correct}")

    if error_msg:
        print(f"  error: {error_msg[:200]}")


                                                                               
                
                                                                               

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoformalization evaluator")
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.lean",
        help="Path to program to evaluate",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
