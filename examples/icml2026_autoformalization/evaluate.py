"""
Evaluator for Lean4 Autoformalization using ShinkaEvolve.

================================================================================
                              设计理念概述
================================================================================

本文件是 autoformalization 系统的"评估器"，负责判断一个 Lean4 statement 的质量。

【核心设计思想】
进化搜索需要一个"适应度函数"来评判候选解的好坏。在 autoformalization 任务中，
我们的候选解是 Lean4 定理声明，适应度由以下因素决定：

1. compile_ok (编译通过) - 最重要，是"硬门槛"
2. beq_ok (形式等价) - 最高标准（若启用；与 ground truth 对比）
3. semantic_ok (语义一致) - 次高标准（LLM-as-a-judge，可能误判）
4. cycle_score (cycle-consistency) - 连续 soft signal（比 semantic 更弱，用于提供细粒度选择压力）
5. potential (潜力分) - 预留接口，当前固定为 0

【为什么用"门控"设计？】
传统做法是把所有指标加权求和，但这会导致一个问题：
- 一个编译失败但"语义很像"的候选，可能比编译成功但语义差的候选分数更高
- 这违反了进化搜索的基本原则：无效程序不应该有竞争力

所以我们用"门控"设计：
- compile_ok = 0 时，整个分数直接为 0，不管其他指标多好
- 这保证了进化压力始终朝着"先编译通过"的方向

【约束满足】
- [D] 编译检查必须在 {header} + {body} 拼接后的完整文件上执行
- [F] compile_ok=0 的候选在排序上永远劣于 compile_ok=1 的候选

================================================================================
                              规约版本: v1.1
================================================================================
"""

# =============================================================================
# 标准库导入
# =============================================================================
import argparse
import asyncio
import importlib.util   # 用于动态加载候选程序模块
import json
import math
import os
import re
import sys
import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests  # HTTP client for Lean Server API

# Reuse the paper-aligned cycle-consistency implementation directly to avoid
# duplicating logic across directories.
#
# Search order:
# 1. Local: examples/autoformalization_v1/autoformalization-cycle-consistency/src
# 2. Environment variable: CYCLE_CC_SRC_PATH
# 3. None (gracefully degrade)
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
    from model_interface import OpenAICompatibleLLM  # type: ignore
except Exception:
    OpenAICompatibleLLM = None  # type: ignore
# =============================================================================
# 环境初始化（必须在其他导入之前）
# =============================================================================
# 必须最先加载 .env，因为后续的 API 调用依赖环境变量中的密钥
# 如：OPENAI_API_KEY、ANTHROPIC_API_KEY 等
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# 设置 Lean 缓存目录（lean_interact 会写入 lock / tmp project）。
# 在 sandbox 环境下，用户家目录可能不可写，因此必须选择一个可写路径。
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

# 覆盖 lean_interact 默认 cache 路径
try:
    import lean_interact.utils as li_utils
    import lean_interact.project as li_project
    import lean_interact.config as li_config

    li_utils.DEFAULT_CACHE_DIR = cache_dir
    li_project.DEFAULT_CACHE_DIR = cache_dir
    li_config.DEFAULT_CACHE_DIR = cache_dir
except Exception:
    pass

# 添加项目路径，使得可以 import 其他模块
# 【为什么需要手动添加路径？】
# 因为 evaluate.py 是作为子进程独立运行的，它的工作目录可能不是项目根目录，
# 所以需要显式添加路径才能找到 ShinkaEvolve 和 autoformalization 模块。
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Optional: allow local BEq+ checkout without hard-coding personal paths.
# If unset, BEq+ is simply disabled unless installed as a Python package.
BEQ_PLUS_PATH = os.environ.get("BEQ_PLUS_PATH", "").strip()
if BEQ_PLUS_PATH:
    sys.path.insert(0, BEQ_PLUS_PATH)

# =============================================================================
# ShinkaEvolve 框架导入
# =============================================================================
# run_shinka_eval: ShinkaEvolve 提供的标准评估入口
# 它会：1) 加载候选程序 2) 执行 3) 收集结果 4) 调用 aggregate_metrics
from shinka.core import run_shinka_eval
from shinka.core.wrap_eval import save_json_results

# =============================================================================
# Autoformalization 模块导入
# =============================================================================
# 这些是我们复用的已有模块，提供 Lean4 相关功能：
# - lean_env: Lean4 编译服务器（使用 lean-interact 库）
# - critic_wrapper: CriticLean 语义检查（LLM-based）
from autoformalization.lean_env import get_lean_server, is_lean_server_available
from autoformalization.models import Candidate, Problem
from autoformalization.critic_wrapper import critic_eval, close_session


# =============================================================================
# 配置加载
# =============================================================================

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
        if start.suffix:  # file path
            start = start.parent
        for parent in (start, *start.parents):
            cand = parent / "problem_config.json"
            if cand.exists():
                return cand

    legacy = Path(__file__).parent / "problem_config.json"
    return legacy if legacy.exists() else None


def get_config(results_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    加载评估配置。

    配置来源（按优先级）：
    1. problem_config.json 文件（由 run_evo.py 在启动时创建）
    2. 环境变量（备用）

    配置项说明：
    - informal: 自然语言数学陈述，需要被形式化
    - header: Lean4 的 import 和 open 语句，作为上下文
    - ground_truth: 标准答案（可选，用于 BEq+ 检查）
    - use_beq: 是否启用 BEq+ 等价性检查
    - compile_timeout: 编译超时时间（秒）
    """
    config_path = _find_problem_config_path(results_dir)

    # 优先从配置文件加载（支持 per-results_dir 隔离）
    if config_path and config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        return _apply_semantic_overrides({
            "informal": config.get("informal", ""),
            "header": config.get("header", "import Mathlib"),
            "ground_truth": config.get("ground_truth", ""),
            "use_beq": config.get("use_beq", False),
            "no_llm": bool(config.get("no_llm", False)),
            # Semantic check is noisy and optional; default off.
            "use_semantic": config.get("use_semantic", False),
            # Cycle-consistency score (continuous) as an alternative signal; default off.
            "use_cycle_consistency": config.get("use_cycle_consistency", False),
            "compile_timeout": config.get("compile_timeout", 600),
            # Scoring weights (must be auditable via problem_config.json; explicit 0 is valid).
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
            # Paper-aligned cycle-consistency config (log-prob via /completions)
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
            # Reduce metric variance from semantically irrelevant naming noise.
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

    # 备用：从环境变量加载
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
        # exp(negative / T) in (0,1]
        score = math.exp(float(fallback_nlp) / t) if math.isfinite(float(fallback_nlp)) else 0.0
        meta["cycle_score_raw"] = float(score)
        return float(score), meta

    a = (target_informal or "").strip()
    # Global no-LLM mode: never touch network; return a deterministic stub.
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

    # Strip the leading `theorem` keyword as required by the new protocol.
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

    # exp(negative / T) in (0,1]
    score = math.exp(nlp / t)
    meta["cycle_score_raw"] = score
    return float(score), meta


# =============================================================================
# 错误类型提取（用于 repair prompt）
# =============================================================================

def extract_compile_error_type(error_msg: str) -> str:
    """
    从 Lean 编译错误信息中提取错误类型。

    【设计目的】
    当编译失败触发 repair 时，我们需要告诉 LLM 是什么类型的错误，
    这样 LLM 可以更有针对性地修复。

    【错误分类】
    - type_mismatch: 类型不匹配，通常是参数类型错误
    - unknown_identifier: 未知标识符，可能是拼写错误或缺少 import
    - unbound_variable: 未绑定变量，通常是作用域问题
    - syntax_error: 语法错误，如缺少 :=
    - typeclass_error: 类型类实例合成失败
    - ambiguous_identifier: 标识符歧义，需要更明确的类型注解
    - invalid_syntax: 无效语法
    - timeout: 编译超时
    - import_error: 导入错误
    - other: 其他未分类错误

    【为什么要分类？】
    不同类型的错误需要不同的修复策略：
    - type_mismatch → 检查类型注解
    - unknown_identifier → 检查拼写或添加 import
    - syntax_error → 检查语法结构
    """
    if not error_msg:
        return "unknown"

    error_lower = error_msg.lower()

    # 按照常见程度排序的 Lean4 错误模式
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
    截断错误信息，避免 prompt 过长。

    【为什么要截断？】
    Lean 的错误信息可能非常长（包含完整的类型推导过程），
    但对于 repair 来说，前 500 字符通常已经包含了关键信息。
    过长的错误信息会浪费 token，影响 LLM 的理解。
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
    移除 Lean 注释（单行和多行）。
    """
    text = re.sub(r"/-(.|\n)*?-/\s*", "", text)
    text = "\n".join([line.split("--")[0].rstrip() for line in text.splitlines()])
    return text


def strip_header_lines(text: str) -> str:
    """移除 import/open/open scoped 行，避免 header 重复。"""
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
    """移除 EVOLVE-BLOCK 标记行。"""
    cleaned = []
    for line in text.splitlines():
        if re.match(r"^\s*(?:#|//|--)?\s*EVOLVE-BLOCK-(?:START|END)\s*$", line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def extract_first_declaration(text: str) -> str:
    """提取第一条 Lean 顶层声明。

    【修复】当找不到声明时返回空字符串而非原始文本，
    避免 'none' 等无效输出被当作有效 statement 流转。

    【语义评估偏好】
    对 semantic_ok（CriticLean）而言，我们希望对 *theorem/lemma* 做一致性判断。
    实际模型输出里可能会先定义一些辅助 `def` / `abbrev`，再给出 `theorem`。
    这时若直接取“第一条声明”，会把 `def` 误当作 formal statement，导致
    semantic_ok 发生系统性误判。
    因此这里优先提取第一条 theorem/lemma；若不存在，再回退到第一条任意声明。
    """
    if not text:
        return ""
    lines = text.splitlines()
    keyword_pattern_any = re.compile(
        rf"^\s*(?:noncomputable\s+)?(?:{'|'.join(LEAN_DECL_KEYWORDS)})\b"
    )
    keyword_pattern_thm = re.compile(r"^\s*(?:noncomputable\s+)?(?:theorem|lemma)\b")
    decl_idx = None
    # Prefer theorem/lemma if present.
    for i, line in enumerate(lines):
        if keyword_pattern_thm.search(line):
            decl_idx = i
            break
    # Fallback: first top-level declaration of any kind.
    if decl_idx is None:
        for i, line in enumerate(lines):
            if keyword_pattern_any.search(line):
                decl_idx = i
                break
    if decl_idx is None:
        return ""  # 【修复】找不到声明时返回空字符串
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
    """清理模型输出并抽取可编译的 Lean statement。

    【修复】增加占位符过滤，阻断 'none' 等无效输出。
    """
    if not text:
        return ""

    # 【修复】过滤常见占位符（模型可能输出这些作为"无结果"标记）
    INVALID_PLACEHOLDERS = {"none", "null", "nil", "n/a", "na", ""}
    text_lower = text.strip().lower()
    if text_lower in INVALID_PLACEHOLDERS:
        return ""

    cleaned = remove_lean_comments(text)
    cleaned = strip_header_lines(cleaned)
    cleaned = strip_evolve_markers(cleaned)
    result = extract_first_declaration(cleaned)

    # 【修复】再次检查结果是否为占位符
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

    # Prefer ```lean / ```lean4 fenced blocks.
    m = re.search(r"```(?:lean4?|lean)\s*\n(.*?)```", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    else:
        # Fallback: any fenced block.
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

# =============================================================================
# Cycle-consistency prompt normalization
# =============================================================================
#
# Cycle-consistency scoring uses log P(informal | "Informalize: {formal_statement}").
# In practice, this can be sensitive to *semantically irrelevant naming noise*,
# especially the top-level declaration name (theorem/lemma/def name).
#
# To reduce this variance (and make comparisons across runs fairer), we
# normalize the first top-level declaration name before constructing the cycle
# prompt.
#
# IMPORTANT:
# - This ONLY affects cycle_score / cycle_normalized_log_prob computation.
# - It does NOT change compilation checks or the stored candidate code.
#

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

    # Replace only the first matching declaration name.
    #
    # We intentionally limit to common named top-level declarations.
    # Anonymous `example : ...` forms do not match and are left unchanged.
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

    # Drop leading attribute lines (e.g. @[simp]) to ensure we match the decl line.
    s = re.sub(r"(?m)^\s*@\[.*\]\s*\n?", "", s).strip()

    # Remove the first occurrence of the declaration keyword.
    s2 = re.sub(r"(?m)^\s*(?:noncomputable\s+)?theorem\s+", "", s, count=1).strip()

    # Remove the declaration name token.
    # Example:
    #   "my_theorem (x : ℝ) : P x := by sorry"
    # → "(x : ℝ) : P x := by sorry"
    m_name = re.match(r"^\s*([^\s:(]+)\s*(.*)\Z", s2, flags=re.DOTALL)
    if not m_name:
        return s2.strip()
    return (m_name.group(2) or "").strip()


# =============================================================================
# Lean 编译检查
# =============================================================================

def check_lean_compile(
    code: str,
    timeout: int = 600,
    lean_server_url: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    检查 Lean 代码是否能编译通过。

    说明：
    - 现在候选应输出「完整 Lean 文件」（imports + theorem）。
    - 编译检查直接对完整代码执行（而不是外部 header + body 拼接）。

    【返回值】
    - (True, "") 表示编译成功
    - (False, error_msg) 表示编译失败，error_msg 包含错误信息
    """
    # This project primarily relies on local compilation via `lean_interact`.
    # An HTTP Lean server may exist in some environments, but is optional.
    #
    # Rules:
    # - If `lean_server_url` (or env `LEAN_SERVER_URL`) starts with http(s):// → try HTTP first.
    # - Otherwise (unset / "local" / etc.) → use local lean_interact directly.
    raw_server_url = (lean_server_url or os.environ.get("LEAN_SERVER_URL", "")).strip()
    use_http = raw_server_url.startswith("http://") or raw_server_url.startswith("https://")
    LEAN_SERVER_URL = raw_server_url if use_http else ""

    full_code = normalize_lean_code(code)
    if not full_code:
        return False, "Empty Lean code"

    # 基本结构检查：必须包含一个顶层声明关键字
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
            timeout=timeout + 30,  # HTTP timeout slightly longer than Lean timeout
        )
        response.raise_for_status()
        result = response.json()

        # 解析结果
        if "results" not in result or len(result["results"]) == 0:
            return False, "No results from Lean server"

        check_result = result["results"][0]
        messages = check_result.get("response", {}).get("messages", [])

        # 检查是否有 error 级别的消息
        errors = [m for m in messages if m.get("severity") == "error"]

        if errors:
            # 提取错误信息
            error_msgs = [e.get("data", "Unknown error") for e in errors]
            return False, "\n".join(error_msgs)

        # 没有错误，编译成功
        return True, ""

    except requests.exceptions.Timeout:
        return _fallback_lean_interact()
    except requests.exceptions.ConnectionError:
        return _fallback_lean_interact()
    except requests.exceptions.RequestException as e:
        return False, f"[RequestError] {type(e).__name__}: {e}"
    except Exception as e:
        return False, f"[Exception] {type(e).__name__}: {e}"


# =============================================================================
# CriticLean 语义检查
# =============================================================================

async def check_critic_lean(informal: str, formal: str) -> Tuple[int, str]:
    """
    使用 CriticLean 检查语义一致性。

    【设计目的】
    编译通过只能保证语法正确，但不能保证数学含义正确。
    CriticLean 是一个 LLM-based 的语义检查器，它会：
    1. 分析 informal statement 的数学含义
    2. 分析 formal statement 的数学含义
    3. 比较两者是否一致

    【为什么只在 compile_ok=1 时调用？】
    1. 编译失败的 statement 可能语法残缺，语义检查没有意义
    2. 节省 API 调用成本
    3. 符合"词典序 gating"的设计原则

    【返回值】
    - (1, reason) 表示语义一致
    - (0, reason) 表示语义不一致或检查失败
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

    # Fallback: try to find the start marker and slice to end.
    m2 = re.search(r"(?is)\b5\s*\.?\s*Accuracy\s*Confirmation\b", s)
    if not m2:
        return ""
    return s[m2.end() :].lstrip(" :\n\t").strip()


# =============================================================================
# BEq+ 等价性检查（可选）
# =============================================================================

def check_beq_plus(statement: str, ground_truth: str, header: str, timeout: int = 600) -> int:
    """
    使用 BEq+ 检查与 ground truth 的形式等价性。

    【设计目的】
    BEq+ 是一个高精度、低召回率的等价性检查器：
    - 高精度：如果返回 1，几乎可以确定两个 statement 数学等价
    - 低召回率：即使两个 statement 等价，也可能返回 0

    【为什么是可选的？】
    1. 需要 ground truth，但很多任务没有标准答案
    2. 计算成本较高
    3. 低召回率意味着可能错过正确答案

    【返回值】
    - 1 表示等价
    - 0 表示不等价或检查失败
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


# =============================================================================
# 辅助函数
# =============================================================================

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
    """为实验运行提供参数。"""
    config = get_config()
    return {
        "informal": config["informal"],
        "header": config["header"],
        "ground_truth": config["ground_truth"],
        "use_beq": config["use_beq"],
    }


# =============================================================================
# 核心：门控评分函数
# =============================================================================

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
    计算门控适应度分数。

    ================================================================================
                              这是整个系统的"心脏"
    ================================================================================

    【约束 F 的实现】
    规约要求：compile_ok=0 的候选在排序上永远劣于 compile_ok=1 的候选。

    【公式 - 与 SPEC.md 一致（加分式，且保证 beq > semantic > cycle）】
    combined_score = compile_ok * (
        score_base
        + score_cycle_weight * cycle_score
        + score_semantic_bonus * semantic_ok
        + score_beq_bonus * beq_ok
    )

    【为什么这个公式能满足约束 F？】
    - 当 compile_ok = 0 时，整个乘积为 0
    - 当 compile_ok = 1 时，最低分是 score_base（默认 100）
    - 因此任何 compile_ok=1 的候选（≥100）都优于 compile_ok=0 的候选（=0）

    【分数分布 - 直觉示例（默认权重）】
    - compile_ok=0: score = 0
    - compile_ok=1, cycle=1.0, semantic=0, beq=0: score = 150
    - compile_ok=1, cycle=0.0, semantic=1, beq=0: score = 200
    - compile_ok=1, cycle=1.0, semantic=1, beq=0: score = 250
    - compile_ok=1, cycle=0.0, semantic=0, beq=1: score = 300
    - compile_ok=1, cycle=1.0, semantic=1, beq=1: score = 450

    【权重设计】
    - score_base=100：compile_ok=1 的基础分
    - score_cycle_weight * cycle_score：连续信号（cycle_score∈[0,1]）
    - score_semantic_bonus * semantic_ok：离散加分（semantic_ok∈{0,1}）
    - score_beq_bonus * beq_ok：离散加分（beq_ok∈{0,1}）

    约束：为保证排序语义（beq > semantic > cycle），默认应满足：
    - score_semantic_bonus > score_cycle_weight
    - score_beq_bonus > score_semantic_bonus + score_cycle_weight
    """
    # potential 预留字段：当前不参与 combined_score（避免口径漂移）
    return float(
        int(compile_ok)
        * (
            float(score_base)
            + float(score_cycle_weight) * float(cycle_score)
            + float(score_semantic_bonus) * int(semantic_ok)
            + float(score_beq_bonus) * int(beq_ok)
        )
    )


# =============================================================================
# 核心：评估聚合函数
# =============================================================================

def aggregate_metrics(
    results: List[str],
    results_dir: str,
) -> Dict[str, Any]:
    """
    聚合评估结果，返回完整的 metrics 字典。

    ================================================================================
                              这是评估器的"主函数"
    ================================================================================

    【设计原则：词典序 gating】
    评估顺序严格按优先级执行：
    1. 先检查 compile_ok（最重要）
    2. 只有 compile_ok=1 时，才计算软信号（cycle-consistency；Critic 语义检查可选）
    3. 只有 compile_ok=1 时，才检查 beq_ok

    这就是"词典序"的含义：先比较第一位，相等才比较第二位，以此类推。

    【约束满足】
    - [D] 编译检查在 {header}+{body} 上执行（见 check_lean_compile）
    - [F] compile_ok=0 时 score 恒为 0（见 compute_gated_score）

    【返回的 metrics 字典】
    {
        "compile_ok": 0 或 1,
        "compile_error_type": 错误类型（仅当 compile_ok=0）,
        "compile_error_msg": 错误信息（仅当 compile_ok=0）,
        "semantic_ok": 0 或 1,
        "critic_raw": CriticLean 的原始输出,
        "beq_ok": 0 或 1,
        "potential": 0.0（v1 固定）,
        "fitness_tuple": (compile_ok, beq_ok, semantic_ok, cycle_score),
        "combined_score": 门控评分,
        "statement": 原始 Lean statement,
    }
    """
    # 处理空结果的边界情况
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

    # Extract the first declaration (statement-only) for cycle / semantic / BEq.
    statement = normalize_lean_statement(lean_code)

    config = get_config()
    lean_server_url = config.get("lean_server_url")  # 从配置中获取 lean_server_url

    # =========================================================================
    # Step 1: Lean 编译检查
    # =========================================================================
    # Compile check on the full Lean file (imports + theorem).
    compile_ok_bool, compile_error = _compile_cached(
        code=lean_code,
        timeout=config["compile_timeout"],
        lean_server_url=lean_server_url,
    )
    compile_ok = 1 if compile_ok_bool else 0

    # 提取错误类型，用于后续的 repair prompt
    compile_error_type = extract_compile_error_type(compile_error) if compile_error else ""
    compile_error_msg = truncate_error_msg(compile_error)

    # potential 在 v1 固定为 0.0，接口保留
    potential = 0.0

    # 初始化 metrics 字典
    metrics = {
        # 核心适应度组件
        "compile_ok": compile_ok,
        "compile_error_type": compile_error_type,
        "compile_error_msg": compile_error_msg,
        "semantic_ok": 0,  # optional / may be disabled
        "critic_raw": "",
        "beq_ok": 0,
        "potential": potential,
        "cycle_score": 0.0,
        # 派生分数
        "fitness_tuple": (compile_ok, 0, 0, 0.0),
        "combined_score": 0.0,
        # Raw artifacts (for repair + audit)
        "code": lean_code,
        # Extracted declaration (for scoring + downstream prompts)
        "statement": statement,
    }

    # =========================================================================
    # 词典序 gating: compile_ok=0 时跳过后续评估
    # =========================================================================
    # 【约束 F】compile_ok=0 时 score 恒为 0
    # 不需要浪费 API 调用去检查语义，直接返回
    if compile_ok == 0:
        metrics["combined_score"] = 0.0
        metrics["fitness_tuple"] = (0, 0, 0, 0.0)
        # 添加 public/private 结构（编译失败时也需要）
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

    # =========================================================================
    # Step 2+4: Semantic (CriticLean) + cycle-consistency (independent; can run in parallel)
    # =========================================================================
    # Default OFF (noisy). When disabled, semantic_ok stays 0 and does not drive scoring.
    semantic_ok = 0
    cycle_score = 0.0

    use_semantic = bool(config.get("use_semantic", False))
    use_cycle = bool(config.get("use_cycle_consistency", False))

    # IMPORTANT: CriticLean should judge the *same input* as compilation: the
    # full Lean file (imports + declaration). We never strip headers. We may
    # optionally normalize the top-level declaration name to reduce naming noise
    # when `semantic_normalize_decl_name=True`.
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
        # Semantic only.
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

        # Cycle only (default ON).
        if use_cycle:
            cycle_score, cycle_meta = cycle_consistency_score(
                target_informal=config.get("informal", ""),
                formal_statement=statement,
                cfg=config,
            )

    if use_cycle:
        metrics.update(cycle_meta)
    metrics["cycle_score"] = float(cycle_score)

    # =========================================================================
    # Step 3: BEq+ 等价性检查（可选，仅当 compile_ok=1）
    # =========================================================================
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

    # NOTE: cycle_score 已在 Step 2+4 中计算并写入 metrics。

    # =========================================================================
    # 计算最终的 fitness_tuple 和门控分数（compile 是硬门槛）
    # =========================================================================
    # NOTE: fitness_tuple 用于“词典序”停滞检测与审计显示，优先级为：
    # compile_ok > beq_ok > semantic_ok > cycle_score
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

    # =========================================================================
    # 添加 public/private 结构供 LLM 反馈使用
    # =========================================================================
    # shinka 框架期望 metrics["public"] 包含要展示给 LLM 的指标
    # 这些指标会通过 perf_str() 格式化后放入 ITER_MSG prompt
    metrics["public"] = {
        "compile_ok": compile_ok,
        "cycle_score": round(cycle_score, 4),
        "semantic_ok": int(semantic_ok),
        "beq_ok": beq_ok,
    }
    # private 包含不展示给 LLM 的调试信息
    metrics["private"] = {
        "compile_error_type": compile_error_type,
        "compile_error_msg": compile_error_msg,
        "cycle_normalized_log_prob": metrics.get("cycle_normalized_log_prob"),
    }

    # =========================================================================
    # 保存详细结果到文件（便于调试和分析）
    # =========================================================================
    try:
        # tuple 转 list 以便 JSON 序列化
        metrics_for_json = metrics.copy()
        metrics_for_json["fitness_tuple"] = list(metrics["fitness_tuple"])
        with open(Path(results_dir) / "eval_details.json", "w") as f:
            json.dump(metrics_for_json, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Evaluator] Failed to save details: {e}")

    return metrics


# =============================================================================
# 主函数：评估入口
# =============================================================================

def main(program_path: str, results_dir: str):
    """
    评估器主入口。

    由 ShinkaEvolve 框架调用，用于评估单个候选程序。

    【调用链路】
    run_evo.py → AutoformalizationRunner → scheduler → evaluate.py:main()

    【执行流程图】
    ┌──────────────────┐
    │ 接收参数          │  program_path: 候选程序文件
    │ (program_path,   │  results_dir: 结果保存目录
    │  results_dir)    │
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │ run_shinka_eval  │  ShinkaEvolve 标准评估框架
    │   ├─ 加载程序     │
    │   ├─ 执行获取输出 │
    │   └─ 调用聚合函数 │
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │ aggregate_metrics│  我们的核心评估逻辑
    │   ├─ Lean 编译    │  check_lean_compile()
    │   ├─ 语义检查     │  check_critic_lean()
    │   ├─ BEq+ 检查    │  check_beq_plus()
    │   └─ 门控评分     │  compute_gated_score()
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │ 返回 metrics     │  包含 fitness_tuple, combined_score 等
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
        # 调用 ShinkaEvolve 的评估框架
        metrics, correct, error_msg = run_shinka_eval(
            program_path=program_path,
            results_dir=results_dir,
            experiment_fn_name="run_formalization",
            num_runs=1,
            get_experiment_kwargs=get_experiment_kwargs,
            validate_fn=_validator,
            aggregate_metrics_fn=_aggregator,
        )

    # 打印结果（按规约 v1.1 格式）
    print(f"\n[Evaluator] Results (规约 v1.1):")
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


# =============================================================================
# 命令行入口
# =============================================================================

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
