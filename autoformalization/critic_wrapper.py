"""
CriticLean API Wrapper.

Provides async interface to CriticLean for semantic consistency evaluation.
Returns binary score (0 or 1) for formalization correctness.
"""

import aiohttp
import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Iterable, List

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)

# ============================================================================
# CriticLean Server Configuration
# ============================================================================

CRITIC_LEAN_URL = os.environ.get(
    "CRITIC_LEAN_URL",
    "http://127.0.0.1:6082/v1/chat/completions",
)
CRITIC_LEAN_MODEL = os.environ.get("CRITIC_LEAN_MODEL", "criticlean-qwen3-14b")

# Async configuration
MAX_CONCURRENT_REQUESTS = _env_int("CRITIC_LEAN_MAX_CONCURRENT_REQUESTS", 10)
REQUEST_TIMEOUT = _env_int("CRITIC_LEAN_TIMEOUT_S", 600)  # seconds (default 10 min)

# Global state
_semaphore: Optional[asyncio.Semaphore] = None
_session: Optional[aiohttp.ClientSession] = None


def _get_sem() -> asyncio.Semaphore:
    """Get or create the global semaphore for rate limiting."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    return _semaphore


async def _get_session() -> aiohttp.ClientSession:
    """Get or create the global aiohttp session."""
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


def _mk_timeout() -> aiohttp.ClientTimeout:
    """Create timeout configuration."""
    return aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)


async def close_session():
    """Close the global aiohttp session."""
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
        _session = None


# ============================================================================
# Output parsing (robust to non-JSON / truncation)
# ============================================================================


def _strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    # Common pattern: ```json\n{...}\n```
    if "```" in s:
        # Remove fence markers but keep content.
        s = re.sub(r"(?is)^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"(?is)\s*```$", "", s)
        s = s.replace("```", "")
    # Some models emit a leading `json` token.
    s = re.sub(r"(?is)^\s*json\s*", "", s)
    return s.strip()


def _extract_balanced_json_objects(text: str) -> List[str]:
    """Extract top-level balanced `{...}` substrings (ignoring braces inside JSON strings)."""
    objs: List[str] = []
    if not text:
        return objs

    depth = 0
    start: Optional[int] = None
    in_str = False
    escape = False

    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start : i + 1])
                    start = None

    return objs


def _normalize_verdict_text(text: str) -> Optional[str]:
    s = str(text or "").strip()
    if not s:
        return None

    # Strip common wrappers: quotes / brackets / parentheses.
    s = s.strip().strip(" \t\r\n")
    s = re.sub(r'^[\s\[\(\{\'"]+', "", s)
    s = re.sub(r'[\s\]\)\}\'"]+$', "", s)
    s = s.strip()

    if s.lower() == "correct":
        return "Correct"
    if s.lower() == "incorrect":
        return "Incorrect"

    # If both appear (e.g. "Correct/Incorrect"), follow the "last occurrence" rule.
    matches = list(re.finditer(r"\b(correct|incorrect)\b", s, flags=re.IGNORECASE))
    if not matches:
        return None
    last = matches[-1].group(1).lower()
    return "Correct" if last == "correct" else "Incorrect"


def _last_verdict_word(text: str) -> Optional[str]:
    """
    Very conservative fallback for extracting a verdict when we failed to parse JSON.

    We intentionally avoid matching verdict-like words inside the long "reasons"
    section (e.g. "not correct", "correct characterization"), because that can
    create false positives when the model output is truncated before emitting
    the final `is_assistant_correct` field.

    Accepted patterns (tail-first):
    - A standalone line: `Correct` / `Incorrect` (case-insensitive)
    - A quoted token: `"Correct"` / `"Incorrect"`
    - A trailing token near EOF (optionally followed by `}`/whitespace)
    """
    s = (text or "").strip()
    if not s:
        return None

    # If the output is short, models often answer in prose rather than emitting a key/value.
    # In this regime, the "last occurrence wins" heuristic is usually the best we can do.
    if len(s) <= 512:
        matches = list(re.finditer(r"\b(correct|incorrect)\b", s, flags=re.IGNORECASE))
        if matches:
            last = matches[-1].group(1).lower()
            return "Correct" if last == "correct" else "Incorrect"

    patterns = [
        # Standalone verdict line.
        re.compile(r"(?im)^\s*(correct|incorrect)\s*$"),
        # Quoted verdict token.
        re.compile(r'(?i)["\']\s*(correct|incorrect)\s*["\']'),
        # Verdict token right before end-of-output, but only if it looks like a value.
        re.compile(r"(?is)(?:^|[:\n])\s*(correct|incorrect)\b\s*[\}\]\s]*\Z"),
    ]
    for pat in patterns:
        matches = list(pat.finditer(s))
        if matches:
            last = matches[-1].group(1).lower()
            return "Correct" if last == "correct" else "Incorrect"
    return None


def _last_verdict_after_key(text: str, *, key: str = "is_assistant_correct") -> Optional[str]:
    key_re = re.compile(rf"\b{re.escape(key)}\b", flags=re.IGNORECASE)
    # Prefer "verdict-ish" occurrences that look like a value, not prose.
    word_re = re.compile(r'(?i)(?:"|\')?\s*\b(incorrect|correct)\b\s*(?:"|\')?')
    last: Optional[str] = None
    for m in key_re.finditer(text or ""):
        window = (text or "")[m.end() : m.end() + 256]
        # Only accept tokens that appear after a `:` (JSON-ish / YAML-ish).
        colon_pos = window.find(":")
        if colon_pos == -1:
            continue
        after = window[colon_pos + 1 : colon_pos + 1 + 128]
        words = list(word_re.finditer(after))
        if not words:
            continue
        w = words[-1].group(1).lower()
        last = "Correct" if w == "correct" else "Incorrect"
    return last


@dataclass(frozen=True)
class _ParsedCriticOutput:
    verdict: Optional[str]  # "Correct"/"Incorrect"/None
    reasons: str
    method: str  # json / extracted_json / salvaged_key / salvaged_word / failed


def _parse_criticlean_output(model_output: str, *, finish_reason: Optional[str]) -> _ParsedCriticOutput:
    cleaned = _strip_code_fences(model_output)
    if not cleaned:
        diag = "[ParseError] Empty model output"
        if finish_reason:
            diag += f" (finish_reason={finish_reason})"
        return _ParsedCriticOutput(verdict=None, reasons=diag, method="failed")

    # 1) Try strict JSON on the whole content first.
    candidates: List[str] = [cleaned]
    # 2) Also try any balanced JSON objects embedded in surrounding text.
    candidates.extend(_extract_balanced_json_objects(cleaned))

    parsed: Optional[_ParsedCriticOutput] = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        verdict = _normalize_verdict_text(obj.get("is_assistant_correct"))
        if not verdict:
            continue
        reasons = str(obj.get("reasons") or "").strip()
        # If reasons is missing, keep the whole cleaned output for debugging.
        if not reasons:
            reasons = cleaned
        method = "json" if cand == cleaned else "extracted_json"
        parsed = _ParsedCriticOutput(verdict=verdict, reasons=reasons, method=method)
        # Keep the last parseable verdict (matches the "last occurrence" rule).
        continue

    if parsed is not None:
        return parsed

    # 3) Salvage: prefer extracting verdict after the declared key (tail-first).
    tail = cleaned[-4000:]
    verdict = _last_verdict_after_key(tail) or _last_verdict_after_key(cleaned)
    if verdict:
        return _ParsedCriticOutput(verdict=verdict, reasons=cleaned, method="salvaged_key")

    # 4) Salvage: fallback to last "Correct/Incorrect" word (tail-first).
    verdict = _last_verdict_word(tail) or _last_verdict_word(cleaned)
    if verdict:
        return _ParsedCriticOutput(verdict=verdict, reasons=cleaned, method="salvaged_word")

    # 5) Failed.
    # Provide a helpful diagnostic; include finish_reason if available.
    diag = f"[ParseError] Could not extract verdict"
    if finish_reason:
        diag += f" (finish_reason={finish_reason})"
    return _ParsedCriticOutput(verdict=None, reasons=diag + "\n" + cleaned, method="failed")


# ============================================================================
# CriticLean Prompt (New Prompt Format)
# ============================================================================

CRITIC_PROMPT_TEMPLATE = """Role: Lean & Formal Verification Expert
Input:
- Mathematical_Text: A math problem and its answer (no proof).
- Lean4Code: A Lean 4 theorem statement formalizing the problem. Proof is intentionally omitted (e.g., sorry).
Goal:
Determine if the Lean theorem statement is an exact and faithful formalization of the mathematical problem.
**Do not evaluate or consider the answer or the proof. Your sole task is to verify the correctness of the formalization.**

Evaluation Stages (All required):
1. Math Assertion Analysis
Identify all structurally and semantically relevant components of the mathematical problem, including
variables, types, quantifiers, constraints, logic structure, conclusion, and so on. The analysis should be
based on the actual content of the text.

2. Lean Statement Analysis (ignore proof part)
Extract all structurally and semantically relevant components from the Lean statement, including
variables, types, conditions, quantifiers, constraints, the final claim, and so on. The analysis should
reflect the actual content present in the Lean code.

3. Comparative Verification
Check for exact correspondence between the math and Lean statements; you may refer to aspects like:
- Semantic alignment, logic structure, and quantifier correctness.
- Preservation of constraints and boundary assumptions.
- Accurate typing and use of variables.
- Syntactic validity and proper Lean usage (free from errors).
- Use of symbols and constructs without semantic drift.
- No missing elements, no unjustified additions, and no automatic corrections or completions.

4. Final Judgement Based solely on the above analysis, judge whether the Lean statement
is a correct and exact formalization of the mathematical problem.

5. Accuracy Confirmation
If correct: clearly confirm why all elements match.
If incorrect: list all mismatches and explain how each one affects correctness.

Note: While the analysis may be broad and open to interpreting all relevant features, the
final judgment must be based only on what is explicitly and formally expressed in the Lean statement.
**Do not consider or assess any part of the proof. Your judgment should be entirely about the accuracy
of the statement formalization.**
The JSON format you output must follow the order of first providing the reasons and then the answer, i.e., is_assistant_correct must always be the last field in the output.
Output Format:
Return exactly one JSON object(just one answer,just one json object!):
json
{{
  "reasons": "Your detailed CoT analysis:\\n1. Math Assertion Analysis: [...]\\n2. Lean Statement Analysis (Proof Ignored): [...]\\n3. Comparative Verification: [...]\\n4. Conclusion: [...]\\n5. Accuracy Confirmation: [...]",
  "is_assistant_correct": "[Correct/Incorrect]"
}}```
Input Data:
— Start of Mathematical_Text —
{informal}
— End of Mathematical_Text —
— Start of Lean4Code —
{formal}
— End of Lean4Code —"""

def _get_audit_dir() -> Optional[Path]:
    p = os.environ.get("CRITIC_LEAN_AUDIT_DIR", "").strip()
    if not p:
        return None
    d = Path(p).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _audit_write(dir_path: Path, *, prompt: str, payload: dict, response: dict, parsed: _ParsedCriticOutput) -> None:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    tag = f"{int(time.time()*1000)}_{digest}"
    out = dir_path / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.txt").write_text(prompt, encoding="utf-8")
    (out / "request.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "response.json").write_text(json.dumps(response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "parsed.json").write_text(
        json.dumps(
            {"verdict": parsed.verdict, "method": parsed.method, "reasons_preview": parsed.reasons[:400]},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


# ============================================================================
# Main API Function
# ============================================================================

async def critic_eval(informal: str, formal: str) -> Tuple[int, str]:
    """
    Evaluate semantic consistency using CriticLean.

    Args:
        informal: Natural language mathematical statement
        formal: Lean4 formalization

    Returns:
        Tuple of (score, raw_response)
        - score: 0 (incorrect) or 1 (correct)
        - raw_response: CriticLean reasoning
    """
    sem = _get_sem()
    async with sem:
        if not informal or not formal:
            return 0, "[Error] Empty informal or formal string"

        user_prompt = CRITIC_PROMPT_TEMPLATE.format(
            informal=informal,
            formal=formal
        )

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "lean_eval_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "reasons": {
                            "type": "string",
                            "description": "Detailed CoT analysis"
                        },
                        "is_assistant_correct": {
                            "type": "string",
                            "enum": ["Correct", "Incorrect"]
                        }
                    },
                    "required": ["reasons", "is_assistant_correct"],
                    "additionalProperties": False
                }
            }
        }

        payload = {
            "model": CRITIC_LEAN_MODEL,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0,
            "max_tokens": 3000,
            "response_format": response_format,
            "seed": 0
        }

        session = await _get_session()
        try:
            req_timeout = _mk_timeout()
            async with session.post(CRITIC_LEAN_URL, json=payload, timeout=req_timeout) as resp:
                data = await resp.json()
                choice = (data.get("choices") or [{}])[0]
                finish_reason = choice.get("finish_reason") or choice.get("finishReason")
                model_output = (choice.get("message", {}) or {}).get("content") or ""
                model_output = model_output.strip()
                parsed = _parse_criticlean_output(model_output, finish_reason=finish_reason)
                audit_dir = _get_audit_dir()
                if audit_dir is not None:
                    try:
                        _audit_write(audit_dir, prompt=user_prompt, payload=payload, response=data, parsed=parsed)
                    except Exception:
                        # Best-effort only; never fail the evaluation because of audit I/O.
                        pass
                if parsed.verdict:
                    score = 1 if parsed.verdict == "Correct" else 0
                    # Keep the raw-ish content for audit/debug when we had to salvage.
                    if parsed.method.startswith("salvaged"):
                        return score, f"[Salvaged:{parsed.method}] {parsed.reasons}"
                    return score, parsed.reasons

                return 0, parsed.reasons

        except asyncio.TimeoutError:
            return 0, "[Timeout] CriticLean request timed out"
        except Exception as e:
            return 0, f"[Exception] {type(e).__name__}: {e}"


async def batch_critic_eval(
    items: list[Tuple[str, str]],
    max_concurrent: int = 10
) -> list[Tuple[int, str]]:
    """
    Batch evaluate multiple (informal, formal) pairs.

    Args:
        items: List of (informal, formal) tuples
        max_concurrent: Maximum concurrent requests

    Returns:
        List of (score, raw_response) tuples
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def eval_one(informal: str, formal: str) -> Tuple[int, str]:
        async with sem:
            return await critic_eval(informal, formal)

    tasks = [eval_one(inf, form) for inf, form in items]
    return await asyncio.gather(*tasks)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def test():
        print("Testing CriticLean wrapper...")

        test_cases = [
            {
                "informal": "Prove that for any odd natural number n, 8 divides n^2 - 1.",
                "formal": "theorem test (n : ℕ) (hn : n % 2 = 1) : 8 ∣ n^2 - 1 := by sorry"
            },
            {
                "informal": "A continuous function on a compact set is bounded.",
                "formal": "theorem test {X : Type*} [TopologicalSpace X] {f : X → ℝ} {K : Set X} (hf : ContinuousOn f K) (hK : IsCompact K) : BddAbove (f '' K) := by sorry"
            }
        ]

        try:
            for i, tc in enumerate(test_cases, 1):
                print(f"\n--- Test {i} ---")
                print(f"Informal: {tc['informal'][:50]}...")
                print(f"Formal: {tc['formal'][:50]}...")

                score, reason = await critic_eval(tc["informal"], tc["formal"])
                print(f"Score: {score}")
                print(f"Reason: {reason[:200]}...")

        finally:
            await close_session()

    asyncio.run(test())
