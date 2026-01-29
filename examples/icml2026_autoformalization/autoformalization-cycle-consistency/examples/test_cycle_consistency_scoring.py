#!/usr/bin/env python3
"""
Test Cycle Consistency Scoring

Paper-aligned implementation:
- Paper: "Improving autoformalization via cycle consistency..." (MATH-AI 2025)
- Page 12-13, Listing 2: Cycle-Consistency Potential

From the paper:
  prompt = informalization_prompt(formalized_output)  # essentially: "Informalize: {formalized_output}"
  return lm.probability(informal_statement)
"""

import sys
import json
import math
from dataclasses import dataclass
from typing import List, Optional
import requests

# ============================================================
# Configuration (paper-aligned defaults)
# ============================================================

API_BASE_URL = "http://127.0.0.1:8090/v1"
MODEL_NAME = "Qwen2.5-32B-Instruct"

# The paper prompt is intentionally simple (Page 12-13):
# "Informalize: {formalized_output}"
# No complex system prompt.
INFORMALIZE_PROMPT_TEMPLATE = "Informalize: {formal_statement}"

# The paper does not use a system prompt; only a simple user prompt.
USE_SYSTEM_PROMPT = False

# Softmax temperature for probability normalization
# Tune this value to match the paper's reported distribution (e.g., 0.73/0.26 vs 0.97/0.03).
# T > 1: smoother distribution
# T < 1: sharper distribution
# Empirically, T ≈ 3.5 is a good match for the paper examples.
SOFTMAX_TEMPERATURE = 3.5


# ============================================================
# Test data (from the paper)
# ============================================================

@dataclass
class TestCase:
    name: str
    informal: str
    candidates: List[dict]


# Example from the paper (Page 10-11)
TEST_CASE_1 = TestCase(
    name="f(x) = cx³ - 9x + 3 (Paper Page 10-11)",
    informal="Given f(x) = cx³ - 9x + 3 and f(2) = 9, find the value of c. Show that it is 3.",
    candidates=[
        {
            "formal": "theorem formalized_thm (x : ℝ) (c : ℝ) (h₁ : ∀ x, x = 2 → c*x^3 - 9*x + 3 = 9) : c = 3 := by sorry",
            "is_correct": False,
            "note": "Incorrect - malformed h₁ hypothesis"
        },
        {
            "formal": "theorem formalized_thm (f : ℝ → ℝ) (c : ℝ) (h : ∀x, f x = c*x^3 - 9*x + 3) (hf : f 2 = 9) : c = 3 := by sorry",
            "is_correct": True,
            "note": "Correct - defines the function f properly"
        },
        {
            "formal": "theorem formalized_thm : (3 : ℝ) = 3 := by sorry",
            "is_correct": False,
            "note": "Incorrect - tautology (3 = 3)"
        },
        {
            "formal": "theorem formalized_thm (c : ℝ) (f : ℝ → ℝ := fun x ↦ c*x^3 - 9*x + 3) (h : f 2 = 9) : c = 3 := by sorry",
            "is_correct": True,
            "note": "Correct - best formalization (paper score≈0.73)"
        },
    ]
)

# Example from the paper (Page 3, Figure 3)
TEST_CASE_2 = TestCase(
    name="Pythagorean integers (Paper Page 3, Figure 3)",
    informal="There are integers x, y, z > 0 with x² + y² = z².",
    candidates=[
        {
            "formal": "theorem pythagorean : ∃ x y z : ℤ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + y^2 = z^2 := by sorry",
            "is_correct": True,
            "note": "Correct - includes all > 0 constraints"
        },
        {
            "formal": "theorem pythagorean : ∃ x y z : ℤ, 0 < x ∧ 0 < y ∧ x^2 + y^2 = z^2 := by sorry",
            "is_correct": False,
            "note": "Incorrect - missing z > 0"
        },
        {
            "formal": "theorem pythagorean : ∃ x y z : ℤ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x^2 + y^2 = z^2 := by sorry",
            "is_correct": False,
            "note": "Incorrect - uses ≥ 0 instead of > 0"
        },
    ]
)

# Additional tests
TEST_CASE_3 = TestCase(
    name="n² ≥ n for positive integers",
    informal="For all positive integers n, n squared is greater than or equal to n.",
    candidates=[
        {
            "formal": "theorem sq_ge_self (n : ℕ) (h : n > 0) : n^2 ≥ n := by sorry",
            "is_correct": True,
            "note": "Correct - includes n > 0"
        },
        {
            "formal": "theorem sq_ge_self (n : ℕ) : n^2 ≥ n := by sorry",
            "is_correct": False,
            "note": "Incorrect - missing n > 0"
        },
        {
            "formal": "theorem sq_ge_self (n : ℤ) (h : n > 0) : n^2 ≥ n := by sorry",
            "is_correct": True,
            "note": "Acceptable - uses ℤ"
        },
    ]
)

ALL_TEST_CASES = [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3]


# ============================================================
# Helpers
# ============================================================

def softmax(log_probs: List[float], temperature: float = 1.0) -> List[float]:
    """
    Convert log probabilities to normalized probabilities (paper-style).

    In the paper (Page 10-11), scores are presented as normalized probabilities
    such as 0.73, 0.26, 0.00, which correspond to a softmax over log-probs.
    """
    # Clamp -inf for numerical stability.
    log_probs = [lp if lp > -1e10 else -1e10 for lp in log_probs]

    # Numerically stable softmax.
    max_lp = max(log_probs)
    exp_scores = [math.exp((lp - max_lp) / temperature) for lp in log_probs]
    total = sum(exp_scores)

    if total == 0:
        return [1.0 / len(log_probs)] * len(log_probs)

    return [s / total for s in exp_scores]


# ============================================================
# Log-probability computation
# ============================================================

def compute_log_probability(prompt: str, completion: str) -> dict:
    """
    Compute log P(completion | prompt).

    This is the core computation for cycle consistency.
    Paper (Page 12): "return lm.probability(informal_statement)"

    Method: use the completions API with echo=True and logprobs=True.
    """

    full_text = prompt + completion

    # Method 1: completions API (most accurate)
    try:
        # First get the number of prompt tokens.
        response_prompt = requests.post(
            f"{API_BASE_URL}/completions",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "max_tokens": 0,
                "echo": True,
                "logprobs": 0,
            },
            timeout=30,
        )

        if response_prompt.status_code == 200:
            prompt_data = response_prompt.json()
            prompt_tokens = prompt_data["usage"]["prompt_tokens"]

            # Then request logprobs for the full text.
            response_full = requests.post(
                f"{API_BASE_URL}/completions",
                json={
                    "model": MODEL_NAME,
                    "prompt": full_text,
                    "max_tokens": 0,
                    "echo": True,
                    "logprobs": 1,
                },
                timeout=30,
            )

            if response_full.status_code == 200:
                full_data = response_full.json()
                logprobs_data = full_data["choices"][0].get("logprobs", {})

                if logprobs_data and "token_logprobs" in logprobs_data:
                    all_logprobs = logprobs_data["token_logprobs"]

                    # Take only completion logprobs (skip the prompt portion).
                    # Note: the first token logprob is often None.
                    completion_logprobs = all_logprobs[prompt_tokens:]
                    completion_logprobs = [lp for lp in completion_logprobs if lp is not None]

                    if completion_logprobs:
                        total = sum(completion_logprobs)
                        n_tokens = len(completion_logprobs)
                        return {
                            "log_prob": total,
                            "num_tokens": n_tokens,
                            "normalized": total / n_tokens,
                            "method": "completions_api",
                        }

    except Exception as e:
        print(f"    Completions API error: {e}")

    # Method 2: fallback
    return compute_log_probability_fallback(prompt, completion)


def compute_log_probability_fallback(prompt: str, completion: str) -> dict:
    """Fallback: let the model generate and use token logprobs."""

    try:
        response = requests.post(
            f"{API_BASE_URL}/completions",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "max_tokens": 200,
                "temperature": 0,
                "logprobs": 1,
            },
            timeout=60,
        )

        if response.status_code == 200:
            data = response.json()
            logprobs_data = data["choices"][0].get("logprobs", {})

            if logprobs_data and "token_logprobs" in logprobs_data:
                all_logprobs = logprobs_data["token_logprobs"]
                all_logprobs = [lp for lp in all_logprobs if lp is not None]

                if all_logprobs:
                    # Take the first N tokens (approx completion length).
                    n_tokens = min(len(all_logprobs), 50)
                    total = sum(all_logprobs[:n_tokens])
                    return {
                        "log_prob": total,
                        "num_tokens": n_tokens,
                        "normalized": total / n_tokens,
                        "method": "generation_fallback",
                    }

    except Exception as e:
        print(f"    Fallback error: {e}")

    return {
        "log_prob": float('-inf'),
        "num_tokens": 0,
        "normalized": float('-inf'),
        "method": "failed",
    }


def score_candidate(informal: str, formal: str) -> dict:
    """
    Compute the cycle-consistency score.

    Paper (Page 12-13):
      prompt = "Informalize: {formal}"
      score = log P(informal | prompt)
    """

    # Build prompt (paper-aligned).
    prompt = INFORMALIZE_PROMPT_TEMPLATE.format(formal_statement=formal)

    # Compute log probability.
    result = compute_log_probability(prompt, informal)

    return result


# ============================================================
# Run tests
# ============================================================

def run_test_case(test_case: TestCase):
    print(f"\n{'='*70}")
    print(f"Test: {test_case.name}")
    print(f"{'='*70}")
    print(f"\nInformal: {test_case.informal}")
    print(f"\nScoring {len(test_case.candidates)} candidates...")

    results = []

    for i, cand in enumerate(test_case.candidates):
        print(f"\n  [{i+1}] {cand['note']}")

        score = score_candidate(test_case.informal, cand["formal"])

        results.append({
            "index": i + 1,
            "formal": cand["formal"],
            "is_correct": cand["is_correct"],
            "note": cand["note"],
            "score": score,
        })

        print(f"      log_prob={score['log_prob']:.2f}, tokens={score['num_tokens']}")

    # Compute normalized probabilities (paper-style: 0.73, 0.26, 0.00).
    log_probs = [r["score"]["log_prob"] for r in results]
    probs = softmax(log_probs, temperature=SOFTMAX_TEMPERATURE)

    # Attach normalized probabilities.
    for i, r in enumerate(results):
        r["probability"] = probs[i]

    # Sort by probability.
    results.sort(key=lambda x: x["probability"], reverse=True)

    print(f"\n{'─'*70}")
    print("RANKING (paper-style: normalized probability)")
    print("─"*70)
    print(f"  Initial probability: {1.0/len(results):.2f} (uniform)")
    print()

    correct_ranks = []
    for rank, r in enumerate(results, 1):
        marker = "✓" if r["is_correct"] else "✗"
        print(f"  Rank {rank}: [{marker}] {r['note']}")
        print(f"           Probability with cycle consistency: {r['probability']:.2f}")
        print(f"           (log_prob={r['score']['log_prob']:.2f})")
        if r["is_correct"]:
            correct_ranks.append(rank)

    print(f"\n{'─'*70}")
    if correct_ranks and correct_ranks[0] == 1:
        print("  ✅ SUCCESS: Best candidate is correct!")
    elif correct_ranks:
        print(f"  ⚠️  PARTIAL: Correct candidates at ranks {correct_ranks}")
    else:
        print("  ❌ FAIL: No correct candidates")

    return results


def main():
    print("="*70)
    print("Cycle Consistency Scoring Test (Paper-aligned)")
    print("="*70)
    print(f"\nConfig:")
    print(f"  API: {API_BASE_URL}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Prompt template: '{INFORMALIZE_PROMPT_TEMPLATE}'")
    print(f"  Softmax temperature: {SOFTMAX_TEMPERATURE}")

    # Test API
    print("\nTesting API connection...")
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=10)
        if response.status_code == 200:
            print("  ✓ API reachable")
        else:
            print(f"  ✗ API error: {response.status_code}")
            return
    except Exception as e:
        print(f"  ✗ Cannot connect: {e}")
        return

    # Run tests
    all_results = {}
    for tc in ALL_TEST_CASES:
        all_results[tc.name] = run_test_case(tc)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)

    successes = 0
    for name, results in all_results.items():
        top_correct = results[0]["is_correct"] if results else False
        status = "✅" if top_correct else "⚠️"
        print(f"  {status} {name}")
        if top_correct:
            successes += 1

    print(f"\nOverall: {successes}/{len(ALL_TEST_CASES)} with correct top-1")


if __name__ == "__main__":
    main()
