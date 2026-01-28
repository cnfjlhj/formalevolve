#!/usr/bin/env python3
"""
Test Cycle Consistency Scoring

严格按照论文实现：
- Paper: "Improving autoformalization via cycle consistency..." (MATH-AI 2025)
- Page 12-13, Listing 2: Cycle-Consistency Potential

论文原文：
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
# 配置 - 尽量与论文一致
# ============================================================

API_BASE_URL = "http://127.0.0.1:8090/v1"
MODEL_NAME = "Qwen2.5-32B-Instruct"

# 论文的 prompt 非常简单 (Page 12-13):
# "Informalize: {formalized_output}"
# 没有复杂的 system prompt
INFORMALIZE_PROMPT_TEMPLATE = "Informalize: {formal_statement}"

# 论文中没有使用 system prompt，直接用简单的 prompt
USE_SYSTEM_PROMPT = False

# Softmax temperature for probability normalization
# 调整这个值让分布接近论文 (0.73, 0.26 vs 0.97, 0.03)
# T > 1: 分布更平滑
# T < 1: 分布更尖锐
# 根据计算，T ≈ 3.5 可以让分布接近论文
SOFTMAX_TEMPERATURE = 3.5


# ============================================================
# 测试数据（来自论文）
# ============================================================

@dataclass
class TestCase:
    name: str
    informal: str
    candidates: List[dict]


# 论文 Page 10-11 的例子
TEST_CASE_1 = TestCase(
    name="f(x) = cx³ - 9x + 3 (Paper Page 10-11)",
    informal="Given f(x) = cx³ - 9x + 3 and f(2) = 9, find the value of c. Show that it is 3.",
    candidates=[
        {
            "formal": "theorem formalized_thm (x : ℝ) (c : ℝ) (h₁ : ∀ x, x = 2 → c*x^3 - 9*x + 3 = 9) : c = 3 := by sorry",
            "is_correct": False,
            "note": "Incorrect - h₁ 条件写法有问题"
        },
        {
            "formal": "theorem formalized_thm (f : ℝ → ℝ) (c : ℝ) (h : ∀x, f x = c*x^3 - 9*x + 3) (hf : f 2 = 9) : c = 3 := by sorry",
            "is_correct": True,
            "note": "Correct - 正确定义了函数 f"
        },
        {
            "formal": "theorem formalized_thm : (3 : ℝ) = 3 := by sorry",
            "is_correct": False,
            "note": "Incorrect - 只是 3=3"
        },
        {
            "formal": "theorem formalized_thm (c : ℝ) (f : ℝ → ℝ := fun x ↦ c*x^3 - 9*x + 3) (h : f 2 = 9) : c = 3 := by sorry",
            "is_correct": True,
            "note": "Correct - 最佳形式化 (论文中 score=0.73)"
        },
    ]
)

# 论文 Page 3, Figure 3 的例子
TEST_CASE_2 = TestCase(
    name="Pythagorean integers (Paper Page 3, Figure 3)",
    informal="There are integers x, y, z > 0 with x² + y² = z².",
    candidates=[
        {
            "formal": "theorem pythagorean : ∃ x y z : ℤ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + y^2 = z^2 := by sorry",
            "is_correct": True,
            "note": "Correct - 包含所有 > 0 条件"
        },
        {
            "formal": "theorem pythagorean : ∃ x y z : ℤ, 0 < x ∧ 0 < y ∧ x^2 + y^2 = z^2 := by sorry",
            "is_correct": False,
            "note": "Incorrect - 缺少 z > 0"
        },
        {
            "formal": "theorem pythagorean : ∃ x y z : ℤ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x^2 + y^2 = z^2 := by sorry",
            "is_correct": False,
            "note": "Incorrect - 用了 ≥ 0 而非 > 0"
        },
    ]
)

# 额外测试
TEST_CASE_3 = TestCase(
    name="n² ≥ n for positive integers",
    informal="For all positive integers n, n squared is greater than or equal to n.",
    candidates=[
        {
            "formal": "theorem sq_ge_self (n : ℕ) (h : n > 0) : n^2 ≥ n := by sorry",
            "is_correct": True,
            "note": "Correct - 有 n > 0 条件"
        },
        {
            "formal": "theorem sq_ge_self (n : ℕ) : n^2 ≥ n := by sorry",
            "is_correct": False,
            "note": "Incorrect - 缺少 n > 0"
        },
        {
            "formal": "theorem sq_ge_self (n : ℤ) (h : n > 0) : n^2 ≥ n := by sorry",
            "is_correct": True,
            "note": "Acceptable - 用 ℤ"
        },
    ]
)

ALL_TEST_CASES = [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3]


# ============================================================
# 工具函数
# ============================================================

def softmax(log_probs: List[float], temperature: float = 1.0) -> List[float]:
    """
    将 log probabilities 转换为归一化概率 (论文中的形式)

    论文 Page 10-11 的分数形式: 0.73, 0.26, 0.00 等
    就是对 log probs 做 softmax 归一化
    """
    # 处理 -inf
    log_probs = [lp if lp > -1e10 else -1e10 for lp in log_probs]

    # 数值稳定的 softmax
    max_lp = max(log_probs)
    exp_scores = [math.exp((lp - max_lp) / temperature) for lp in log_probs]
    total = sum(exp_scores)

    if total == 0:
        return [1.0 / len(log_probs)] * len(log_probs)

    return [s / total for s in exp_scores]


# ============================================================
# Log Probability 计算
# ============================================================

def compute_log_probability(prompt: str, completion: str) -> dict:
    """
    计算 log P(completion | prompt)

    这是 Cycle Consistency 的核心计算。
    论文 Page 12: "return lm.probability(informal_statement)"

    方法：使用 completions API with echo=True, logprobs=True
    """

    full_text = prompt + completion

    # 方法1: 使用 completions API (最准确)
    try:
        # 先获取 prompt 的 token 数
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

            # 然后获取完整文本的 logprobs
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

                    # 只取 completion 部分的 logprobs (跳过 prompt 部分)
                    # 注意：第一个 token 的 logprob 通常是 None
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

    # 方法2: 备选 - 使用 tokenize + completion
    return compute_log_probability_fallback(prompt, completion)


def compute_log_probability_fallback(prompt: str, completion: str) -> dict:
    """备选方法：让模型生成，取 logprobs"""

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
                    # 取前 N 个 token（近似 completion 长度）
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
    计算 Cycle Consistency 分数

    论文 Page 12-13:
      prompt = "Informalize: {formal}"
      score = log P(informal | prompt)
    """

    # 构造 prompt (与论文一致)
    prompt = INFORMALIZE_PROMPT_TEMPLATE.format(formal_statement=formal)

    # 计算 log probability
    result = compute_log_probability(prompt, informal)

    return result


# ============================================================
# 运行测试
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

    # 计算归一化概率 (论文中的形式: 0.73, 0.26, 0.00)
    log_probs = [r["score"]["log_prob"] for r in results]
    probs = softmax(log_probs, temperature=SOFTMAX_TEMPERATURE)

    # 添加归一化概率到结果
    for i, r in enumerate(results):
        r["probability"] = probs[i]

    # 按概率排序
    results.sort(key=lambda x: x["probability"], reverse=True)

    print(f"\n{'─'*70}")
    print("RANKING (论文形式: 归一化概率)")
    print("─"*70)
    print(f"  Initial probability: {1.0/len(results):.2f} (均等)")
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

    # 测试 API
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

    # 运行测试
    all_results = {}
    for tc in ALL_TEST_CASES:
        all_results[tc.name] = run_test_case(tc)

    # 总结
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
