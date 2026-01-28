# Cycle Consistency Autoformalization

基于论文 ["Improving autoformalization via cycle consistency and incremental type-checking using language-model probabilistic programs"](https://arxiv.org/abs/...) (MATH-AI 2025) 的简单实现。

## 核心思想

Cycle Consistency 通过"反向翻译"来评估形式化的质量：

```
Informal Statement ──[formalize]──> Formal Candidates
                                          │
                                          ▼
                            ┌─────────────────────────┐
                            │  Cycle Consistency      │
                            │  Score = log P(informal │
                            │            | "Informalize: formal") │
                            └─────────────────────────┘
                                          │
                                          ▼
                                    Best Candidate
```

好的形式化 → 反向翻译后应该接近原文 → 高分
错误的形式化 → 反向翻译后会有偏差 → 低分

## 目录结构

```
autoformalization-cycle-consistency/
├── src/
│   ├── __init__.py
│   ├── config.py           # 配置
│   ├── model_interface.py  # 模型接口抽象
│   └── cycle_consistency.py # 核心实现
├── examples/
│   ├── demo_with_dummy.py      # 用假模型测试流程
│   └── demo_with_real_models.py # 用真实模型运行
├── configs/
└── README.md
```

## 快速开始

### 1. 测试流程（不需要模型）

```bash
cd examples
python demo_with_dummy.py
```

### 2. 用真实模型运行

首先部署模型（例如用 vLLM）：

```bash
# 终端1: 部署形式化模型
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B --port 8000

# 终端2: 部署反向翻译模型
vllm serve meta-llama/Llama-3.2-3B-Instruct --port 8001
```

然后修改 `examples/demo_with_real_models.py` 中的配置并运行：

```bash
python demo_with_real_models.py
```

## 使用方法

### 基本用法

```python
from src import (
    CycleConsistencyAutoformalization,
    OpenAICompatibleLLM,
    Config,
)

# 1. 创建模型
formalizer = OpenAICompatibleLLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    base_url="http://localhost:8000/v1",
)
informalizer = OpenAICompatibleLLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    base_url="http://localhost:8001/v1",
)

# 2. 创建 Cycle Consistency 实例
cc = CycleConsistencyAutoformalization(
    formalizer=formalizer,
    informalizer=informalizer,
)

# 3. 运行自动形式化
result = cc.autoformalize(
    "For all positive integers n, n squared is greater than or equal to n."
)

print(result.best_formalization)
# theorem forall_n_sq_ge_n (n : ℕ) (h : n > 0) : n^2 ≥ n := by sorry
```

### 使用 HuggingFace 模型（本地 GPU）

```python
from src import HuggingFaceLLM

# 直接加载模型到 GPU
informalizer = HuggingFaceLLM(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    device="auto",
)

# compute_log_prob 会精确计算 log probability
score = informalizer.compute_log_prob(
    prompt="Informalize: theorem (n : ℕ) : n^2 ≥ n",
    completion="For all natural numbers n, n squared is at least n.",
)
print(f"Log prob: {score.log_prob}, Tokens: {score.num_tokens}")
```

### 自定义配置

```python
from src import Config

config = Config()
config.model.num_candidates = 10      # 生成更多候选
config.model.temperature = 0.8        # 更高多样性
config.cycle_consistency.normalize_by_length = True  # 按长度归一化分数
config.verbose = True                 # 打印详细日志

cc = CycleConsistencyAutoformalization(
    formalizer=formalizer,
    informalizer=informalizer,
    config=config,
)
```

## 模型接口

提供三种模型接口：

| 接口 | 说明 | 依赖 |
|------|------|------|
| `OpenAICompatibleLLM` | 用于 vLLM/Ollama 等 OpenAI 兼容 API | `openai` |
| `HuggingFaceLLM` | 直接加载 HuggingFace 模型 | `transformers`, `torch` |
| `DummyLLM` | 测试用，返回假数据 | 无 |

你也可以实现自己的接口，只需继承 `LLMInterface` 并实现 `generate()` 和 `compute_log_prob()` 方法。

## 依赖

```bash
# 最小依赖
pip install openai

# 如果要用 HuggingFace 模型
pip install transformers torch

# 推荐：用 vLLM 部署模型
pip install vllm
```

## 论文引用

```bibtex
@inproceedings{barbadacosta2025improving,
  title={Improving autoformalization via cycle consistency and incremental
         type-checking using language-model probabilistic programs},
  author={Barba da Costa, Mauricio and Zaiser, Fabian and Collins, Katherine M.
          and Patel, Romir and O'Donnell, Timothy J. and Lew, Alexander K.
          and Tenenbaum, Joshua B. and Mansinghka, Vikash K. and Freer, Cameron E.},
  booktitle={The 5th Workshop on Mathematical Reasoning and AI (MATH-AI)},
  year={2025}
}
```

## 扩展

这个实现只包含 Cycle Consistency 部分。论文还有：

- **增量类型检查 (Incremental Type-Checking)**: 需要与 Lean 4 集成
- **SMC 采样**: 需要 GenLM 框架

如果需要完整复现，可以参考 [GenLM](https://github.com/genlm/genlm-control) 库。
