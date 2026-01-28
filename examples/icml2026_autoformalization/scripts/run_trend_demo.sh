#!/usr/bin/env bash
set -euo pipefail

# Trend-level reproduction script.
#
# This script runs the same dataset slice twice (e.g., ours vs batchN),
# then computes strict summaries from the SQLite DB and prints a short diff.
#
# Usage:
#   ./run_trend_demo.sh proofnet_test 10 100
#
# Notes:
# - For real LLM runs:
#   - Set OPENAI_API_KEY (OpenAI official), OR
#   - Set OPENAI_LLM_BASE_URL + AUTOFORMAL_LLM_MODELS (OpenAI-compatible local server).
# - Semantic judging is enabled via `--use_semantic` and requires CriticLean:
#     CRITIC_LEAN_BASE_URL=http://...  (or CRITIC_LEAN_URL=http://.../v1/chat/completions)
#     CRITIC_LEAN_MODEL=<model_id>     (auto-detected if the server hosts a single model)
# - Other optional services (e.g. cycle-consistency) are disabled by default.
# - Semantic judging / cycle-consistency are controlled via flags passed through
#   `run_dataset_pilot.py` (which writes per-problem problem_config.json).

DATASET="${1:-proofnet_test}"
NUM_PROBLEMS="${2:-10}"
MAX_CALLS="${3:-100}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="${EXAMPLE_ROOT}/results_trend_${DATASET}_calls${MAX_CALLS}_${TS}"

echo "[trend] dataset=${DATASET} num_problems=${NUM_PROBLEMS} max_llm_calls=${MAX_CALLS}"
echo "[trend] out_base=${OUT_BASE}"

python "${EXAMPLE_ROOT}/scripts/run_dataset_pilot.py" \
  --dataset "${DATASET}" \
  --num_problems "${NUM_PROBLEMS}" \
  --max_llm_calls "${MAX_CALLS}" \
  --baseline_mode ours \
  --use_semantic \
  --no_cycle_consistency \
  --out_root "${OUT_BASE}/ours"

python "${EXAMPLE_ROOT}/scripts/run_dataset_pilot.py" \
  --dataset "${DATASET}" \
  --num_problems "${NUM_PROBLEMS}" \
  --max_llm_calls "${MAX_CALLS}" \
  --baseline_mode batchN \
  --use_semantic \
  --no_cycle_consistency \
  --out_root "${OUT_BASE}/batchN"

python "${EXAMPLE_ROOT}/scripts/strict_eval_run.py" --run_root "${OUT_BASE}/ours"
python "${EXAMPLE_ROOT}/scripts/strict_eval_run.py" --run_root "${OUT_BASE}/batchN"

python "${EXAMPLE_ROOT}/scripts/compare_strict_summaries.py" \
  --a "${OUT_BASE}/ours/strict_summary.json" \
  --b "${OUT_BASE}/batchN/strict_summary.json" \
  --label-a "ours" \
  --label-b "batchN"

echo "[trend] done"
