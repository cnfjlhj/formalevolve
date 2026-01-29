#!/usr/bin/env bash
set -euo pipefail


DATASET="${1:-proofnet_test}"
NUM_PROBLEMS="${2:-10}"
MAX_CALLS="${3:-100}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="${EXAMPLE_ROOT}/results_trend_${DATASET}_calls${MAX_CALLS}_${TS}"

echo "[trend] dataset=${DATASET} num_problems=${NUM_PROBLEMS} max_llm_calls=${MAX_CALLS}"
echo "[trend] out_base=${OUT_BASE}"

EXTRA_SEM_ARGS=()
if [[ -n "${CRITIC_LEAN_BASE_URL:-}" || -n "${CRITIC_LEAN_URL:-}" ]]; then
  EXTRA_SEM_ARGS+=(--use_semantic --enable_semantic_repair)
else
  echo "[trend] NOTE: CRITIC_LEAN_BASE_URL/CRITIC_LEAN_URL not set; running compile-only (SH@T will be 0)."
fi

python "${EXAMPLE_ROOT}/scripts/run_dataset_pilot.py" \
  --dataset "${DATASET}" \
  --num_problems "${NUM_PROBLEMS}" \
  --max_llm_calls "${MAX_CALLS}" \
  --baseline_mode ours \
  --paper_protocol \
  "${EXTRA_SEM_ARGS[@]}" \
  --no_cycle_consistency \
  --out_root "${OUT_BASE}/ours"

python "${EXAMPLE_ROOT}/scripts/run_dataset_pilot.py" \
  --dataset "${DATASET}" \
  --num_problems "${NUM_PROBLEMS}" \
  --max_llm_calls "${MAX_CALLS}" \
  --baseline_mode batchN \
  --paper_protocol \
  "${EXTRA_SEM_ARGS[@]}" \
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
