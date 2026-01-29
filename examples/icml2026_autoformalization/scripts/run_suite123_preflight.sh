#!/usr/bin/env bash
set -euo pipefail


ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"

DATASET="${1:-proofnet_test}"      # proofnet_test | proofnet_full | combibench
NUM_PROBLEMS="${2:-2}"
MAX_CALLS="${3:-10}"
LLM_MODE="${4:-mock}"             # mock | auto | real | replay
CONCURRENCY="${5:-1}"

shift $(( $# > 5 ? 5 : $# )) || true
EXTRA_ARGS=("$@")

OUT_BASE="/tmp/formalevolve_suite123_${DATASET}_n${NUM_PROBLEMS}_calls${MAX_CALLS}_${TS}"
mkdir -p "${OUT_BASE}"

export AUTOFORMAL_MOCK_STATEMENTS_PATH="${ROOT}/fixtures/mock_statements.json"

run_one() {
  local MODE="$1"
  local OUT="${OUT_BASE}/${MODE}"
  mkdir -p "${OUT}"

  echo "=== suite (${MODE}) dataset=${DATASET} n=${NUM_PROBLEMS} calls=${MAX_CALLS} llm_mode=${LLM_MODE} out=${OUT}"

  python "${ROOT}/scripts/run_dataset_pilot.py" \
    --dataset "${DATASET}" \
    --num_problems "${NUM_PROBLEMS}" \
    --baseline_mode "${MODE}" \
    --llm_mode "${LLM_MODE}" \
    --openai_llm_base_url "${OPENAI_LLM_BASE_URL:-}" \
    --llm_models "${AUTOFORMAL_LLM_MODELS:-Kimina-Autoformalizer-7B}" \
    --max_llm_calls "${MAX_CALLS}" \
    --concurrency "${CONCURRENCY}" \
    --num_generations 50 \
    --max_parallel_jobs 1 \
    --meta_rec_interval 0 \
    --compile_timeout 30 \
    --lean_server_url local \
    --no_cycle_consistency \
    --num_init_candidates_gen0 1 \
    --out_root "${OUT}" \
    "${EXTRA_ARGS[@]}"

  python "${ROOT}/scripts/strict_eval_run.py" --run_root "${OUT}"
  python "${ROOT}/scripts/paper_summary_run.py" --run_root "${OUT}"
}

run_one "ours"
run_one "batchN"
run_one "repairloop1"

echo "=== OK: suite1-3 finished. out_base=${OUT_BASE}"
