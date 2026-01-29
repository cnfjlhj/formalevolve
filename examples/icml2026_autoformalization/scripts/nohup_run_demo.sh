#!/usr/bin/env bash
set -euo pipefail


die() { echo "[nohup_run_demo] ERROR: $*" >&2; exit 2; }

DATASET="${1:-proofnet_test}"
NUM_PROBLEMS="${2:-20}"
MAX_CALLS="${3:-100}"
CONCURRENCY="${CONCURRENCY:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-400}"
COMPILE_TIMEOUT="${COMPILE_TIMEOUT:-60}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${OPENAI_LLM_BASE_URL:-}" ]]; then
  die "Set OPENAI_LLM_BASE_URL (OpenAI-compatible, e.g. http://<host>:<port>/v1)"
fi
if [[ -z "${AUTOFORMAL_LLM_MODELS:-}" ]]; then
  die "Set AUTOFORMAL_LLM_MODELS (e.g. Qwen3-30B-A3B)"
fi

CRITIC_BASE="${CRITIC_LEAN_BASE_URL:-}"
CRITIC_URL="${CRITIC_LEAN_URL:-}"
if [[ -z "${CRITIC_BASE}" && -z "${CRITIC_URL}" ]]; then
  die "Semantic judging enabled by default; set CRITIC_LEAN_BASE_URL (recommended) or CRITIC_LEAN_URL."
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/results_demo_${DATASET}__n${NUM_PROBLEMS}__calls${MAX_CALLS}__${TS}}"
mkdir -p "${OUT_ROOT}"

INIT_ROOT="${INIT_PROGRAMS_ROOT:-}"
NUM_INIT="${NUM_INIT_CANDIDATES_GEN0:-16}"
export AUTOFORMAL_REUSE_INIT_EVAL="${AUTOFORMAL_REUSE_INIT_EVAL:-1}"

export AUTOFORMAL_SEEDBANK_DEBIT_CALLS="${AUTOFORMAL_SEEDBANK_DEBIT_CALLS:-1}"
export AUTOFORMAL_SEEDBANK_CALLS_PER_SEED="${AUTOFORMAL_SEEDBANK_CALLS_PER_SEED:-1}"

LOG_OUT="${OUT_ROOT}/nohup.out"
LOG_ERR="${OUT_ROOT}/nohup.err"
PID_FILE="${OUT_ROOT}/nohup.pid"

echo "[nohup_run_demo] dataset=${DATASET} num_problems=${NUM_PROBLEMS} max_llm_calls=${MAX_CALLS}"
echo "[nohup_run_demo] out_root=${OUT_ROOT}"
echo "[nohup_run_demo] generator_base_url=${OPENAI_LLM_BASE_URL}"
echo "[nohup_run_demo] generator_models=${AUTOFORMAL_LLM_MODELS}"
echo "[nohup_run_demo] criticlean_base_url=${CRITIC_BASE}"
echo "[nohup_run_demo] criticlean_url=${CRITIC_URL}"
echo "[nohup_run_demo] seedbank_root=${INIT_ROOT}"

CMD=(python "${ROOT_DIR}/scripts/run_dataset_pilot.py"
  --dataset "${DATASET}"
  --num_problems "${NUM_PROBLEMS}"
  --baseline_mode ours
  --llm_mode auto
  --openai_llm_base_url "${OPENAI_LLM_BASE_URL}"
  --llm_models "${AUTOFORMAL_LLM_MODELS}"
  --max_llm_calls "${MAX_CALLS}"
  --concurrency "${CONCURRENCY}"
  --num_generations "${NUM_GENERATIONS}"
  --max_parallel_jobs 1
  --compile_timeout "${COMPILE_TIMEOUT}"
  --lean_server_url local
  --paper_protocol
  --use_semantic
  --enable_semantic_repair
  --no_cycle_consistency
  --out_root "${OUT_ROOT}"
)

if [[ -n "${CRITIC_BASE}" ]]; then
  CMD+=(--criticlean_base_url "${CRITIC_BASE}")
fi
if [[ -n "${INIT_ROOT}" ]]; then
  CMD+=(--init_programs_root "${INIT_ROOT}" --num_init_candidates_gen0 "${NUM_INIT}")
fi

echo "[nohup_run_demo] launching..."
nohup "${CMD[@]}" >"${LOG_OUT}" 2>"${LOG_ERR}" &
PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "[nohup_run_demo] pid=${PID}"
echo "[nohup_run_demo] logs: ${LOG_OUT} / ${LOG_ERR}"
echo "[nohup_run_demo] status: ${OUT_ROOT}/status.json"
echo "[nohup_run_demo] after completion, run:"
echo "  python ${ROOT_DIR}/scripts/strict_eval_run.py --run_root ${OUT_ROOT}"
echo "  python ${ROOT_DIR}/scripts/paper_summary_run.py --run_root ${OUT_ROOT}"
