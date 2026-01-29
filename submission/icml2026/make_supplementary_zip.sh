#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ZIP="${1:-/tmp/icml2026_formalevolve_supplementary_${TS}.zip}"

cd "${REPO_ROOT}"

rm -f "${OUT_ZIP}"

zip -r "${OUT_ZIP}" \
  README.md \
  pyproject.toml \
  problem_config.json \
  shinka \
  autoformalization \
  configs \
  examples \
  tests \
  submission/icml2026/README_SUPPLEMENTARY.md \
  submission/icml2026/supplementary_results.pdf \
  submission/icml2026/supplementary_results.tex \
  -x \
  "*/__pycache__/*" \
  "__pycache__/*" \
  "*.pyc" \
  "*.bak" \
  "*.bak_*" \
  "*.backup" \
  "*.aux" \
  "*.out" \
  "*.xdv" \
  "*.fdb_latexmk" \
  "*.fls" \
  "*.synctex.gz" \
  "submission/icml2026/*.log" \
  ".git/*" \
  ".env" \
  ".venv/*" \
  "env/*" \
  "venv/*" \
  "results/*" \
  "results_*/*" \
  "outputs/*" \
  "output/*" \
  "*/results/*" \
  "*/results_*/*" \
  "*/outputs/*" \
  "*/output/*" \
  "LICENSE"

echo "${OUT_ZIP}"
