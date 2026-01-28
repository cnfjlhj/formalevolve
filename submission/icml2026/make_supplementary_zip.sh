#!/usr/bin/env bash
set -euo pipefail

# Create an ICML-2026-friendly supplementary code snapshot zip.
#
# Key properties:
# - No `.git/`
# - No run outputs / caches
# - No `.env` / secrets
# - Excludes `LICENSE` to avoid leaking author/license metadata during double-blind review.
#
# Usage:
#   bash submission/icml2026/make_supplementary_zip.sh /tmp/icml2026_supplementary.zip

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ZIP="${1:-/tmp/icml2026_formalevolve_supplementary_${TS}.zip}"

cd "${REPO_ROOT}"

# Always rebuild the zip from scratch so exclude patterns take effect.
rm -f "${OUT_ZIP}"

# Build from an allowlist to reduce the chance of accidentally zipping large artifacts.
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
