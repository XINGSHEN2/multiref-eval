#!/usr/bin/env bash
set -euo pipefail

FINAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MLLM_DIR="${FINAL_ROOT}/mllm"
cd "${MLLM_DIR}"

if [[ "${OIS_NO_PROXY:-0}" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
else
  PROXY_URL="${OIS_PROXY_URL:-http://oversea-squid2.ko.txyun:11080}"
  export http_proxy="${PROXY_URL}" https_proxy="${PROXY_URL}" HTTP_PROXY="${PROXY_URL}" HTTPS_PROXY="${PROXY_URL}"
fi

PYTHON="${OIS_PYTHON:-python}"
XLSX="${XLSX:-}"
CSV="${CSV:-}"
BOARD="${BOARD:-all}"
STAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OIS_OUTPUT_DIR:-${OUTPUT_DIR:-${MLLM_DIR}/outputs/anatomical_consistency}}"
MANIFEST="${OIS_MANIFEST:-${OUTPUT_BASE}/inputs/mllm_${BOARD}_${STAMP}.json}"
RUN_NAME="${OIS_RUN_NAME:-anatomical_${BOARD}_${STAMP}}"
MODEL="${OIS_EVAL_MODEL:-gemini-3.1-pro-preview}"
MAX_WORKERS="${OIS_MAX_WORKERS:-4}"
TIMEOUT="${OIS_TIMEOUT:-1800}"

if [[ -n "${CSV}" ]]; then
  BUILD_ARGS=(--csv "${CSV}" --output "${MANIFEST}" --board "${BOARD}")
elif [[ -n "${XLSX}" ]]; then
  BUILD_ARGS=(--xlsx "${XLSX}" --output "${MANIFEST}" --board "${BOARD}")
else
  echo "ERROR: set CSV=/path/to/input.csv" >&2
  exit 2
fi
if [[ -n "${LIMIT_ROWS:-}" ]]; then BUILD_ARGS+=(--limit-rows "${LIMIT_ROWS}"); fi
if [[ "${ALLOW_MISSING_VIDEOS:-0}" == "1" ]]; then BUILD_ARGS+=(--allow-missing-videos); fi
"${PYTHON}" build_mllm_manifest_from_xlsx.py "${BUILD_ARGS[@]}"

ARGS=(--manifest "${MANIFEST}" --prompt "${MLLM_DIR}/MLLM_Prompt/anatomical knowledge.md" --output-dir "${OUTPUT_BASE}" --model "${MODEL}" --max-workers "${MAX_WORKERS}" --timeout "${TIMEOUT}" --run-name "${RUN_NAME}")
if [[ -n "${OIS_MODELS:-all}" ]]; then ARGS+=(--models "${OIS_MODELS:-all}"); fi
if [[ -n "${OIS_SAMPLES:-}" ]]; then ARGS+=(--samples "${OIS_SAMPLES}"); fi
if [[ -n "${OIS_LIMIT:-}" ]]; then ARGS+=(--limit "${OIS_LIMIT}"); fi
if [[ "${OIS_DRY_RUN:-0}" == "1" ]]; then ARGS+=(--dry-run); fi
if [[ "${OIS_OVERWRITE:-0}" == "1" ]]; then ARGS+=(--overwrite); fi

"${PYTHON}" eval_anatomical_consistency.py "${ARGS[@]}"
LATEST_FILE="${OUTPUT_BASE}/latest_run.txt"
if [[ -f "${LATEST_FILE}" ]]; then
  LATEST_RUN="$(cat "${LATEST_FILE}")"
else
  LATEST_RUN="$(find "${OUTPUT_BASE}" -maxdepth 1 -type d -name "${RUN_NAME}_*" | sort | tail -1)"
fi
if [[ -z "${LATEST_RUN}" || ! -f "${LATEST_RUN}/anatomical_results.json" ]]; then
  echo "ERROR: cannot locate anatomical run output for RUN_NAME=${RUN_NAME}" >&2
  exit 1
fi
"${PYTHON}" summarize_anatomical_consistency.py --run-dir "${LATEST_RUN}"
echo "[done] anatomical latest: ${LATEST_RUN}"
echo "[done] anatomical ranking md: ${LATEST_RUN}/anatomical_ranking.md"
