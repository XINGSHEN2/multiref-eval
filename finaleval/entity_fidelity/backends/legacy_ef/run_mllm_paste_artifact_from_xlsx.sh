#!/usr/bin/env bash
set -euo pipefail

FINAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLLM_DIR="${FINAL_ROOT}/mllm"

export RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export XLSX="${XLSX:-}"
export CSV="${CSV:-}"
export BOARD="${BOARD:-all}"

cd "${MLLM_DIR}"

PYTHON="${PASTE_PYTHON:-python}"
MANIFEST="${PASTE_MANIFEST:-${MLLM_DIR}/inputs/mllm_paste_${BOARD}_${RUN_TIMESTAMP}.json}"
PROMPT="${PASTE_PROMPT:-${MLLM_DIR}/MLLM_Prompt/paste artifact.md}"
OUTPUT_DIR="${PASTE_OUTPUT_DIR:-${MLLM_DIR}/outputs/paste_artifact}"
RUN_NAME="${PASTE_RUN_NAME:-paste_artifact_${BOARD}_${RUN_TIMESTAMP}}"
MODEL="${PASTE_EVAL_MODEL:-gemini-3.1-pro-preview}"
LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
CREDENTIALS="${PASTE_GEMINI_CREDENTIALS:-${MULTIREF_COMPASS_GEMINI_CREDENTIALS:-${FINALEVAL_GEMINI_CREDENTIALS:-}}}"
MODELS="${PASTE_MODELS:-all}"
SAMPLES="${PASTE_SAMPLES:-all}"
MAX_WORKERS="${PASTE_MAX_WORKERS:-12}"
TIMEOUT="${PASTE_TIMEOUT:-1800}"
TEMPERATURE="${PASTE_TEMPERATURE:-0.0}"

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

ARGS=(
  --manifest "${MANIFEST}"
  --prompt "${PROMPT}"
  --output-dir "${OUTPUT_DIR}"
  --run-name "${RUN_NAME}"
  --model "${MODEL}"
  --location "${LOCATION}"
  --credentials "${CREDENTIALS}"
  --models "${MODELS}"
  --samples "${SAMPLES}"
  --max-workers "${MAX_WORKERS}"
  --timeout "${TIMEOUT}"
  --temperature "${TEMPERATURE}"
)
if [[ -n "${PASTE_LIMIT:-}" ]]; then ARGS+=(--limit "${PASTE_LIMIT}"); fi
if [[ "${PASTE_OVERWRITE:-0}" == "1" ]]; then ARGS+=(--overwrite); fi

"${PYTHON}" paste_artifact_eval.py "${ARGS[@]}"

LATEST_RUN="$(cat "${OUTPUT_DIR}/latest_run.txt")"
echo "[done] paste artifact latest: ${LATEST_RUN}"
