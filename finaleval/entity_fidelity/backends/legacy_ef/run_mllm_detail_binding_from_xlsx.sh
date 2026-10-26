#!/usr/bin/env bash
set -euo pipefail

FINAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLLM_DIR="${FINAL_ROOT}/mllm"
export RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export XLSX="${XLSX:-}"
export CSV="${CSV:-}"
export BOARD="${BOARD:-all}"
METRIC="${METRIC:-both}"

run_detail() {
  cd "${MLLM_DIR}"
  PYTHON="${DP_PYTHON:-python}"
  MANIFEST="${DP_MANIFEST:-${MLLM_DIR}/inputs/mllm_${BOARD}_${RUN_TIMESTAMP}.json}"
  PROMPT="${DP_PROMPT:-${MLLM_DIR}/MLLM_Prompt/detail preservetion.md}"
  OUTPUT_DIR="${DP_OUTPUT_DIR:-${MLLM_DIR}/outputs/detail_preservation}"
  RUN_NAME="${DP_RUN_NAME:-detail_${BOARD}_${RUN_TIMESTAMP}}"
  MODEL="${DP_EVAL_MODEL:-gemini-3.5-flash}"
  LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
  CREDENTIALS="${DP_GEMINI_CREDENTIALS:-${FINALEVAL_GEMINI_CREDENTIALS:-}}"
  MODELS="${DP_MODELS:-all}"
  SAMPLES="${DP_SAMPLES:-all}"
  MAX_WORKERS="${DP_MAX_WORKERS:-12}"
  TIMEOUT="${DP_TIMEOUT:-1800}"
  TEMPERATURE="${DP_TEMPERATURE:-0.0}"

  ARGS=(--manifest "${MANIFEST}" --prompt "${PROMPT}" --output-dir "${OUTPUT_DIR}" --run-name "${RUN_NAME}" --model "${MODEL}" --location "${LOCATION}" --credentials "${CREDENTIALS}" --models "${MODELS}" --samples "${SAMPLES}" --max-workers "${MAX_WORKERS}" --timeout "${TIMEOUT}" --temperature "${TEMPERATURE}")
  if [[ -n "${DP_LIMIT:-}" ]]; then ARGS+=(--limit "${DP_LIMIT}"); fi
  if [[ "${DP_OVERWRITE:-0}" == "1" ]]; then ARGS+=(--overwrite); fi
  if [[ "${DP_REWRITE_ONLY:-0}" == "1" ]]; then ARGS+=(--rewrite-only); fi

  "${PYTHON}" detail_preservation_eval.py "${ARGS[@]}"
  LATEST_RUN="$(cat "${OUTPUT_DIR}/latest_run.txt")"
  echo "[done] detail latest: ${LATEST_RUN}"
}

run_binding() {
  cd "${MLLM_DIR}"
  PYTHON="${BINDING_PYTHON:-python}"
  MANIFEST="${BINDING_MANIFEST:-${MLLM_DIR}/inputs/mllm_${BOARD}_${RUN_TIMESTAMP}.json}"
  PROMPT="${BINDING_PROMPT:-${MLLM_DIR}/MLLM_Prompt/binding.md}"
  OUTPUT_DIR="${BINDING_OUTPUT_DIR:-${MLLM_DIR}/outputs/binding}"
  RUN_NAME="${BINDING_RUN_NAME:-binding_${BOARD}_${RUN_TIMESTAMP}}"
  MODEL="${BINDING_EVAL_MODEL:-gemini-3.5-flash}"
  LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
  CREDENTIALS="${BINDING_GEMINI_CREDENTIALS:-${FINALEVAL_GEMINI_CREDENTIALS:-}}"
  MODELS="${BINDING_MODELS:-all}"
  SAMPLES="${BINDING_SAMPLES:-all}"
  MAX_WORKERS="${BINDING_MAX_WORKERS:-12}"
  TIMEOUT="${BINDING_TIMEOUT:-1800}"
  TEMPERATURE="${BINDING_TEMPERATURE:-0.0}"

  ARGS=(--manifest "${MANIFEST}" --prompt "${PROMPT}" --output-dir "${OUTPUT_DIR}" --run-name "${RUN_NAME}" --model "${MODEL}" --location "${LOCATION}" --credentials "${CREDENTIALS}" --models "${MODELS}" --samples "${SAMPLES}" --max-workers "${MAX_WORKERS}" --timeout "${TIMEOUT}" --temperature "${TEMPERATURE}")
  if [[ -n "${BINDING_LIMIT:-}" ]]; then ARGS+=(--limit "${BINDING_LIMIT}"); fi
  if [[ "${BINDING_OVERWRITE:-0}" == "1" ]]; then ARGS+=(--overwrite); fi

  "${PYTHON}" binding_eval.py "${ARGS[@]}"
  LATEST_RUN="$(cat "${OUTPUT_DIR}/latest_run.txt")"
  echo "[done] binding latest: ${LATEST_RUN}"
}

cd "${MLLM_DIR}"
PYTHON_FOR_MANIFEST="${MLLM_PYTHON:-${DP_PYTHON:-${BINDING_PYTHON:-python}}}"
DEFAULT_MANIFEST_ROOT="${OUTPUT_DIR:-${DP_OUTPUT_DIR:-${BINDING_OUTPUT_DIR:-${MLLM_DIR}/outputs}}}"
MANIFEST="${MLLM_MANIFEST:-${DEFAULT_MANIFEST_ROOT}/inputs/mllm_${BOARD}_${RUN_TIMESTAMP}.json}"
export DP_MANIFEST="${DP_MANIFEST:-${MANIFEST}}"
export BINDING_MANIFEST="${BINDING_MANIFEST:-${MANIFEST}}"
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
"${PYTHON_FOR_MANIFEST}" build_mllm_manifest_from_xlsx.py "${BUILD_ARGS[@]}"

case "${METRIC}" in
  detail)
    run_detail
    ;;
  binding)
    run_binding
    ;;
  both|detail_binding|all)
    run_detail
    run_binding
    ;;
  *)
    echo "ERROR: METRIC must be detail, binding, both, detail_binding, or all; got '${METRIC}'" >&2
    exit 2
    ;;
esac
