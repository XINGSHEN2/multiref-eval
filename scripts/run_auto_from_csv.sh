#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "$#" -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  bash scripts/run_auto_from_csv.sh INPUT.csv [OUTPUT_DIR]

Automatic metric entrypoints:
  all
  or comma-separated metric names, for example:
  voice_timbre_similarity,visual_quality

All 14 metrics have backend entrypoints. Heavy/MLLM metrics may require model
weights, CUDA, ffmpeg, or Gemini/Vertex credentials; missing dependencies are
reported as status=error in per_sample.csv.

Environment:
  MULTIREF_COMPASS_METRICS=all
  MODELS=model1,model2
  MAX_WORKERS=4
  MULTIREF_COMPASS_GEMINI_CREDENTIALS=/path/to/service-account.json
  VOICE_TIMBRE_MIN_SPEECH_SEC=2.5
  VOICE_TIMBRE_DEVICE=auto|cpu|cuda
  MULTIREF_COMPASS_MODEL_CACHE=/path/to/cache

Legacy FINALEVAL_* variables are still accepted for compatibility.
USAGE
  exit 2
fi

INPUT_CSV="$1"
OUTPUT_DIR="${2:-${ROOT}/outputs/$(basename "${INPUT_CSV}" .csv)}"
METRICS="${MULTIREF_COMPASS_METRICS:-${FINALEVAL_METRICS:-all}}"
MODEL_CACHE="${MULTIREF_COMPASS_MODEL_CACHE:-${FINALEVAL_MODEL_CACHE:-${ROOT}/models}}"

if [[ -n "${MULTIREF_COMPASS_GEMINI_CREDENTIALS:-}" && -z "${FINALEVAL_GEMINI_CREDENTIALS:-}" ]]; then
  export FINALEVAL_GEMINI_CREDENTIALS="${MULTIREF_COMPASS_GEMINI_CREDENTIALS}"
fi
if [[ -n "${MULTIREF_COMPASS_MODEL_CACHE:-}" && -z "${FINALEVAL_MODEL_CACHE:-}" ]]; then
  export FINALEVAL_MODEL_CACHE="${MULTIREF_COMPASS_MODEL_CACHE}"
fi
if [[ -n "${MULTIREF_COMPASS_CACHE_ROOT:-}" && -z "${FINALEVAL_CACHE_ROOT:-}" ]]; then
  export FINALEVAL_CACHE_ROOT="${MULTIREF_COMPASS_CACHE_ROOT}"
fi

"${PYTHON_BIN}" -m finaleval.cli evaluate \
  --input "${INPUT_CSV}" \
  --output "${OUTPUT_DIR}" \
  --metrics "${METRICS}" \
  --model-cache "${MODEL_CACHE}" \
  --voice-timbre-min-speech-sec "${VOICE_TIMBRE_MIN_SPEECH_SEC:-2.5}" \
  --voice-timbre-device "${VOICE_TIMBRE_DEVICE:-auto}"
