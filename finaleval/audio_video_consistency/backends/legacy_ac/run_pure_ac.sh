#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINALEVAL_ROOT="${FINALEVAL_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

PYTHON_BIN="${PUREAC_PYTHON:-${AC_PYTHON:-python}}"
VSLS_PYTHON_BIN="${VSLS_PYTHON:-python}"
LATENTSYNC_DIR="${LATENTSYNC_DIR:-LatentSync}"
MODEL_PATH="${MODEL_PATH:-${LATENTSYNC_DIR}/checkpoints/auxiliary/syncnet_v2.model}"
DEVICE="${VSLS_DEVICE:-cuda}"
VAD_BACKEND="${VSLS_VAD_BACKEND:-silero}"
OUTPUT_ROOT="${PUREAC_OUTPUT_ROOT:-${SCRIPT_DIR}/outputs}"
METADATA_DIRS="${PUREAC_METADATA_DIRS:-${SCRIPT_DIR}/inputs/metadata}"

INPUT_CSV=""
METRICS="allmllm"
RUN_NAME="pureac"
MODELS="all"
SAMPLES="all"
MAX_WORKERS="${AC_MAX_WORKERS:-4}"
TIMEOUT="${AC_TIMEOUT:-300}"
EVAL_MODEL="${AC_EVAL_MODEL:-${IF_EVAL_MODEL:-${RC_EVAL_MODEL:-gemini-3.5-flash}}}"
LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
CREDENTIALS="${AC_GEMINI_CREDENTIALS:-${IF_GEMINI_CREDENTIALS:-${RC_GEMINI_CREDENTIALS:-${FINALEVAL_GEMINI_CREDENTIALS:-}}}}"
DRY_RUN="${AC_DRY_RUN:-0}"
OVERWRITE="0"
LIMIT=""
SKIP_MISSING="1"
VSLS_SUBPROCESS="0"
DELETE_CROPS="0"

usage() {
  cat <<'USAGE'
Usage:
  bash run_pure_ac.sh --input-csv CSV [options]

Required:
  --input-csv CSV

Metric selection:
  --metrics visual_event_sound_alignment
  --metrics source_correctness
  --metrics voice_timbre_similarity
  --metrics voice_timbre_embedding
  --metrics stable_speaking_frontal_face
  --metrics allmllm
  --metrics vsls_eval
  --metrics all
  Comma-separated combinations are allowed, for example:
  --metrics allmllm,vsls_eval

Options:
  --run-name NAME
  --models model1,model2
  --samples board1_001,board2_003
  --max-workers N
  --timeout SEC
  --eval-model MODEL
  --credentials PATH
  --location LOCATION
  --output-root DIR
  --metadata-dirs DIR1,DIR2,DIR3
  --limit N
  --dry-run
  --overwrite
  --keep-missing-videos
  --vsls-subprocess
  --delete-crops

Environment:
  PUREAC_PYTHON, VSLS_PYTHON, LATENTSYNC_DIR, MODEL_PATH, VSLS_DEVICE,
  VSLS_VAD_BACKEND, AC_GEMINI_CREDENTIALS, FINALEVAL_GEMINI_CREDENTIALS,
  AC_EVAL_MODEL, PUREAC_METADATA_DIRS,
  VOICE_TIMBRE_MIN_SPEECH_SEC, VOICE_TIMBRE_MODEL_DIR
USAGE
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s\n' "${value}"
}

contains_metric() {
  local haystack="$1"
  local needle="$2"
  [[ ",${haystack}," == *",${needle},"* ]]
}

normalize_metrics() {
  local raw="$1"
  local item
  local mllm=()
  RUN_VSLS="0"
  RUN_VOICE_EMBEDDING="0"
  IFS=',' read -ra metric_items <<< "${raw}"
  for item in "${metric_items[@]}"; do
    item="$(trim "${item}")"
    case "${item,,}" in
      all)
        mllm=("Visual Event-Sound Alignment" "Source Correctness")
        RUN_VSLS="1"
        ;;
      allmllm|mllm)
        mllm=("Visual Event-Sound Alignment" "Source Correctness")
        ;;
      stable_speaking_frontal_face|speaking_stable_frontal|stable_frontal|speaking_frontal)
        mllm+=("Stable Speaking Frontal Face")
        ;;
      visual_event_sound_alignment|visual-event-sound-alignment|av_sync)
        mllm+=("Visual Event-Sound Alignment")
        ;;
      source_correctness|source-correctness|source)
        mllm+=("Source Correctness")
        ;;
      voice_timbre_similarity|voice-timbre-similarity|voice_timbre|timbre_similarity|audio_timbre_similarity|voice_timbre_embedding|voice-timbre-embedding|voice_embedding)
        RUN_VOICE_EMBEDDING="1"
        ;;
      voice_timbre_similarity_mllm|voice-timbre-similarity-mllm|voice_timbre_mllm)
        mllm+=("Voice Timbre Similarity")
        ;;
      vsls_eval|vsls|syncnet)
        RUN_VSLS="1"
        ;;
      "")
        ;;
      *)
        echo "ERROR: unknown metric: ${item}" >&2
        usage >&2
        exit 2
        ;;
    esac
  done

  local deduped=()
  local seen=","
  for item in "${mllm[@]}"; do
    if [[ "${seen}" != *",${item},"* ]]; then
      deduped+=("${item}")
      seen+="${item},"
    fi
  done
  local IFS=','
  MLLM_DIMENSIONS="${deduped[*]}"
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --input-csv)
      INPUT_CSV="$2"
      shift 2
      ;;
    --metrics)
      METRICS="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --models)
      MODELS="$2"
      shift 2
      ;;
    --samples)
      SAMPLES="$2"
      shift 2
      ;;
    --max-workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --eval-model)
      EVAL_MODEL="$2"
      shift 2
      ;;
    --credentials)
      CREDENTIALS="$2"
      shift 2
      ;;
    --location)
      LOCATION="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --metadata-dirs)
      METADATA_DIRS="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    --overwrite)
      OVERWRITE="1"
      shift
      ;;
    --keep-missing-videos)
      SKIP_MISSING="0"
      shift
      ;;
    --vsls-subprocess)
      VSLS_SUBPROCESS="1"
      shift
      ;;
    --delete-crops)
      DELETE_CROPS="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${INPUT_CSV}" ]]; then
  echo "ERROR: --input-csv is required" >&2
  usage >&2
  exit 2
fi

normalize_metrics "${METRICS}"

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}_${RUN_TIMESTAMP}"
TASKS_JSON="${RUN_DIR}/ac_tasks.json"
mkdir -p "${RUN_DIR}"

if [[ "${RUN_VSLS}" == "1" && "${DELETE_CROPS}" == "1" ]]; then
  echo "[error] VSLS stable-frontal ranking needs crop videos. Remove --delete-crops." >&2
  exit 1
fi

if [[ "${AC_NO_PROXY:-0}" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
  unset ftp_proxy FTP_PROXY ws_proxy wss_proxy WS_PROXY WSS_PROXY
else
  PROXY_URL="${AC_PROXY_URL:-${IF_PROXY_URL:-${RC_PROXY_URL:-http://oversea-squid2.ko.txyun:11080}}}"
  export http_proxy="${PROXY_URL}"
  export https_proxy="${PROXY_URL}"
  export HTTP_PROXY="${PROXY_URL}"
  export HTTPS_PROXY="${PROXY_URL}"
  export no_proxy="${AC_NO_PROXY_LIST:-localhost,127.0.0.1,localaddress,localdomain.com}"
  export NO_PROXY="${no_proxy}"
fi

echo "[cfg] input csv: ${INPUT_CSV}"
echo "[cfg] metrics: ${METRICS}"
echo "[cfg] MLLM dimensions: ${MLLM_DIMENSIONS:-none}"
echo "[cfg] run VSLS: ${RUN_VSLS}"
echo "[cfg] run voice timbre embedding: ${RUN_VOICE_EMBEDDING}"
echo "[cfg] run dir: ${RUN_DIR}"
echo "[cfg] metadata dirs: ${METADATA_DIRS}"

MLLM_INPUT_CSV="${INPUT_CSV}"
VSLS_INPUT_CSV="${INPUT_CSV}"
if [[ ",${MLLM_DIMENSIONS}," == *",Stable Speaking Frontal Face,"* && "${PUREAC_STABLE_SPEECH_ONLY:-1}" == "1" ]]; then
  SPEECH_INPUT_CSV="${RUN_DIR}/speech_input.csv"
  "${PYTHON_BIN}" - "${INPUT_CSV}" "${SPEECH_INPUT_CSV}" <<'PY_FILTER_SPEECH'
from __future__ import annotations
import csv
import sys
from pathlib import Path

input_csv = Path(sys.argv[1]).expanduser().resolve()
output_csv = Path(sys.argv[2]).expanduser().resolve()
truthy = {"1", "true", "yes", "y", "是"}
with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    if not reader.fieldnames:
        raise SystemExit(f"CSV has no header: {input_csv}")
    rows = [row for row in reader]
    fieldnames = list(reader.fieldnames)
kept = [row for row in rows if str(row.get("has_speech", "")).strip().lower() in truthy]
output_csv.parent.mkdir(parents=True, exist_ok=True)
with output_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(kept)
print(f"[speech filter] kept {len(kept)} / {len(rows)} rows with has_speech=True -> {output_csv}")
PY_FILTER_SPEECH
  MLLM_INPUT_CSV="${SPEECH_INPUT_CSV}"
fi

if [[ -n "${MLLM_DIMENSIONS}" || "${RUN_VOICE_EMBEDDING}" == "1" ]]; then
  echo "[1/5] Building AC task JSON from CSV"
  BUILD_ARGS=(
    --input-csv "${MLLM_INPUT_CSV}"
    --output-json "${TASKS_JSON}"
    --metadata-dirs "${METADATA_DIRS}"
    --models "${MODELS}"
    --samples "${SAMPLES}"
  )
  if [[ "${SKIP_MISSING}" == "1" ]]; then
    BUILD_ARGS+=(--skip-missing-videos)
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/csv_to_ac_tasks.py" "${BUILD_ARGS[@]}"
else
  echo "[1/5] Skipping AC task JSON build"
fi

if [[ -n "${MLLM_DIMENSIONS}" ]]; then
  echo "[2/5] Running MLLM Audio Consistency"
  EVAL_ARGS=(
    --tasks "${TASKS_JSON}"
    --prompts-dir "${SCRIPT_DIR}/prompts"
    --output "${RUN_DIR}/ac_results.json"
    --models "${MODELS}"
    --samples "${SAMPLES}"
    --dimensions "${MLLM_DIMENSIONS}"
    --speech-lip-method mllm
    --max-workers "${MAX_WORKERS}"
    --timeout "${TIMEOUT}"
    --model "${EVAL_MODEL}"
    --location "${LOCATION}"
    --credentials "${CREDENTIALS}"
  )
  if [[ -n "${LIMIT}" ]]; then
    EVAL_ARGS+=(--limit "${LIMIT}")
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    EVAL_ARGS+=(--dry-run)
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    EVAL_ARGS+=(--overwrite)
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/eval_audio_consistency.py" "${EVAL_ARGS[@]}"

  echo "[3/5] Summarizing MLLM outputs"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_audio_consistency.py" --run-dir "${RUN_DIR}"
  mkdir -p "${RUN_DIR}/ranking_metadata"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/rank_audio_consistency.py" --run-dir "${RUN_DIR}" --input-dir "${RUN_DIR}/ranking_metadata"
  if [[ ",${MLLM_DIMENSIONS}," == *",Stable Speaking Frontal Face,"* && "${DRY_RUN}" != "1" ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/filter_stable_speaking_frontal_csv.py" \
      --input-csv "${MLLM_INPUT_CSV}" \
      --results-json "${RUN_DIR}/ac_results.json" \
      --output-csv "${RUN_DIR}/stable_speaking_frontal_input.csv" \
      --rejected-csv "${RUN_DIR}/stable_speaking_frontal_rejected.csv" \
      --min-score "${STABLE_SPEAKING_FRONTAL_MIN_SCORE:-4.0}"
    VSLS_INPUT_CSV="${RUN_DIR}/stable_speaking_frontal_input.csv"
  fi
else
  echo "[2/5] Skipping MLLM metrics"
fi

if [[ "${RUN_VOICE_EMBEDDING}" == "1" ]]; then
  echo "[4/5] Running voice timbre embedding metric"
  VOICE_TIMBRE_MODEL_DIR="${VOICE_TIMBRE_MODEL_DIR:-${SCRIPT_DIR}/pretrained_models/spkrec-ecapa-voxceleb}"
  VOICE_TIMBRE_MIN_SPEECH_SEC="${VOICE_TIMBRE_MIN_SPEECH_SEC:-2.5}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/voice_timbre_embedding_eval.py" \
    --tasks "${TASKS_JSON}" \
    --output-dir "${RUN_DIR}/voice_timbre_embedding" \
    --min-speech-sec "${VOICE_TIMBRE_MIN_SPEECH_SEC}" \
    --device "${VOICE_TIMBRE_DEVICE:-auto}" \
    --savedir "${VOICE_TIMBRE_MODEL_DIR}"
else
  echo "[4/5] Skipping voice timbre embedding metric"
fi

if [[ "${RUN_VSLS}" == "1" ]]; then
  echo "[5/5] Running VSLS automatic metric"
  FFMPEG_BIN="${FFMPEG_BIN:-$("${VSLS_PYTHON_BIN}" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUN_DIR}/.matplotlib_cache}"
  mkdir -p "${MPLCONFIGDIR}"
  VSLS_ARGS=(
    --input-csv "${VSLS_INPUT_CSV}"
    --output-dir "${RUN_DIR}/vsls_eval"
    --python-bin "${VSLS_PYTHON_BIN}"
    --latentsync-dir "${LATENTSYNC_DIR}"
    --model-path "${MODEL_PATH}"
    --device "${DEVICE}"
    --ffmpeg-bin "${FFMPEG_BIN}"
    --vad-backend "${VAD_BACKEND}"
    --video-column video_path
    --id-column index
  )
  if [[ -n "${LIMIT}" ]]; then
    VSLS_ARGS+=(--limit "${LIMIT}")
  fi
  if [[ "${VSLS_SUBPROCESS}" == "1" ]]; then
    VSLS_ARGS+=(--subprocess)
  fi
  if [[ "${DELETE_CROPS}" == "1" ]]; then
    VSLS_ARGS+=(--delete-crops)
  fi
  "${VSLS_PYTHON_BIN}" "${SCRIPT_DIR}/vsls_eval/batch_vsls_eval.py" "${VSLS_ARGS[@]}"
  "${VSLS_PYTHON_BIN}" "${SCRIPT_DIR}/vsls_eval/rank_vsls_results.py" \
    --results-dir "${RUN_DIR}/vsls_eval/results" \
    --metadata-csv "${VSLS_INPUT_CSV}" \
    --output-md "${RUN_DIR}/vsls_eval/ranking.md"
  "${VSLS_PYTHON_BIN}" "${SCRIPT_DIR}/vsls_eval/rank_stable_frontal_lse.py" \
    --results-dir "${RUN_DIR}/vsls_eval/results" \
    --metadata-csv "${VSLS_INPUT_CSV}" \
    --output-csv "${RUN_DIR}/vsls_eval/stable_frontal_combined_lsec_lsed_rankings_absoffset_lt10.csv" \
    --output-md "${RUN_DIR}/vsls_eval/stable_frontal_combined_lsec_lsed_rankings_absoffset_lt10.md" \
    --detail-csv "${RUN_DIR}/vsls_eval/stable_frontal_combined_lsec_lsed_detail_absoffset_lt10.csv"
else
  echo "[5/5] Skipping VSLS automatic metric"
fi

echo "[done] Run directory: ${RUN_DIR}"
if [[ -f "${RUN_DIR}/ac_summary.csv" ]]; then
  echo "[done] AC summary: ${RUN_DIR}/ac_summary.csv"
  echo "[done] AC per sample: ${RUN_DIR}/ac_per_sample.csv"
  echo "[done] AC ranking: ${RUN_DIR}/ac_ranking.md"
  if [[ -f "${RUN_DIR}/stable_speaking_frontal_input.csv" ]]; then
    echo "[done] Stable speaking frontal input CSV: ${RUN_DIR}/stable_speaking_frontal_input.csv"
  fi
fi
if [[ -f "${RUN_DIR}/vsls_eval/summary.csv" ]]; then
  echo "[done] VSLS summary: ${RUN_DIR}/vsls_eval/summary.csv"
  echo "[done] VSLS input CSV: ${VSLS_INPUT_CSV}"
  echo "[done] VSLS ranking: ${RUN_DIR}/vsls_eval/ranking.md"
  echo "[done] Stable frontal LSE ranking: ${RUN_DIR}/vsls_eval/stable_frontal_combined_lsec_lsed_rankings_absoffset_lt10.md"
fi
if [[ -f "${RUN_DIR}/voice_timbre_embedding/voice_timbre_embedding_summary.md" ]]; then
  echo "[done] Voice timbre embedding summary: ${RUN_DIR}/voice_timbre_embedding/voice_timbre_embedding_summary.md"
  echo "[done] Voice timbre embedding scores: ${RUN_DIR}/voice_timbre_embedding/voice_timbre_embedding_scores.csv"
fi
