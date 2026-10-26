#!/usr/bin/env bash
set -euo pipefail

WORK="${WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BQ_ROOT="${BQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONDA_DIR="${CONDA_DIR:-${WORK}/.conda}"
CONDA_ENV="${CONDA_ENV:-py310}"
CONDA_CACHE="${BQ_CONDA_ROOT:-${WORK}/.cache/basic_quality_conda}"
BQ_CACHE="${BQ_CACHE_ROOT:-${WORK}/.cache/basic_quality_cache}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TARGET="${1:-${BQ_TARGET:-custom}}"

case "${TARGET}" in
  gemini_subset_251|gemini)
    INPUT_CSV="${INPUT_CSV:-${BQ_ROOT}/gemini_subset_251_manifest.csv}"
    MODELS="${MODELS:-gemini}"
    EXPECTED_SAMPLES_PER_MODEL="${EXPECTED_SAMPLES_PER_MODEL:-251}"
    RUN_NAME="${RUN_NAME:-gemini_subset_251_$(date +%Y%m%d_%H%M%S)}"
    ;;
  all)
    INPUT_CSV="${INPUT_CSV:-${BQ_ROOT}/inputs/wan21_skyreels_new_306_manifest.csv}"
    MODELS="${MODELS:-wan2.1,skyreels-new}"
    EXPECTED_SAMPLES_PER_MODEL="${EXPECTED_SAMPLES_PER_MODEL:-306}"
    RUN_NAME="${RUN_NAME:-wan21_skyreels_new_306_basic_quality}"
    ;;
  board1)
    INPUT_CSV="${INPUT_CSV:-${BQ_ROOT}/inputs/wan21_skyreels_new_board1_manifest.csv}"
    MODELS="${MODELS:-wan2.1,skyreels-new}"
    EXPECTED_SAMPLES_PER_MODEL="${EXPECTED_SAMPLES_PER_MODEL:-106}"
    RUN_NAME="${RUN_NAME:-wan21_skyreels_new_board1_basic_quality}"
    ;;
  board2)
    INPUT_CSV="${INPUT_CSV:-${BQ_ROOT}/inputs/wan21_skyreels_new_board2_manifest.csv}"
    MODELS="${MODELS:-wan2.1,skyreels-new}"
    EXPECTED_SAMPLES_PER_MODEL="${EXPECTED_SAMPLES_PER_MODEL:-100}"
    RUN_NAME="${RUN_NAME:-wan21_skyreels_new_board2_basic_quality}"
    ;;
  board3)
    INPUT_CSV="${INPUT_CSV:-${BQ_ROOT}/inputs/wan21_skyreels_new_board3_manifest.csv}"
    MODELS="${MODELS:-wan2.1,skyreels-new}"
    EXPECTED_SAMPLES_PER_MODEL="${EXPECTED_SAMPLES_PER_MODEL:-100}"
    RUN_NAME="${RUN_NAME:-wan21_skyreels_new_board3_basic_quality}"
    ;;
  custom|csv|*)
    INPUT_CSV="${INPUT_CSV:-${BQ_ROOT}/inputs/input.csv}"
    SAFE_TARGET="$(echo "${TARGET}" | tr -cs 'A-Za-z0-9_.-' '_' | sed 's/^_*//; s/_*$//')"
    SAFE_TARGET="${SAFE_TARGET:-custom}"
    RUN_NAME="${RUN_NAME:-${SAFE_TARGET}_basic_quality_$(date +%Y%m%d_%H%M%S)}"
    ;;
esac

OUT="${OUT:-${BQ_ROOT}/run_outputs/${RUN_NAME}}"
AUDIO_TOOL="${AUDIO_TOOL:-audiobox}"
NUM_TEMPORAL_FRAMES="${NUM_TEMPORAL_FRAMES:-32}"

mkdir -p "${OUT}" "${CONDA_CACHE}" "${BQ_CACHE}" /tmp

if [[ ! -x "${CONDA_DIR}/bin/conda" ]]; then
  echo "ERROR: conda not found at ${CONDA_DIR}/bin/conda" >&2
  echo "Set CONDA_DIR=/path/to/miniconda if your mounted conda is elsewhere." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-${CONDA_DIR}/envs/${CONDA_ENV}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: python not found at ${PYTHON_BIN}" >&2
  echo "Set CONDA_ENV=your_env_name or PYTHON_BIN=/path/to/python." >&2
  exit 1
fi

if [[ -z "${MODELS:-}" || -z "${EXPECTED_SAMPLES_PER_MODEL:-}" ]]; then
  if [[ ! -f "${INPUT_CSV}" ]]; then
    echo "ERROR: INPUT_CSV not found: ${INPUT_CSV}" >&2
    exit 1
  fi
  INFER_OUTPUT="$(${PYTHON_BIN} - "${INPUT_CSV}" <<'INFERPY'
import csv
import sys
from collections import Counter
from pathlib import Path
path = Path(sys.argv[1])
with path.open('r', encoding='utf-8-sig', newline='') as handle:
    rows = list(csv.DictReader(handle))
counts = Counter((row.get('model') or row.get('model_name') or '').strip() for row in rows)
counts.pop('', None)
models = sorted(counts)
print(','.join(models))
values = sorted(set(counts.values()))
print(values[0] if len(values) == 1 else '')
print(';'.join(f'{model}:{counts[model]}' for model in models))
INFERPY
)"
  INFER_MODELS="$(printf '%s
' "${INFER_OUTPUT}" | sed -n '1p')"
  INFER_EXPECTED="$(printf '%s
' "${INFER_OUTPUT}" | sed -n '2p')"
  INFER_COUNTS="$(printf '%s
' "${INFER_OUTPUT}" | sed -n '3p')"
  MODELS="${MODELS:-${INFER_MODELS}}"
  if [[ -z "${EXPECTED_SAMPLES_PER_MODEL:-}" ]]; then
    if [[ -n "${INFER_EXPECTED}" ]]; then
      EXPECTED_SAMPLES_PER_MODEL="${INFER_EXPECTED}"
    else
      EXPECTED_SAMPLES_PER_MODEL="0"
      NO_DEFAULT_VALIDATION="${NO_DEFAULT_VALIDATION:-1}"
    fi
  fi
  echo "[inferred counts] ${INFER_COUNTS}"
fi

export BQ_CONDA_EXE="${CONDA_DIR}/bin/conda"
export BQ_CONDA_ROOT="${CONDA_CACHE}"
export CONDA_ENVS_PATH="${CONDA_CACHE}/envs"
export CONDA_PKGS_DIRS="${CONDA_CACHE}/pkgs"
export BQ_CACHE_ROOT="${BQ_CACHE}"
export HF_HOME="${BQ_CACHE}/huggingface"
export TORCH_HOME="${BQ_CACHE}/torch"
export XDG_CACHE_HOME="${BQ_CACHE}/xdg"
export PIP_CACHE_DIR="${BQ_CACHE}/pip"
export TMPDIR="${BQ_CACHE}/tmp"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHON_BIN
export BQ_USE_ACTIVE_ENV=1

mkdir -p \
  "${CONDA_ENVS_PATH}" \
  "${CONDA_PKGS_DIRS}" \
  "${HF_HOME}" \
  "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${TMPDIR}"

cd "${WORK}"

echo "[target] ${TARGET}"
echo "[input] ${INPUT_CSV}"
echo "[models] ${MODELS}"
echo "[expected samples/model] ${EXPECTED_SAMPLES_PER_MODEL}"
echo "[output] ${OUT}"
echo "[python] ${PYTHON_BIN}"
echo "[cuda] ${CUDA_VISIBLE_DEVICES}"
if [[ "${BQ_PRINT_CONFIG_ONLY:-0}" == "1" ]]; then
  echo "[print config only] 1"
  exit 0
fi

BQ_ARGS=(
  --input-csv "${INPUT_CSV}"
  --output-dir "${OUT}"
  --models "${MODELS}"
  --expected-samples-per-model "${EXPECTED_SAMPLES_PER_MODEL}"
  --audio-tool "${AUDIO_TOOL}"
  --num-temporal-frames "${NUM_TEMPORAL_FRAMES}"
  --skip-existing
)
if [[ "${NO_DEFAULT_VALIDATION:-0}" == "1" ]]; then
  BQ_ARGS+=(--no-default-validation)
fi

"${PYTHON_BIN}" "${BQ_ROOT}/run_basic_quality_from_csv.py" "${BQ_ARGS[@]}"

"${PYTHON_BIN}" "${BQ_ROOT}/generate_basic_quality_board123_ranking.py" \
  --run-dir "${OUT}"

echo "Done. Outputs in ${OUT}"
