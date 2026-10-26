#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash extract_audio.sh <input_dir> <output_dir>" >&2
  exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
mkdir -p "${OUTPUT_DIR}"

FFMPEG_BIN="${FFMPEG_BIN:-}"
if [[ -z "${FFMPEG_BIN}" ]]; then
  if command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG_BIN="$(command -v ffmpeg)"
  elif [[ -n "${PYTHON_BIN:-}" ]]; then
    FFMPEG_BIN="$("${PYTHON_BIN}" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
  fi
fi
if [[ -z "${FFMPEG_BIN}" || ! -x "${FFMPEG_BIN}" ]]; then
  echo "ERROR: ffmpeg not found. Install ffmpeg or imageio-ffmpeg in PYTHON_BIN env, or set FFMPEG_BIN." >&2
  exit 1
fi

shopt -s nullglob nocaseglob
count=0
extracted_count=0
for video in "${INPUT_DIR}"/*.{mp4,avi,mov,mkv,webm,m4v}; do
  [[ -f "${video}" ]] || continue
  name="$(basename "${video}")"
  stem="${name%.*}"
  output_file="${OUTPUT_DIR}/${stem}.wav"
  if ! "${FFMPEG_BIN}" -nostdin -hide_banner -loglevel error -y -i "${video}" -vn -ac 1 -ar 16000 "${output_file}"; then
    echo "WARN: failed to extract audio from ${video}; leaving audio missing" >&2
    rm -f "${output_file}"
  else
    extracted_count=$((extracted_count + 1))
  fi
  count=$((count + 1))
done
shopt -u nullglob nocaseglob

if [[ ${count} -eq 0 ]]; then
  echo "ERROR: no supported video files found in ${INPUT_DIR}" >&2
  exit 1
fi

if [[ ${extracted_count} -eq 0 ]]; then
  echo "WARN: no audio tracks extracted from ${INPUT_DIR}" >&2
fi
