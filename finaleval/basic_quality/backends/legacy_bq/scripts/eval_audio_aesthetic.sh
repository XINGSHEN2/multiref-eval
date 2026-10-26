#!/usr/bin/env bash
set -euo pipefail
SETUP_ONLY=0
if [[ ${1:-} == --setup-only ]]; then
  SETUP_ONLY=1
  shift
fi
source $(cd $(dirname ${BASH_SOURCE[0]}) && pwd)/common.sh
ensure_cache_layout
ensure_conda
ensure_audiobox_env
if [[ ${SETUP_ONLY} -eq 1 ]]; then
  echo ${BQ_CORE_ENV} ready
  exit 0
fi
INPUT_DIR=$(resolve_path ${1:-input})
OUTPUT_DIR=$(resolve_path ${2:-Output})
AUDIO_DIR=${OUTPUT_DIR}/audio_wav
LOCAL_AUDIOBOX_CKPT=${CODE_ROOT}/weights/audiobox-aesthetics/checkpoint.pt
AUDIOBOX_CKPT=${LOCAL_AUDIOBOX_CKPT}
if [[ ! -s "${AUDIOBOX_CKPT}" ]]; then
  AUDIOBOX_CKPT=${CACHE_ROOT}/weights/audiobox-aesthetics/checkpoint.pt
fi
require_dir ${INPUT_DIR}
require_video_files ${INPUT_DIR}
mkdir -p ${OUTPUT_DIR}
bash ${CODE_ROOT}/scripts/extract_audio.sh ${INPUT_DIR} ${AUDIO_DIR}
if [[ -n "$(find "${AUDIO_DIR}" -maxdepth 1 -type f \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.flac' -o -iname '*.m4a' -o -iname '*.ogg' \) -print -quit)" ]]; then
  conda_run_in ${BQ_CORE_ENV} python ${CODE_ROOT}/scripts/run_audiobox_batch.py --audio_dir ${AUDIO_DIR} --output ${OUTPUT_DIR}/audio_aesthetic.json --ckpt ${AUDIOBOX_CKPT}
else
  python - <<PY
import json
from pathlib import Path
out = Path("${OUTPUT_DIR}") / "audio_aesthetic.json"
out.write_text(json.dumps({"metric": "audiobox_aesthetics", "summary": {}, "results": []}, indent=2) + "\\n", encoding="utf-8")
print(f"Saved empty AudioBox results to: {out}")
PY
fi
