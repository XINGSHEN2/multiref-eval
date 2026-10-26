#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
FINAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="${EVAL_ROOT:-$FINAL_ROOT}"
cd "$EVAL_ROOT"

export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"
export HF_HOME="${HF_HOME:-$PWD/ef_eval/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PWD/ef_eval/.cache/pip}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/ef_eval/.cache/matplotlib}"

PY="${PY:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
RUNNER="${RUNNER:-$FINAL_ROOT/eval_py/run_entity_fidelity_from_xlsx.py}"

XLSX="${XLSX:-}"
CSV="${CSV:-}"
BOARD="${BOARD:-all}"
OUT_DIR="${OUT_DIR:-$FINAL_ROOT/outputs/entity_fidelity}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SAMPLE_JSON_ROOT="${SAMPLE_JSON_ROOT:-generated_csv_testvideos}"
SOURCE_XLSX_ROOT="${SOURCE_XLSX_ROOT:-}"
VIDEO_REFERENCE_WORK_ROOT="${VIDEO_REFERENCE_WORK_ROOT:-}"
VIDEO_REFERENCE_FRAMES="${VIDEO_REFERENCE_FRAMES:-16}"
VIDEO_REFERENCE_COUNT="${VIDEO_REFERENCE_COUNT:-3}"
FORCE_VIDEO_REFERENCE_BUILD="${FORCE_VIDEO_REFERENCE_BUILD:-0}"

NUM_FRAMES="${NUM_FRAMES:-32}"
FACE_WEIGHT="${FACE_WEIGHT:-0.6}"
REFERENCE_TOP_K="${REFERENCE_TOP_K:-1}"
MIN_FRAME_EF="${MIN_FRAME_EF:-0.1}"
DEVICE="${DEVICE:-cuda}"
DETECTOR_DEVICE="${DETECTOR_DEVICE:-cuda:0}"
CARTOON_MIN_BODY_EF="${CARTOON_MIN_BODY_EF:-0.6}"
CARTOON_FACE_DETECTOR="${CARTOON_FACE_DETECTOR:-yolov8-anime-face}"
CARTOON_FACE_SCORE_THRESHOLD="${CARTOON_FACE_SCORE_THRESHOLD:-0.15}"
SAVE_FRAME_BOXES="${SAVE_FRAME_BOXES:-1}"
MAKE_FRAME_BOX_SHEET="${MAKE_FRAME_BOX_SHEET:-1}"
KEEP_FRAME_BOX_IMAGES="${KEEP_FRAME_BOX_IMAGES:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
LIMIT_ROWS="${LIMIT_ROWS:-}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"

FACE_ROOT="${FACE_ROOT:-ef_eval/model_assets/OpenS2V-Weight/face_extractor}"
CURRICULAR="${CURRICULAR:-ef_eval/model_assets/OpenS2V-Weight/glint360k_curricular_face_r101_backbone.bin}"
FACE_RECOGNITION_NAME="${FACE_RECOGNITION_NAME:-elasticface_r100}"
FACE_RECOGNITION_MODEL="${FACE_RECOGNITION_MODEL:-ef_eval/model_assets/face_recognizers/elasticface_r100.pth}"
SIGLIP="${SIGLIP:-ef_eval/hf_home/hub/models--google--siglip-base-patch16-224/snapshots/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed}"
DINOV2="${DINOV2:-ef_eval/hf_home/hub/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415}"
YOLO_CKPT="${YOLO_CKPT:-ef_eval/model_assets/OpenS2V-Weight/yolo_world_v2_l_image_prompt_adapter-719a7afb.pth}"
YOLO_REPO="${YOLO_REPO:-}"
YOLO_CFG="${YOLO_CFG:-$FINAL_ROOT/eval_py/yolo_world_clip_fix.py}"

args=(
  "$RUNNER"
)
if [[ -n "$CSV" ]]; then
  args+=(--csv "$CSV")
elif [[ -n "$XLSX" ]]; then
  args+=(--xlsx "$XLSX")
else
  echo "ERROR: set CSV=/path/to/input.csv or XLSX=/path/to/input.xlsx" >&2
  exit 2
fi
args+=(
  --run-timestamp "$RUN_TIMESTAMP"
  --sample-json-root "$SAMPLE_JSON_ROOT"
  --source-xlsx-root "$SOURCE_XLSX_ROOT"
  --video-reference-frames "$VIDEO_REFERENCE_FRAMES"
  --video-reference-count "$VIDEO_REFERENCE_COUNT"
  --num-frames "$NUM_FRAMES"
  --face-weight "$FACE_WEIGHT"
  --reference-top-k "$REFERENCE_TOP_K"
  --min-frame-ef "$MIN_FRAME_EF"
  --device "$DEVICE"
  --detector-device "$DETECTOR_DEVICE"
  --cartoon-min-body-ef "$CARTOON_MIN_BODY_EF"
  --cartoon-face-detector "$CARTOON_FACE_DETECTOR"
  --cartoon-face-score-threshold "$CARTOON_FACE_SCORE_THRESHOLD"
  --face-model-root "$FACE_ROOT"
  --curricular-checkpoint "$CURRICULAR"
  --face-recognition-model "$FACE_RECOGNITION_MODEL"
  --face-recognition-name "$FACE_RECOGNITION_NAME"
  --dinov2-model "$DINOV2"
  --siglip-model "$SIGLIP"
  --yoloworld-repo "$YOLO_REPO"
  --yoloworld-config "$YOLO_CFG"
  --yoloworld-checkpoint "$YOLO_CKPT"
  --save-frame-boxes "$SAVE_FRAME_BOXES"
  --make-frame-box-sheet "$MAKE_FRAME_BOX_SHEET"
  --keep-frame-box-images "$KEEP_FRAME_BOX_IMAGES"
  --skip-existing "$SKIP_EXISTING"
  --shard-index "$SHARD_INDEX"
  --shard-count "$SHARD_COUNT"
)
if [[ -n "$VIDEO_REFERENCE_WORK_ROOT" ]]; then
  args+=(--video-reference-work-root "$VIDEO_REFERENCE_WORK_ROOT")
fi
if [[ "$FORCE_VIDEO_REFERENCE_BUILD" == "1" ]]; then
  args+=(--force-video-reference-build)
fi

if [[ -n "$LIMIT_ROWS" ]]; then
  args+=(--limit-rows "$LIMIT_ROWS")
fi

run_one_board() {
  local board_name="$1"
  local board_out_dir="${OUT_DIR}"
  if [[ "${BOARD}" == "all" ]]; then
    board_out_dir="${OUT_DIR}/${board_name}"
  fi
  mkdir -p "${board_out_dir}"
  "$PY" "${args[@]}" --board "${board_name}" --out-dir "${board_out_dir}"
}

case "${BOARD}" in
  all|"")
    run_one_board board1
    run_one_board board2
    run_one_board board3
    ;;
  board1|board2|board3)
    run_one_board "${BOARD}"
    ;;
  *)
    echo "ERROR: BOARD must be all, board1, board2, or board3; got '${BOARD}'" >&2
    exit 2
    ;;
esac

SUMMARY_CSV="${OUT_DIR}/summary_${RUN_TIMESTAMP}.csv"
EF_SCORE_SUMMARY_MD="${EF_SCORE_SUMMARY_MD:-${OUT_DIR}/ef_score_summary.md}"
EF_SCORE_SUMMARY_CSV="${EF_SCORE_SUMMARY_CSV:-${OUT_DIR}/ef_score_summary.csv}"
PASTE_PER_SAMPLE="${PASTE_PER_SAMPLE:-$FINAL_ROOT/mllm/outputs/paste_artifact/latest/paste_artifact_per_sample.csv}"
PASTE_MODEL_ALIAS="${PASTE_MODEL_ALIAS:-happyhourse=happyhouse1.1}"
if [[ -f "$SUMMARY_CSV" && -f "$PASTE_PER_SAMPLE" ]]; then
  "$PY" "$FINAL_ROOT/summary_py/generate_ef_paste_summary.py" \
    --ef-summary-csv "$SUMMARY_CSV" \
    --paste-per-sample "$PASTE_PER_SAMPLE" \
    --paste-model-alias "$PASTE_MODEL_ALIAS" \
    --fallback-to-model-paste-mean \
    --output-md "$EF_SCORE_SUMMARY_MD" \
    --output-csv "$EF_SCORE_SUMMARY_CSV"
  echo "EF score summary: $EF_SCORE_SUMMARY_MD"
else
  echo "WARN: skip EF score summary; missing SUMMARY_CSV=$SUMMARY_CSV or PASTE_PER_SAMPLE=$PASTE_PER_SAMPLE" >&2
fi
