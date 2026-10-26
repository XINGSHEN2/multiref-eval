#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINALEVAL_ROOT="${FINALEVAL_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
IF_ROOT="${IF_ROOT:-${SCRIPT_DIR}}"

if [[ "$#" -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  bash run_if_full_from_table.sh INPUT.xlsx|INPUT.csv [all|board1 board2 board3 board4]
  # default boards: board2 board3

Input table format:
  One row per (sample_id, model_name, video_path). Required columns:
    sample_id, board, prompt, model_name, video_path
  Optional columns:
    source_sample_id, chinese_prompt, has_sound, has_music, has_speech,
    is_multi_shot, image1_path, image1_type, image1_description, ...

Main env vars:
  PYTHON_BIN=/path/to/python
  MODELS=gemini,seedance,kling,vidu,wan
  IF_DIMENSION=all|visual|audio|temporal|speech
  IF_GEMINI_VIDEO_FPS=4
  IF_FINAL_REUSE_CHECKLISTS_ROOT=/path/to/existing/checklists/root
  IF_FINAL_SKIP_MISSING_CHECKLISTS=0|1  # default 1; skip samples absent from reused checklist files
  IF_REGENERATE_CHECKLISTS=0|1  # 1 disables reuse and regenerates checklists with Gemini/rule prompts
  IF_EVAL_MODEL=gemini-3.1-pro-preview
  IF_CHECKLIST_MODEL=gemini-3.1-pro-preview
  SPEECH_EVAL_MODEL=gemini-3.1-pro-preview
  IF_GEMINI_CREDENTIALS=/path/to/gemini-service-account.json
  GOOGLE_CLOUD_LOCATION=global
  MAX_WORKERS=12
USAGE
  exit 2
fi

INPUT_TABLE="$1"
shift

if [[ "$#" -gt 0 ]]; then
  BOARDS=("$@")
else
  BOARDS=(board2 board3)
fi

RUN_BOARDS=()
for board in "${BOARDS[@]}"; do
  case "${board}" in
    all)
      RUN_BOARDS=(board1 board2 board3 board4)
      break
      ;;
    board1|board2|board3|board4)
      RUN_BOARDS+=("${board}")
      ;;
    *)
      echo "ERROR: unsupported board '${board}'; expected all, board1, board2, board3, or board4" >&2
      exit 2
      ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS="${MODELS:-gemini,seedance,kling,vidu,wan}"
MAX_WORKERS="${MAX_WORKERS:-12}"
DIMENSION="${IF_DIMENSION:-visual}"
# audio temporal speech visual all
VIDEO_FPS="${IF_GEMINI_VIDEO_FPS:-4}"
EVAL_MODEL="${IF_EVAL_MODEL:-gemini-3.1-pro-preview}"
CHECKLIST_MODEL="${IF_CHECKLIST_MODEL:-${EVAL_MODEL}}"
SPEECH_MODEL="${SPEECH_EVAL_MODEL:-${EVAL_MODEL}}"
CREDENTIALS="${IF_GEMINI_CREDENTIALS:-${RC_GEMINI_CREDENTIALS:-${FINALEVAL_GEMINI_CREDENTIALS:-}}}"
LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
RUN_STEM="${IF_RUN_STEM:-$(basename "${INPUT_TABLE}")}"
RUN_STEM="${RUN_STEM%.*}"

OUTPUT_DIR="${IF_FINAL_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/${RUN_STEM}}"
INPUTS_DIR="${IF_FINAL_INPUTS_DIR:-${OUTPUT_DIR}/inputs}"
METADATA_ROOT="${IF_FINAL_METADATA_ROOT:-${INPUTS_DIR}/${RUN_STEM}_metadata}"
COMBINED_OUTPUT_DIR="${IF_FINAL_COMBINED_OUTPUT_DIR:-${OUTPUT_DIR}/all sample}"
WORK_DIR="${IF_FINAL_WORK_DIR:-${OUTPUT_DIR}/work}"
LONG_CSV="${WORK_DIR}/${RUN_STEM}_long.csv"
MARKER="${WORK_DIR}/run_started.marker"

VISUAL_CHECKLIST_PROMPT="${IF_FINAL_CHECKLIST_PROMPT:-${SCRIPT_DIR}/if_judge_prompts_v3/visual_checklist_generation.md}"
NONVISUAL_CHECKLIST_PROMPT="${IF_FINAL_NONVISUAL_CHECKLIST_PROMPT:-${SCRIPT_DIR}/if_judge_prompts_v3/if_checklist_generation_prompt_v2.md}"
DEFAULT_REUSE_CHECKLISTS_ROOT="${SCRIPT_DIR}/inputs/final_checklists"
REUSE_CHECKLISTS_ROOT="${IF_FINAL_REUSE_CHECKLISTS_ROOT:-${DEFAULT_REUSE_CHECKLISTS_ROOT}}"
if [[ "${IF_REGENERATE_CHECKLISTS:-0}" == "1" ]]; then
  REUSE_CHECKLISTS_ROOT=""
fi
VISUAL_JUDGE_PROMPT="${VISUAL_JUDGE_PROMPT:-${SCRIPT_DIR}/if_judge_prompts_v3/visual_task_following.md}"
AUDIO_JUDGE_PROMPT="${AUDIO_JUDGE_PROMPT:-${SCRIPT_DIR}/if_judge_prompts_v3/audio_task_following.md}"
TEMPORAL_JUDGE_PROMPT="${TEMPORAL_JUDGE_PROMPT:-${SCRIPT_DIR}/if_judge_prompts_v3/temporal_order_following.md}"
SPEECH_JUDGE_PROMPT="${IF_60_SPEECH_PROMPT:-${SPEECH_PROMPT:-${SCRIPT_DIR}/if_judge_prompts_v3/speech_accuracy.md}}"

mkdir -p "${WORK_DIR}" "${INPUTS_DIR}" "${OUTPUT_DIR}" "${COMBINED_OUTPUT_DIR}"
: > "${MARKER}"

echo "============================================================"
echo "[input] ${INPUT_TABLE}"
echo "[boards] ${BOARDS[*]}"
echo "[run boards] ${RUN_BOARDS[*]}"
echo "[models] ${MODELS}"
echo "[dimension] ${DIMENSION}"
echo "[video fps] ${VIDEO_FPS}"
echo "[python] ${PYTHON_BIN}"
echo "[eval model] ${EVAL_MODEL}"
echo "[checklist model] ${CHECKLIST_MODEL}"
echo "[speech model] ${SPEECH_MODEL}"
if [[ -n "${CREDENTIALS}" ]]; then
  echo "[credentials] configured"
else
  echo "[credentials] not configured"
fi
echo "[location] ${LOCATION}"
echo "[inputs] ${INPUTS_DIR}"
echo "[outputs] ${OUTPUT_DIR}"
echo "[combined] ${COMBINED_OUTPUT_DIR}"
echo "[reuse checklists] ${REUSE_CHECKLISTS_ROOT:-disabled; regenerate checklists}"
echo "[checklist prompt visual] ${VISUAL_CHECKLIST_PROMPT}"
echo "[checklist prompt audio/temporal/speech] ${NONVISUAL_CHECKLIST_PROMPT}"
echo "[judge prompt visual] ${VISUAL_JUDGE_PROMPT}"
echo "[judge prompt audio] ${AUDIO_JUDGE_PROMPT}"
echo "[judge prompt temporal] ${TEMPORAL_JUDGE_PROMPT}"
echo "[judge prompt speech] ${SPEECH_JUDGE_PROMPT}"
echo "============================================================"

"${PYTHON_BIN}" - "${INPUT_TABLE}" "${LONG_CSV}" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

input_path = Path(sys.argv[1]).expanduser().resolve()
output_csv = Path(sys.argv[2]).expanduser().resolve()
output_csv.parent.mkdir(parents=True, exist_ok=True)

required = {"sample_id", "model_name", "video_path", "prompt"}

def normalize_header(value: object) -> str:
    return str(value or "").strip()

if input_path.suffix.lower() in {".xlsx", ".xlsm"}:
    from openpyxl import load_workbook

    wb = load_workbook(input_path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        raise SystemExit(f"empty xlsx: {input_path}")
    headers = [normalize_header(value) for value in rows[0]]
    records = []
    for values in rows[1:]:
        row = {headers[idx]: "" if value is None else str(value) for idx, value in enumerate(values) if idx < len(headers)}
        if any(str(value).strip() for value in row.values()):
            records.append(row)
elif input_path.suffix.lower() == ".csv":
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        records = [{key: (value or "") for key, value in row.items() if key is not None} for row in reader]
else:
    raise SystemExit("input must be .xlsx, .xlsm, or .csv")

normalized_records = []
fieldnames = []
for row in records:
    out = {}
    for key, value in row.items():
        out[key] = value
        if key not in fieldnames:
            fieldnames.append(key)
    normalized_records.append(out)

missing = sorted(required - set(fieldnames))
if missing:
    raise SystemExit(f"missing required columns: {', '.join(missing)}")
if "board" not in fieldnames:
    fieldnames.append("board")
for row in normalized_records:
    if not str(row.get("board", "")).strip():
        sample_id = str(row.get("sample_id", ""))
        row["board"] = sample_id.split("_", 1)[0] if "_" in sample_id else ""

with output_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(normalized_records)
print(output_csv)
PY

BOARDS_CSV="$(IFS=,; echo "${BOARDS[*]}")"

echo "[1/4] Preparing IF tasks/checklists"
PREPARE_EXTRA_ARGS=()
if [[ "${IF_FINAL_SKIP_MISSING_VIDEOS:-0}" == "1" ]]; then
  PREPARE_EXTRA_ARGS+=(--skip-missing-videos)
fi
if [[ -n "${REUSE_CHECKLISTS_ROOT}" ]]; then
  PREPARE_EXTRA_ARGS+=(--reuse-checklists-root "${REUSE_CHECKLISTS_ROOT}")
  if [[ "${IF_FINAL_SKIP_MISSING_CHECKLISTS:-1}" == "1" ]]; then
    PREPARE_EXTRA_ARGS+=(--skip-missing-checklists)
  fi
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_if_inputs_from_csv.py" \
  --csv "${LONG_CSV}" \
  --output-root "${INPUTS_DIR}" \
  --metadata-root "${METADATA_ROOT}" \
  --models "${MODELS}" \
  --boards "${BOARDS_CSV}" \
  --checklist-source "${IF_FINAL_CHECKLIST_SOURCE:-gemini}" \
  --checklist-model "${CHECKLIST_MODEL}" \
  --checklist-prompt "${VISUAL_CHECKLIST_PROMPT}" \
  --nonvisual-checklist-source "${IF_FINAL_NONVISUAL_CHECKLIST_SOURCE:-gemini}" \
  --nonvisual-checklist-prompt "${NONVISUAL_CHECKLIST_PROMPT}" \
  --checklist-credentials "${CREDENTIALS}" \
  --checklist-location "${LOCATION}" \
  --checklist-timeout "${IF_CHECKLIST_TIMEOUT:-600}" \
  "${PREPARE_EXTRA_ARGS[@]}"

echo "[2/4] Running IF dimension(s): ${DIMENSION}"

run_mllm_dimension() {
  local board="$1"
  local dimension_key="$2"
  local dimension_name=""
  local judge_prompt=""
  local task_path="${INPUTS_DIR}/if_${board}_tasks.json"
  local run_name=""
  local run_dir=""
  local output_path=""

  case "${dimension_key}" in
    visual)
      dimension_name="Visual Task Following"
      judge_prompt="${VISUAL_JUDGE_PROMPT}"
      run_name="if_${board}_visual"
      ;;
    audio)
      dimension_name="Audio Task Following"
      judge_prompt="${AUDIO_JUDGE_PROMPT}"
      run_name="if_${board}_audio"
      ;;
    temporal)
      dimension_name="Temporal Order Following"
      judge_prompt="${TEMPORAL_JUDGE_PROMPT}"
      run_name="if_${board}_temporal"
      ;;
    *)
      echo "ERROR: unsupported MLLM dimension: ${dimension_key}" >&2
      exit 2
      ;;
  esac

  if [[ ! -f "${task_path}" ]]; then
    echo "[skip] tasks file not found for ${board} ${dimension_key}: ${task_path}" >&2
    return 0
  fi

  run_dir="${OUTPUT_DIR}/${run_name}"
  output_path="${run_dir}/if_results.json"
  mkdir -p "${run_dir}"

  echo "------------------------------------------------------------"
  echo "[run] ${board} ${dimension_key}"
  echo "[tasks] ${task_path}"
  echo "[prompt] ${judge_prompt}"
  echo "[output] ${output_path}"
  echo "------------------------------------------------------------"

  EVAL_ARGS=(
    "${SCRIPT_DIR}/eval_instruction_following.py"
    --tasks "${task_path}"
    --prompt "${judge_prompt}"
    --output "${output_path}"
    --models "${MODELS}"
    --samples all
    --max-workers "${MAX_WORKERS}"
    --timeout "${IF_TIMEOUT:-300}"
    --model "${EVAL_MODEL}"
    --location "${LOCATION}"
    --credentials "${CREDENTIALS}"
    --dimensions "${dimension_name}"
  )
  if [[ "${IF_DRY_RUN:-0}" == "1" ]]; then
    EVAL_ARGS+=(--dry-run)
  fi
  if [[ "${IF_OVERWRITE:-0}" == "1" ]]; then
    EVAL_ARGS+=(--overwrite)
  fi

  IF_GEMINI_VIDEO_FPS="${VIDEO_FPS}" "${PYTHON_BIN}" "${EVAL_ARGS[@]}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_instruction_following.py" --run-dir "${run_dir}"

  if [[ "${dimension_key}" == "visual" ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/score_visual_checklist_final.py" --run-dir "${run_dir}"
  fi

  "${PYTHON_BIN}" "${SCRIPT_DIR}/rank_instruction_following.py" --run-dir "${run_dir}"
}

run_speech_dimension() {
  local board="$1"
  local task_path="${INPUTS_DIR}/if_${board}_speech_tasks.json"
  local run_name="if_${board}_speech"
  local run_dir="${OUTPUT_DIR}/${run_name}"
  local output_path="${run_dir}/speech_results.json"

  if [[ ! -f "${task_path}" ]]; then
    task_path="${INPUTS_DIR}/if_${board}_tasks.json"
  fi
  if [[ ! -f "${task_path}" ]]; then
    echo "[skip] speech tasks file not found for ${board}: ${task_path}" >&2
    return 0
  fi

  mkdir -p "${run_dir}"

  echo "------------------------------------------------------------"
  echo "[run] ${board} speech"
  echo "[tasks] ${task_path}"
  echo "[prompt] ${SPEECH_JUDGE_PROMPT}"
  echo "[output] ${output_path}"
  echo "------------------------------------------------------------"

  SPEECH_ARGS=(
    "${SCRIPT_DIR}/eval_speech_content_mllm.py"
    --tasks "${task_path}"
    --prompt "${SPEECH_JUDGE_PROMPT}"
    --output "${output_path}"
    --models "${MODELS}"
    --samples all
    --max-workers "${MAX_WORKERS}"
    --timeout "${SPEECH_TIMEOUT:-${IF_TIMEOUT:-300}}"
    --model "${SPEECH_MODEL}"
    --location "${LOCATION}"
    --credentials "${CREDENTIALS}"
  )
  if [[ "${IF_DRY_RUN:-0}" == "1" || "${SPEECH_DRY_RUN:-0}" == "1" ]]; then
    SPEECH_ARGS+=(--dry-run)
  fi
  if [[ "${IF_OVERWRITE:-0}" == "1" || "${SPEECH_OVERWRITE:-0}" == "1" ]]; then
    SPEECH_ARGS+=(--overwrite)
  fi

  IF_GEMINI_VIDEO_FPS="${VIDEO_FPS}" "${PYTHON_BIN}" "${SPEECH_ARGS[@]}"
}

normalize_dimension() {
  case "${1,,}" in
    all) echo "all" ;;
    visual|"visual task following") echo "visual" ;;
    audio|"audio task following") echo "audio" ;;
    temporal|"temporal order following") echo "temporal" ;;
    speech|speech_content|speech-content|"speech content accuracy") echo "speech" ;;
    *)
      echo "ERROR: unsupported IF_DIMENSION=${1}. Use all, visual, audio, temporal, or speech." >&2
      exit 2
      ;;
  esac
}

DIMENSION_NORM="$(normalize_dimension "${DIMENSION}")"
for board in "${RUN_BOARDS[@]}"; do
  case "${DIMENSION_NORM}" in
    all)
      run_mllm_dimension "${board}" visual
      run_mllm_dimension "${board}" audio
      run_mllm_dimension "${board}" temporal
      run_speech_dimension "${board}"
      ;;
    visual|audio|temporal)
      run_mllm_dimension "${board}" "${DIMENSION_NORM}"
      ;;
    speech)
      run_speech_dimension "${board}"
      ;;
  esac
done

collect_run_dirs() {
  local suffix="$1"
  local required_csv="$2"
  local board
  local run_dir

  for board in "${RUN_BOARDS[@]}"; do
    run_dir="${OUTPUT_DIR}/if_${board}_${suffix}"
    if [[ -d "${run_dir}" && -f "${run_dir}/${required_csv}" ]]; then
      printf '%s\n' "${run_dir}"
    fi
  done
}

mapfile -t VISUAL_RUN_DIRS < <(collect_run_dirs visual if_per_sample.csv)
mapfile -t AUDIO_RUN_DIRS < <(collect_run_dirs audio if_per_sample.csv)
mapfile -t TEMPORAL_RUN_DIRS < <(collect_run_dirs temporal if_per_sample.csv)
mapfile -t SPEECH_RUN_DIRS < <(collect_run_dirs speech speech_per_sample.csv)

IF_RUN_DIRS=("${VISUAL_RUN_DIRS[@]}" "${AUDIO_RUN_DIRS[@]}" "${TEMPORAL_RUN_DIRS[@]}")

echo "[4/4] Combining ranking"
COMBINE_ARGS=()
for run_dir in "${IF_RUN_DIRS[@]}"; do
  [[ -n "${run_dir}" ]] && COMBINE_ARGS+=(--if-run-dir "${run_dir}")
done
for run_dir in "${SPEECH_RUN_DIRS[@]}"; do
  [[ -n "${run_dir}" ]] && COMBINE_ARGS+=(--speech-run-dir "${run_dir}")
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/combine_if_all_results.py" \
  "${COMBINE_ARGS[@]}" \
  --output-dir "${COMBINED_OUTPUT_DIR}" \
  --prefix "weighted_4task" \
  --ranking-basename "weighted_4task"

"${PYTHON_BIN}" - "${COMBINED_OUTPUT_DIR}/weighted_4task.csv" "${COMBINED_OUTPUT_DIR}/weighted_4task.md" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])
with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
for row in rows:
    row["model"] = row.get("model_name") or row.get("model") or ""

metric_map = [
    ("Instruction Following Ranking", "Instruction Following", "IF All"),
    ("Visual Ranking", "Visual Task Following", "Visual"),
    ("Audio Ranking", "Audio Task Following", "Audio"),
    ("Temporal Ranking", "Temporal Order Following", "Temporal"),
    ("Speech Ranking", "Speech Content Accuracy", "Speech"),
]

def as_float(value):
    try:
        return float(value)
    except Exception:
        return None

def ranked(metric):
    data = []
    for row in rows:
        value = as_float(row.get(metric))
        if value is not None:
            data.append((row["model"], value, row.get("count", "")))
    data.sort(key=lambda item: (-item[1], item[0]))
    out = []
    prev = None
    prev_rank = 0
    for idx, item in enumerate(data, 1):
        rank = prev_rank if prev is not None and item[1] == prev else idx
        out.append((rank, *item))
        prev = item[1]
        prev_rank = rank
    return out

base = md_path.read_text(encoding="utf-8") if md_path.exists() else "# IF Ranking\n"
base = base.split("\n## Metric Sub-Rankings", 1)[0].rstrip()
lines = [base, "", "## Metric Sub-Rankings", ""]
for title, metric, label in metric_map:
    lines.extend([f"### {title}", "", f"| Rank | Model | {label} | Count |", "|---:|---|---:|---:|"])
    for rank, model, value, count in ranked(metric):
        lines.append(f"| {rank} | {model} | {value:.4f} | {count} |")
    lines.append("")
md_path.write_text("\n".join(lines), encoding="utf-8")
PY

echo "[done] combined ranking md: ${COMBINED_OUTPUT_DIR}/weighted_4task.md"
echo "[done] combined ranking csv: ${COMBINED_OUTPUT_DIR}/weighted_4task.csv"
echo "[done] converted long csv: ${LONG_CSV}"
