#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_METADATA_DIRS = [
    SCRIPT_DIR / "inputs" / "metadata",
]

VISUAL_EVENT_SOUND_ALIGNMENT = "Visual Event-Sound Alignment"
SOURCE_CORRECTNESS = "Source Correctness"
SPEECH_LIP_SYNC = "Speech-Lip Synchronization"
STABLE_SPEAKING_FRONTAL_FACE = "Stable Speaking Frontal Face"

AUDIO_KEYWORDS = (
    "sound",
    "audio",
    "music",
    "voice",
    "says",
    "speaks",
    "speech",
    "dialogue",
    "footstep",
    "collision",
    "impact",
    "ambient",
    "traffic",
    "water",
    "glass",
    "metal",
    "wood",
    "plastic",
    "rustling",
    "clatter",
    "noise",
)
VISUAL_ACTION_KEYWORDS = (
    "walk",
    "run",
    "step",
    "turn",
    "move",
    "place",
    "pick",
    "grab",
    "hold",
    "open",
    "close",
    "pour",
    "splash",
    "collide",
    "hit",
    "tap",
    "knock",
    "speak",
    "talk",
    "mouth",
    "smile",
)
REFERENCE_AUDIO_COLUMNS = (
    "reference_audio_path",
    "input_audio_path",
    "source_audio_path",
    "prompt_audio_path",
    "audio_path",
)
REFERENCE_AUDIO_LIST_COLUMNS = ("reference_audios",)


def parse_csv_list(value: str | None) -> set[str] | None:
    if not value or value.lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def normalize_sample_id(value: str) -> str:
    value = value.strip()
    match = re.search(r"(board[0-9]+)[_-]?([0-9]{1,3})", value, flags=re.IGNORECASE)
    if not match:
        return value
    return f"{match.group(1).lower()}_{int(match.group(2)):03d}"


def infer_sample_id(row: dict[str, str], video_path: Path) -> str:
    for key in ("sample_id", "source_sample_id", "sample", "id", "index"):
        if row.get(key):
            sample_id = normalize_sample_id(row[key])
            if sample_id:
                return sample_id
    return normalize_sample_id(video_path.stem)


def infer_board(sample_id: str, row: dict[str, str], video_path: Path) -> str:
    if row.get("source_board"):
        return row["source_board"].strip()
    match = re.match(r"(board[0-9]+)", sample_id, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    for part in video_path.parts:
        if re.fullmatch(r"board[0-9]+", part, flags=re.IGNORECASE):
            return part.lower()
    return ""


def infer_model_name(row: dict[str, str], video_path: Path) -> str:
    for key in ("model_name", "model", "method"):
        if row.get(key):
            return row[key].strip()
    parent = video_path.parent.name
    if re.fullmatch(r"board[0-9]+", parent, flags=re.IGNORECASE) and len(video_path.parents) > 1:
        return video_path.parents[1].name
    return parent or "model"


def sentence_chunks(prompt: str) -> list[str]:
    chunks = []
    for part in re.split(r"(?<=[.!?。！？])\s+", prompt):
        part = re.sub(r"\s+", " ", part).strip()
        if part:
            chunks.append(part)
    return chunks


def infer_language(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def placeholder_from_key(value: str) -> str:
    value = str(value or "").strip()
    if value.startswith("@"):
        return "@" + value[1:].capitalize()
    if value.lower().startswith("image"):
        return "@" + value.capitalize()
    return value


def extract_quoted_speech(prompt: str) -> list[dict[str, str]]:
    rows = []
    pattern = re.compile(r"(@Image\d+)\s+says,\s*[\"“](.*?)[\"”]", flags=re.IGNORECASE)
    for match in pattern.finditer(prompt):
        text = match.group(2).strip()
        rows.append({"speaker": placeholder_from_key(match.group(1)), "text": text, "language": infer_language(text)})
    if rows:
        return rows
    for text in re.findall(r"[\"“](.*?)[\"”]", prompt):
        text = text.strip()
        if text:
            rows.append({"speaker": "", "text": text, "language": infer_language(text)})
    return rows


def infer_actor(text: str) -> str:
    match = re.search(r"@Image\d+", text, flags=re.IGNORECASE)
    return placeholder_from_key(match.group(0)) if match else ""


def infer_audio_source(text: str) -> str:
    actor = infer_actor(text)
    if actor:
        return actor
    lower = text.lower()
    if "traffic" in lower or "ambient" in lower or "environment" in lower:
        return "environment"
    if "music" in lower:
        return "music_track"
    return "visible event source if identifiable"


def infer_timing_hint(text: str) -> str:
    lower = text.lower()
    for token in ("initially", "then", "finally", "as the sequence continues", "concludes", "during", "while"):
        if token in lower:
            return token
    return ""


def audio_event_rows(prompt: str) -> list[dict[str, str]]:
    rows = []
    for idx, sent in enumerate(sentence_chunks(prompt), start=1):
        lower = sent.lower()
        if any(token in lower for token in AUDIO_KEYWORDS):
            rows.append(
                {
                    "event_id": f"audio_event_{idx}",
                    "source": infer_audio_source(sent),
                    "sound": sent,
                    "expected_visual_anchor": "visible event or scene described in the same prompt sentence when present",
                    "timing_hint": infer_timing_hint(sent),
                }
            )
    return rows


def visual_event_rows(prompt: str) -> list[dict[str, str | None]]:
    rows = []
    for idx, sent in enumerate(sentence_chunks(prompt), start=1):
        lower = sent.lower()
        if any(token in lower for token in VISUAL_ACTION_KEYWORDS):
            rows.append(
                {
                    "event_id": f"visual_event_{idx}",
                    "actor": infer_actor(sent),
                    "action": sent,
                    "expected_sound": sent if any(token in lower for token in AUDIO_KEYWORDS) else None,
                    "timing_hint": infer_timing_hint(sent),
                }
            )
    return rows


def source_rows(speech_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for idx, speech in enumerate(speech_rows, start=1):
        rows.append(
            {
                "source_id": f"speech_{idx}",
                "source_type": "human",
                "expected_speaker": speech.get("speaker") or "specified speaker",
                "expected_speech": speech.get("text", ""),
                "language": speech.get("language", ""),
                "check": "whether this prompted spoken line is spoken by the prompted speaker in the generated video",
            }
        )
    return rows


def image_items(sample: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in sample.get("image_items", []) or []:
        rows.append(
            {
                "placeholder": placeholder_from_key(item.get("placeholder", "")),
                "type": item.get("type", "unknown"),
                "path": item.get("path"),
                "relative_path": item.get("relative_path"),
                "description": item.get("description", ""),
            }
        )
    return rows


def load_metadata(metadata_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    metadata = {}
    for directory in metadata_dirs:
        for path in sorted(directory.expanduser().glob("*.json")):
            if path.name == "manifest.json":
                continue
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            sample_id = str(row.get("id") or path.stem)
            row["_source_json"] = str(path.resolve())
            metadata[sample_id] = row
    return metadata


def video_column(row: dict[str, str]) -> str:
    for key in ("video_path", "combined_video", "path", "file", "filepath"):
        if row.get(key):
            return row[key]
    raise ValueError("CSV row is missing a video column. Expected one of: video_path, combined_video, path, file, filepath")


def reference_audio_item(row: dict[str, str], csv_path: Path) -> dict[str, str] | None:
    for key in REFERENCE_AUDIO_LIST_COLUMNS:
        value = (row.get(key) or "").strip()
        if not value:
            continue
        try:
            items = json.loads(value)
        except json.JSONDecodeError:
            items = []
        if not isinstance(items, list) or not items:
            continue
        first = items[0]
        if isinstance(first, dict):
            first = first.get("path") or first.get("audio_path") or first.get("file")
        if not first:
            continue
        audio_path = Path(str(first)).expanduser()
        if not audio_path.is_absolute():
            audio_path = (csv_path.parent / audio_path).resolve()
        else:
            audio_path = audio_path.resolve()
        return {
            "path": str(audio_path),
            "column": key,
            "exists": audio_path.exists(),
            "description": "Input/reference speech audio for voice timbre similarity.",
        }
    for key in REFERENCE_AUDIO_COLUMNS:
        value = (row.get(key) or "").strip()
        if not value:
            continue
        audio_path = Path(value).expanduser()
        if not audio_path.is_absolute():
            audio_path = (csv_path.parent / audio_path).resolve()
        else:
            audio_path = audio_path.resolve()
        return {
            "path": str(audio_path),
            "column": key,
            "exists": audio_path.exists(),
            "description": "Input/reference speech audio for voice timbre similarity.",
        }
    return None


def build_spec(
    sample_id: str,
    source_board: str,
    row: dict[str, str],
    metadata_row: dict[str, Any] | None,
    video_path: Path,
    csv_path: Path,
) -> dict[str, Any]:
    prompt = row.get("prompt") or (metadata_row or {}).get("prompt") or ""
    speech_rows = extract_quoted_speech(prompt)
    audio_events = audio_event_rows(prompt)
    visual_events = visual_event_rows(prompt)
    has_speech_value = row.get("has_speech")
    has_speech = bool(speech_rows) if has_speech_value is None or has_speech_value == "" else has_speech_value.lower() in {"1", "true", "yes", "y"}
    reference_audio = reference_audio_item(row, csv_path)

    spec = {
        "sample_id": sample_id,
        "source_sample_id": row.get("source_sample_id") or (metadata_row or {}).get("source_sample_id") or sample_id,
        "source_board": source_board,
        "prompt": prompt,
        "chinese_prompt": row.get("chinese_prompt") or (metadata_row or {}).get("chinese_prompt", ""),
        "references": image_items(metadata_row or {}),
        "metadata": {
            "human_entity_mode": (metadata_row or {}).get("human_entity_mode"),
            "human_entity_count": (metadata_row or {}).get("human_entity_count"),
            "source_human_entity_count": (metadata_row or {}).get("source_human_entity_count"),
            "has_sound": row.get("has_sound") or (metadata_row or {}).get("has_sound"),
            "has_music": row.get("has_music") or (metadata_row or {}).get("has_music"),
            "has_speech": has_speech,
            "is_multi_shot": row.get("is_multi_shot") or (metadata_row or {}).get("is_multi_shot"),
            "duration_sec": row.get("duration_sec") or (metadata_row or {}).get("duration_sec"),
        },
        "speech_prompt": speech_rows,
        "expected_audio_events": audio_events,
        "expected_visual_events": visual_events,
        "expected_sources": source_rows(speech_rows),
        "applicable_dimensions": {
            VISUAL_EVENT_SOUND_ALIGNMENT: True,
            SOURCE_CORRECTNESS: bool(speech_rows),
            SPEECH_LIP_SYNC: has_speech,
            STABLE_SPEAKING_FRONTAL_FACE: has_speech,
            "Voice Timbre Similarity": bool(reference_audio and reference_audio.get("exists")),
            "Audio-Visual Synchronization": True,
            "Event-Sound Matching": True,
        },
        "source_json": (metadata_row or {}).get("_source_json"),
        "csv_video_path": str(video_path),
    }
    if reference_audio:
        spec["reference_audio"] = reference_audio
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a video CSV into Audio Consistency task JSON.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--metadata-dirs", default=",".join(str(path) for path in DEFAULT_METADATA_DIRS))
    parser.add_argument("--models", default="all")
    parser.add_argument("--samples", default="all")
    parser.add_argument("--skip-missing-videos", action="store_true")
    args = parser.parse_args()

    metadata_dirs = [Path(item.strip()) for item in args.metadata_dirs.split(",") if item.strip()]
    metadata = load_metadata(metadata_dirs)
    selected_models = parse_csv_list(args.models)
    selected_samples = parse_csv_list(args.samples)
    tasks = []
    missing = []

    input_csv = args.input_csv.expanduser().resolve()
    with input_csv.open(newline="", encoding="utf-8") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=1):
            video_path = Path(video_column(row)).expanduser().resolve()
            sample_id = infer_sample_id(row, video_path)
            model_name = infer_model_name(row, video_path)
            if selected_models and model_name not in selected_models:
                continue
            if selected_samples and sample_id not in selected_samples:
                continue
            if args.skip_missing_videos and not video_path.exists():
                missing.append(str(video_path))
                continue
            source_board = infer_board(sample_id, row, video_path)
            metadata_row = metadata.get(sample_id)
            spec = build_spec(sample_id, source_board, row, metadata_row, video_path, input_csv)
            item_id = row.get("index") or row.get("id") or f"{sample_id}::{model_name}"
            tasks.append(
                {
                    "task_id": str(item_id),
                    "sample_id": sample_id,
                    "model_name": model_name,
                    "video_path": str(video_path),
                    "video_exists": video_path.exists(),
                    "csv_row_number": row_number,
                    "ac_eval_spec": spec,
                }
            )

    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(tasks)} tasks to {output}")
    print(f"Samples: {len({task['sample_id'] for task in tasks})}; models: {len({task['model_name'] for task in tasks})}")
    if missing:
        print(f"Skipped missing videos: {len(missing)}")


if __name__ == "__main__":
    main()
