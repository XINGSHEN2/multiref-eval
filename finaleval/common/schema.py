from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


BASE_REQUIRED_COLUMNS = ["sample_id", "model_name", "prompt"]


@dataclass
class InputSample:
    sample_id: str
    model_name: str
    prompt: str
    video_path: str = ""
    board: str = ""
    chinese_prompt: str = ""
    has_speech: str = ""
    has_sound: str = ""
    has_music: str = ""
    reference_audio_path: str = ""
    reference_image_paths: str = ""
    reference_video_path: str = ""
    reference_json_path: str = ""


def resolve_path(value: str, base_dir: Path) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def resolve_path_list(value: str, base_dir: Path) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            paths = [resolve_path(str(item), base_dir) for item in parsed if str(item).strip()]
            return json.dumps(paths, ensure_ascii=False)
    except json.JSONDecodeError:
        pass

    separator = ";" if ";" in value else ","
    paths = [resolve_path(item, base_dir) for item in value.split(separator) if item.strip()]
    return json.dumps(paths, ensure_ascii=False)


def read_input_csv(path: Path) -> list[InputSample]:
    base_dir = path.expanduser().resolve().parent
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        missing = [name for name in BASE_REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV is missing required column(s): {', '.join(missing)}. "
                "MultiRef-Compass open-source CSV uses fixed column names and does not infer aliases."
            )
        rows = []
        for row in reader:
            rows.append(
                InputSample(
                    sample_id=(row.get("sample_id") or "").strip(),
                    model_name=(row.get("model_name") or "").strip(),
                    prompt=row.get("prompt") or "",
                    video_path=resolve_path(row.get("video_path") or "", base_dir),
                    board=(row.get("board") or "").strip(),
                    chinese_prompt=row.get("chinese_prompt") or "",
                    has_speech=(row.get("has_speech") or "").strip(),
                    has_sound=(row.get("has_sound") or "").strip(),
                    has_music=(row.get("has_music") or "").strip(),
                    reference_audio_path=resolve_path(row.get("reference_audio_path") or "", base_dir),
                    reference_image_paths=resolve_path_list(row.get("reference_image_paths") or "", base_dir),
                    reference_video_path=resolve_path(row.get("reference_video_path") or "", base_dir),
                    reference_json_path=resolve_path(row.get("reference_json_path") or "", base_dir),
                )
            )
    return rows
