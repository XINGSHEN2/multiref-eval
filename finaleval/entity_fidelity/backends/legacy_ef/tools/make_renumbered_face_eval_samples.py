#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATASETS_ROOT = ROOT / "datasets"
REN_ROOT = DATASETS_ROOT / "cdatasets/final_success_tables_20260521/renumbered_20260522"
OUT_ROOT = DATASETS_ROOT / "evaltest/renumbered_face_ef_samples"
MODELS = ("happyhourse", "seedance", "veo", "vidu", "wan")
SAMPLES = {
    "board1": ("027", "034", "045", "104"),
    "board2": ("004", "024", "034", "100"),
    "board3": ("001", "002", "011", "027"),
}


def parse_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def rel_to_datasets(path: str) -> str:
    value = Path(path)
    try:
        return str(value.relative_to(DATASETS_ROOT))
    except ValueError:
        return str(value)


def image_type(path: str) -> str:
    lowered = path.lower()
    if "/human/" in lowered:
        return "human"
    if "/object/" in lowered:
        return "object"
    if "/scene/" in lowered:
        return "scene"
    return "unknown"


def person_id_from_path(path: str) -> str | None:
    value = Path(path)
    parts = value.parts
    if "human" not in parts:
        return None
    index = parts.index("human")
    if len(parts) > index + 2:
        return "/".join(parts[index : index + 3])
    return None


def human_reference_images(path: str) -> list[str]:
    """Return only the human reference image that was actually provided to the sample."""
    return [path]


def person_detection_prompts(person_id: str, paths: list[str]) -> list[str]:
    text = " ".join([person_id, *paths]).lower()
    prompts = ["person"]
    if "cartoon" in text:
        prompts = ["cartoon character", "animated character", "toy character", "person"]
    elif "woman" in text:
        prompts = ["woman", "person"]
    elif "man" in text:
        prompts = ["man", "person"]
    return prompts


def load_summary(board: str) -> dict[str, dict[str, str]]:
    path = REN_ROOT / f"{board}_summary_with_appearance_consistency.md"
    rows: dict[str, dict[str, str]] = {}
    header: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells:
            continue
        if cells[0] == "index":
            header = cells
            continue
        if cells[0].startswith("---") or header is None or len(cells) != len(header):
            continue
        rows[cells[0]] = dict(zip(header, cells))
    return rows


def has_audio_flag(prompt: str, keywords: tuple[str, ...]) -> bool:
    lowered = prompt.lower()
    return any(keyword in lowered for keyword in keywords)


def build_image_items(image_paths: list[str]) -> list[dict[str, Any]]:
    items = []
    for index, path in enumerate(image_paths, start=1):
        kind = image_type(path)
        items.append(
            {
                "placeholder": f"@image{index}",
                "path": path,
                "relative_path": rel_to_datasets(path),
                "type": kind,
                "description": Path(path).stem,
            }
        )
    return items


def build_human_entities(image_paths: list[str], keep_primary_only: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[str]] = {}
    source_keys: dict[str, list[str]] = {}
    for index, path in enumerate(image_paths, start=1):
        if image_type(path) != "human":
            continue
        person_id = person_id_from_path(path) or f"human_{index}"
        grouped.setdefault(person_id, []).append(path)
        source_keys.setdefault(person_id, []).append(f"image{index}")

    entities = []
    for entity_index, (person_id, paths) in enumerate(grouped.items(), start=1):
        ref_images: list[str] = []
        for path in paths:
            for ref in human_reference_images(path):
                if ref not in ref_images:
                    ref_images.append(ref)
        entities.append(
            {
                "entity_id": f"person_{entity_index}",
                "type": "human",
                "person_id": person_id,
                "role": "main_subject" if entity_index == 1 else "secondary_subject",
                "detection_prompts": person_detection_prompts(person_id, paths),
                "source_image_keys": source_keys[person_id],
                "reference_images": ref_images,
                "relative_reference_images": [rel_to_datasets(path) for path in ref_images],
            }
        )
    if keep_primary_only and entities:
        return [entities[0]], entities
    return entities, entities


def build_sample(board: str, seq: str, row: pd.Series, summary: dict[str, str]) -> dict[str, Any]:
    sample_id = f"MultiOmni-{board}-{seq}"
    image_paths = parse_list(row.get("image_path")) or parse_list(row.get("reference_image_paths"))
    image_items = build_image_items(image_paths)
    human_entities, all_human_entities = build_human_entities(image_paths, keep_primary_only=False)
    detection_prompts = []
    for entity in all_human_entities:
        for prompt in entity.get("detection_prompts", []):
            if prompt not in detection_prompts:
                detection_prompts.append(prompt)
    prompt = "" if pd.isna(row.get("prompt")) else str(row.get("prompt"))
    zh_prompt = "" if pd.isna(row.get("中文prompt")) else str(row.get("中文prompt"))
    appearance_requirement = summary.get("appearance_requirement") == "Y"
    multi_shot = summary.get("multi_shot") == "Y"

    data: dict[str, Any] = {
        "index": int(seq),
        "id": f"{board}_{seq}",
        "sample": sample_id,
        "source_sample_id": str(row.get("original_index") or sample_id),
        "task": "multi_image_reference_video_generation",
        "duration_sec": 10,
        "source_board": board,
        "source_xlsx": str(REN_ROOT / f"MultiOmni_{board}_final_success_renumbered.xlsx"),
        "prompt": prompt,
        "prompt_char_count": len(prompt),
        "prompt_word_count": len(prompt.split()),
        "video_prompt": prompt,
        "chinese_prompt": zh_prompt,
        "has_sound": has_audio_flag(prompt, ("sound", "audio", "ambient", "footsteps", "rustling")),
        "has_music": has_audio_flag(prompt, ("music", "track", "melody", "guitar", "piano", "strings")),
        "has_speech": bool(re.search(r'\b(says|speaks|speaking|line|dialogue)\b|说道|说话|对白|台词', prompt, re.I)),
        "duration": "10s",
        "is_multi_shot": multi_shot,
        "has_clothing_or_makeup_change": appearance_requirement,
        "appearance_requirement": appearance_requirement,
        "appearance_consistency": summary.get("appearance_consistency", ""),
        "appearance_consistency_note": summary.get("appearance_consistency_note", ""),
        "multi_shot_evidence": summary.get("multi-shot evidence", ""),
        "appearance_evidence": summary.get("appearance evidence", ""),
        "reference_placeholders": {f"image{i}": f"@image{i}" for i in range(1, len(image_paths) + 1)},
        "image_items": image_items,
        "human_entities": human_entities,
        "all_human_entities": all_human_entities,
        "person_detection_prompts": detection_prompts or ["person"],
        "human_entity_mode": "multi_human" if len(human_entities) > 1 else "single_human",
        "human_entity_count": len(human_entities),
        "source_human_entity_mode": "multi_human" if len(all_human_entities) > 1 else "single_human",
        "source_human_entity_count": len(all_human_entities),
        "face_eval_primary_human_only": len(all_human_entities) > len(human_entities),
        "model_videos": {
            model: str(REN_ROOT / "videos" / model / f"{sample_id}.mp4")
            for model in MODELS
        },
    }
    for index, path in enumerate(image_paths, start=1):
        data[f"image{index}"] = path
        data[f"relative_image{index}"] = rel_to_datasets(path)
    return data


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for board, seqs in SAMPLES.items():
        workbook = REN_ROOT / f"MultiOmni_{board}_final_success_renumbered.xlsx"
        frame = pd.read_excel(workbook)
        rows = {str(item["index"]): item for _, item in frame.iterrows()}
        summary_rows = load_summary(board)
        for seq in seqs:
            sample_id = f"MultiOmni-{board}-{seq}"
            if sample_id not in rows:
                raise KeyError(f"Missing {sample_id} in {workbook}")
            data = build_sample(board, seq, rows[sample_id], summary_rows.get(sample_id, {}))
            output_path = OUT_ROOT / f"{board}_{seq}.json"
            output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest.append({"id": data["id"], "json": str(output_path), "sample": sample_id})
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} sample JSON files to {OUT_ROOT}")


if __name__ == "__main__":
    main()
