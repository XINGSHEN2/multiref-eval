from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConditionReference:
    key: str
    type: str
    path: Path
    description: str
    style: str | None
    raw_item: dict[str, Any]


def _clean_key(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    return value.strip().lstrip("@").replace("/", "_")


def load_condition_references(sample_json: str | Path, condition_type: str) -> list[ConditionReference]:
    sample_path = Path(sample_json)
    data = json.loads(sample_path.read_text(encoding="utf-8"))
    refs: list[ConditionReference] = []
    for index, item in enumerate(data.get("image_items") or [], start=1):
        if str(item.get("type") or "").lower() != condition_type:
            continue
        image_path = item.get("path")
        if not image_path:
            continue
        placeholder = _clean_key(item.get("placeholder"), f"{condition_type}_{index}")
        refs.append(
            ConditionReference(
                key=f"{condition_type}_{placeholder}",
                type=condition_type,
                path=Path(image_path),
                description=str(item.get("description") or ""),
                style=item.get("style"),
                raw_item=item,
            )
        )
    return refs


def object_prompt(ref: ConditionReference) -> str:
    text = ref.description.lower().strip()
    text = re.sub(r"\bobject\b", "", text)
    text = re.sub(r"^[a-z]+/\d+\s*", "", text)
    text = text.split(",", 1)[0]
    text = re.sub(r"[^a-z0-9 _-]+", " ", text)
    text = " ".join(text.replace("_", " ").split())
    if text:
        return text

    stem = ref.path.stem.lower()
    stem = re.sub(r"^object_[a-z]+_\d+$", "", stem)
    stem = re.sub(r"[^a-z0-9 _-]+", " ", stem)
    stem = " ".join(stem.replace("_", " ").split())
    return stem or "object"
