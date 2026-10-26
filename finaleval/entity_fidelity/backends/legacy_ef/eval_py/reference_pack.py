from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class ReferenceImage:
    key: str
    path: Path
    roles: dict[str, Any]

    def pil(self) -> Image.Image:
        return Image.open(self.path).convert("RGB")


@dataclass(frozen=True)
class ReferencePack:
    sample_id: str
    references: list[ReferenceImage]
    raw_sample: dict[str, Any]


@dataclass(frozen=True)
class HumanEntityPack:
    entity_id: str
    person_id: str | None
    references: list[ReferenceImage]
    raw_entity: dict[str, Any]


@dataclass(frozen=True)
class HumanReferencePack:
    sample_id: str
    entities: list[HumanEntityPack]
    mode: str
    raw_sample: dict[str, Any]


def infer_roles_from_path(path: str) -> dict[str, Any]:
    name = Path(path).stem.lower()
    if any(token in name for token in ("side", "profile")):
        view = "profile"
    elif any(token in name for token in ("quarter", "three_quarter")):
        view = "three_quarter"
    else:
        view = "frontal"

    expression = "neutral"
    for token in ("smile", "speaking", "angry", "surprised"):
        if token in name:
            expression = token
            break

    framing = "full_body" if any(token in name for token in ("full", "body")) else "upper_body"
    return {"view": view, "expression": expression, "framing": framing}


def entity_reference_images(sample: dict[str, Any], entity: dict[str, Any]) -> list[str]:
    source_keys = entity.get("source_image_keys") or []
    references: list[str] = []
    for key in source_keys:
        image_path = sample.get(str(key))
        if image_path and str(image_path) not in references:
            references.append(str(image_path))
    if references:
        return references
    return [str(path) for path in (entity.get("reference_images") or [])]


def load_reference_pack(sample_json: str | Path) -> ReferencePack:
    sample_path = Path(sample_json)
    data = json.loads(sample_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected one sample object in {sample_path}")

    human_entities = data.get("human_entities") or []
    if len(human_entities) == 1:
        entity = human_entities[0]
        references = []
        for index, image_path in enumerate(entity_reference_images(data, entity), start=1):
            key = f"{entity.get('entity_id', 'person_1')}_ref{index}"
            references.append(ReferenceImage(key=key, path=Path(image_path), roles=infer_roles_from_path(image_path)))
        if references:
            return ReferencePack(
                sample_id=str(data.get("id") or sample_path.stem),
                references=references,
                raw_sample=data,
            )

    roles = data.get("reference_roles") or {}
    references: list[ReferenceImage] = []
    for key in ("image1", "image2", "image3"):
        image_path = data.get(key)
        if not image_path:
            continue
        item_roles = roles.get(key) or infer_roles_from_path(image_path)
        references.append(ReferenceImage(key=key, path=Path(image_path), roles=item_roles))

    if not references:
        raise ValueError(f"No image1/image2/image3 paths found in {sample_path}")

    return ReferencePack(
        sample_id=str(data.get("id") or sample_path.stem),
        references=references,
        raw_sample=data,
    )


def load_human_reference_pack(sample_json: str | Path) -> HumanReferencePack:
    sample_path = Path(sample_json)
    data = json.loads(sample_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected one sample object in {sample_path}")

    human_entities = data.get("human_entities") or []
    entities: list[HumanEntityPack] = []
    for entity_index, entity in enumerate(human_entities, start=1):
        entity_id = str(entity.get("entity_id") or f"person_{entity_index}")
        references: list[ReferenceImage] = []
        for ref_index, image_path in enumerate(entity_reference_images(data, entity), start=1):
            key = f"{entity_id}_ref{ref_index}"
            references.append(ReferenceImage(key=key, path=Path(image_path), roles=infer_roles_from_path(image_path)))
        if references:
            entities.append(
                HumanEntityPack(
                    entity_id=entity_id,
                    person_id=entity.get("person_id"),
                    references=references,
                    raw_entity=entity,
                )
            )

    if not entities:
        single_pack = load_reference_pack(sample_path)
        entities = [
            HumanEntityPack(
                entity_id="person_1",
                person_id=data.get("person_id"),
                references=single_pack.references,
                raw_entity={},
            )
        ]

    return HumanReferencePack(
        sample_id=str(data.get("id") or sample_path.stem),
        entities=entities,
        mode="multi_human" if len(entities) > 1 else "single_human",
        raw_sample=data,
    )
