from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from subject_detector import Box


COLORS = (
    (255, 80, 80),
    (80, 180, 255),
    (100, 220, 120),
    (255, 190, 70),
    (210, 120, 255),
    (255, 120, 190),
)
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def clean_label(text: str | None, fallback: str = "subject") -> str:
    if not text:
        return fallback
    label = Path(str(text)).name
    if label.startswith("human_"):
        label = label[len("human_") :]
    return label or fallback


def box_to_dict(box: Box) -> dict[str, Any]:
    return {
        "x1": float(box.x1),
        "y1": float(box.y1),
        "x2": float(box.x2),
        "y2": float(box.y2),
        "score": float(box.score),
        "label": box.label,
    }


def _font_for_image(width: int, height: int) -> ImageFont.ImageFont:
    size = max(18, min(44, int(min(width, height) * 0.032)))
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size=size)
    return ImageFont.load_default()


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _clamp_label_rect(
    x: int,
    y: int,
    label_w: int,
    label_h: int,
    image_w: int,
    image_h: int,
) -> tuple[int, int, int, int]:
    x = max(0, min(x, max(0, image_w - label_w)))
    y = max(0, min(y, max(0, image_h - label_h)))
    return (x, y, x + label_w, y + label_h)


def _choose_label_rect(
    box_rect: tuple[int, int, int, int],
    label_w: int,
    label_h: int,
    image_size: tuple[int, int],
    line_width: int,
    label_slot: int,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    image_w, image_h = image_size
    x1, y1, x2, y2 = box_rect
    offset = line_width + (label_slot // 2) * max(4, label_h // 5)
    candidates = [
        (x1, y1 - label_h - offset),
        (x1, y2 + offset),
        (x1, y1 + offset),
        (x1, y2 - label_h - offset),
        (x2 + offset, y1),
        (x1 - label_w - offset, y1),
    ]
    rects = [_clamp_label_rect(x, y, label_w, label_h, image_w, image_h) for x, y in candidates]
    for rect in rects:
        if not any(_intersects(rect, used) for used in occupied):
            return rect
    return min(rects, key=lambda rect: sum(_intersection_area(rect, used) for used in occupied))


def _draw_box(
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    box: Box,
    label: str,
    color: tuple[int, int, int],
    display_text: str | None = None,
    label_slot: int = 0,
    occupied_labels: list[tuple[int, int, int, int]] | None = None,
) -> None:
    if occupied_labels is None:
        occupied_labels = []
    image_w, image_h = image_size
    x1, y1, x2, y2 = [int(round(v)) for v in (box.x1, box.y1, box.x2, box.y2)]
    line_width = max(4, int(min(image_size) * 0.006))
    draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

    font = _font_for_image(*image_size)
    text = display_text or f"{label} {box.score:.2f}"
    bbox = draw.textbbox((x1, y1), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x = max(10, text_h // 3)
    pad_y = max(6, text_h // 5)
    label_w = text_w + 2 * pad_x
    label_h = text_h + 2 * pad_y
    label_rect = _choose_label_rect(
        (x1, y1, x2, y2),
        label_w,
        label_h,
        (image_w, image_h),
        line_width,
        label_slot,
        occupied_labels,
    )
    x_text, y_text, x_text2, y_text2 = label_rect
    draw.rectangle((x_text, y_text, x_text + label_w, y_text + label_h), fill=color)
    draw.text((x_text + pad_x, y_text + pad_y), text, fill=(0, 0, 0), font=font)
    occupied_labels.append((x_text, y_text, x_text2, y_text2))


def save_annotated_frame(
    image_rgb: np.ndarray,
    boxes: list[tuple],
    output_dir: Path,
    frame_index: int,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    box_items = []
    occupied_labels: list[tuple[int, int, int, int]] = []
    for idx, item in enumerate(boxes):
        if len(item) == 2:
            box, label = item
            display_text = None
        else:
            box, label, display_text = item
        color = COLORS[idx % len(COLORS)]
        _draw_box(draw, image.size, box, label, color, display_text, label_slot=idx, occupied_labels=occupied_labels)
        box_items.append({"label": label, "display_text": display_text, "box": box_to_dict(box)})

    image_path = output_dir / f"frame_{frame_index:06d}_boxes.jpg"
    json_path = output_dir / f"frame_{frame_index:06d}_boxes.json"
    image.save(image_path, quality=95)

    payload = {
        "frame_index": int(frame_index),
        "annotated_image": str(image_path),
        "boxes": box_items,
    }
    if metadata:
        payload.update(metadata)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"image": str(image_path), "json": str(json_path), "boxes": box_items}
