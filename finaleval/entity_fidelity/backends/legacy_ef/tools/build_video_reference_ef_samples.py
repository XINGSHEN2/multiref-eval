#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image


FINAL_EF_ROOT = Path(__file__).resolve().parents[1]
EVAL_PY_ROOT = FINAL_EF_ROOT / "eval_py"
DEFAULT_WORKSPACE = Path(__file__).resolve().parents[3]
if str(EVAL_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_PY_ROOT))

from frame_sampler import sample_video_frames  # noqa: E402
from run_entity_fidelity import build_detector  # noqa: E402
from subject_detector import Box, crop_rgb  # noqa: E402


REFERENCE_IMAGE_COLUMNS = ("reference_images",)
REFERENCE_VIDEO_COLUMNS = ("reference_videos",)


def parse_json_list(value: str | None) -> list[Any]:
    text = (value or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def normalize_sample_id(value: str) -> str:
    text = str(value or "").strip()
    match = re.search(r"(board[0-9]+)[_-]?([0-9]{1,3})", text, flags=re.IGNORECASE)
    if not match:
        return text
    return f"{match.group(1).lower()}_{int(match.group(2)):03d}"


def board_from_sample(sample_id: str, row: dict[str, str]) -> str:
    board = str(row.get("board") or row.get("source_board") or "").strip().lower()
    if board in {"1", "2", "3"}:
        return f"board{board}"
    if board in {"board1", "board2", "board3"}:
        return board
    match = re.match(r"(board[0-9]+)", sample_id, flags=re.IGNORECASE)
    return match.group(1).lower() if match else ""


def seq_from_sample(sample_id: str, row: dict[str, str]) -> str:
    seq = str(row.get("seq") or "").strip()
    if seq:
        return str(int(seq)).zfill(3) if seq.isdigit() else seq.zfill(3)
    match = re.search(r"([0-9]{1,3})$", sample_id)
    if not match:
        raise ValueError(f"Cannot infer seq from sample_id={sample_id!r}")
    return f"{int(match.group(1)):03d}"


def abs_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def path_from_item(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("path") or item.get("video_path") or item.get("image_path") or item.get("file")
    return None


def video_hash(path: Path) -> str:
    try:
        stat = path.stat()
        key = f"{path.resolve()}::{stat.st_size}::{int(stat.st_mtime)}"
    except OSError:
        key = str(path.resolve())
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def infer_image_type(path: Path) -> str:
    text = str(path).lower()
    if "/scene/" in text or "scene_" in path.stem.lower():
        return "scene"
    if "/object/" in text or "object_" in path.stem.lower():
        return "object"
    if "/human/" in text or "human_" in path.stem.lower():
        return "human"
    return "unknown"


def placeholder(index: int) -> str:
    return f"@Image{index}"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def triangle_score(value: float, low: float, ideal: float, high: float) -> float:
    if value <= low or value >= high:
        return 0.0
    if value == ideal:
        return 1.0
    if value < ideal:
        return clamp01((value - low) / max(1e-6, ideal - low))
    return clamp01((high - value) / max(1e-6, high - ideal))


def sharpness_score(crop: np.ndarray) -> tuple[float, float]:
    gray = np.asarray(Image.fromarray(crop).convert("L"), dtype=np.float32)
    try:
        import cv2

        value = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        gy, gx = np.gradient(gray)
        value = float(np.var(gx) + np.var(gy))
    return value, clamp01((math.log1p(value) - math.log1p(20.0)) / (math.log1p(250.0) - math.log1p(20.0)))


def exposure_score(crop: np.ndarray) -> tuple[float, float]:
    gray = np.asarray(Image.fromarray(crop).convert("L"), dtype=np.float32)
    mean = float(gray.mean())
    if mean < 25.0 or mean > 235.0:
        return mean, 0.0
    return mean, clamp01(1.0 - abs(mean - 128.0) / 128.0)


def box_quality(box: Box, image_rgb: np.ndarray) -> dict[str, float | dict[str, float]]:
    height, width = image_rgb.shape[:2]
    frame_area = max(1.0, float(width * height))
    area_ratio = box.area / frame_area
    area = triangle_score(area_ratio, 0.03, 0.28, 0.85)

    cx = (box.x1 + box.x2) / 2.0
    cy = (box.y1 + box.y2) / 2.0
    norm_dist = math.sqrt(((cx / width) - 0.5) ** 2 + ((cy / height) - 0.5) ** 2) / math.sqrt(0.5)
    center = clamp01(1.0 - norm_dist)

    margin = min(box.x1 / width, box.y1 / height, (width - box.x2) / width, (height - box.y2) / height)
    completeness = clamp01(margin / 0.04)

    crop = np.asarray(crop_rgb(image_rgb, box))
    sharp_raw, sharp = sharpness_score(crop)
    exposure_raw, exposure = exposure_score(crop)

    components = {
        "area": area,
        "center": center,
        "sharpness": sharp,
        "exposure": exposure,
        "completeness": completeness,
        "face_bonus": 0.0,
    }
    quality = (
        0.25 * area
        + 0.20 * center
        + 0.20 * sharp
        + 0.15 * exposure
        + 0.10 * completeness
        + 0.10 * components["face_bonus"]
    )
    return {
        "quality": clamp01(quality),
        "area_ratio": area_ratio,
        "sharpness_raw": sharp_raw,
        "exposure_raw": exposure_raw,
        "components": components,
    }


def full_frame_box(image_rgb: np.ndarray) -> Box:
    height, width = image_rgb.shape[:2]
    return Box(0.0, 0.0, float(width), float(height), 1.0, "full_frame")


def choose_subject_box(boxes: list[Box], image_rgb: np.ndarray) -> Box | None:
    if not boxes:
        return None
    height, width = image_rgb.shape[:2]

    def rank(box: Box) -> float:
        center_x = (box.x1 + box.x2) / 2.0
        center_y = (box.y1 + box.y2) / 2.0
        center_dist = math.sqrt(((center_x / width) - 0.5) ** 2 + ((center_y / height) - 0.5) ** 2)
        return float(box.score) + 0.35 * min(1.0, box.area / max(1.0, width * height)) - 0.15 * center_dist

    return max(boxes, key=rank)


def save_crop(image_rgb: np.ndarray, box: Box, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    crop_rgb(image_rgb, box).save(path, quality=95)


def select_diverse_candidates(candidates: list[dict[str, Any]], count: int, min_gap: int) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda item: float(item["quality"]), reverse=True)
    selected: list[dict[str, Any]] = []
    for gap in (min_gap, max(1, min_gap // 2), 0):
        selected = []
        for item in ordered:
            sampled_index = int(item["sampled_index"])
            if any(abs(sampled_index - int(chosen["sampled_index"])) < gap for chosen in selected):
                continue
            selected.append(item)
            if len(selected) >= count:
                return selected
    return selected[:count]


def extract_reference_video(
    video_path: Path,
    cache_root: Path,
    detector,
    num_frames: int,
    select_count: int,
    min_gap: int,
) -> dict[str, Any]:
    key = video_hash(video_path)
    video_dir = cache_root / key
    metadata_path = video_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            selected_paths = [Path(path) for path in metadata.get("selected_reference_images") or []]
            if selected_paths and all(path.exists() for path in selected_paths):
                return metadata
        except Exception:
            pass

    frames = sample_video_frames(video_path, num_frames=num_frames)
    candidates: list[dict[str, Any]] = []
    for sampled_index, sampled in enumerate(frames):
        boxes = detector.detect(sampled.image_rgb, None) if detector else []
        box = choose_subject_box(boxes, sampled.image_rgb)
        fallback = False
        if box is None:
            box = full_frame_box(sampled.image_rgb)
            fallback = True
        quality_payload = box_quality(box, sampled.image_rgb)
        crop_path = video_dir / "candidates" / f"sample_{sampled_index:02d}_frame_{sampled.frame_index:06d}_crop.jpg"
        save_crop(sampled.image_rgb, box, crop_path)
        candidates.append(
            {
                "sampled_index": sampled_index,
                "frame_index": sampled.frame_index,
                "path": str(crop_path),
                "box": box.__dict__,
                "quality": quality_payload["quality"],
                "fallback_full_frame": fallback,
                "quality_details": quality_payload,
            }
        )

    selected = select_diverse_candidates(candidates, select_count, min_gap=min_gap)
    selected_paths = []
    for output_index, item in enumerate(selected, start=1):
        src = Path(item["path"])
        dst = video_dir / "selected" / f"ref_{output_index:02d}_frame_{int(item['frame_index']):06d}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.open(src).convert("RGB").save(dst, quality=95)
        selected_paths.append(str(dst))
        item["selected_path"] = str(dst)

    used_detector_crop = any(not bool(item.get("fallback_full_frame")) for item in candidates)
    metadata = {
        "video_path": str(video_path),
        "video_hash": key,
        "num_sampled_frames": len(frames),
        "selected_count": len(selected_paths),
        "selected_reference_images": selected_paths,
        "selected_candidates": selected,
        "candidate_count": len(candidates),
        "extraction_method": "yolo_world_subject_crop_quality_top3" if used_detector_crop else "full_frame_quality_top3",
    }
    video_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def image_items_from_paths(paths: list[Path], start_index: int = 1) -> list[dict[str, Any]]:
    items = []
    for offset, path in enumerate(paths, start=start_index):
        image_type = infer_image_type(path)
        if image_type == "unknown":
            continue
        items.append(
            {
                "placeholder": placeholder(offset),
                "type": image_type,
                "path": str(path),
                "relative_path": "",
                "description": path.stem,
            }
        )
    return items


def build_sample_json(
    sample_id: str,
    board: str,
    seq: str,
    row: dict[str, str],
    reference_images: list[Path],
    video_reference_images: list[Path],
    source_videos: list[Path],
    extraction_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    static_human_images = [path for path in reference_images if infer_image_type(path) == "human"]
    human_refs = [str(path) for path in video_reference_images + static_human_images]
    image_items = image_items_from_paths(reference_images)
    sample: dict[str, Any] = {
        "id": sample_id,
        "board": board,
        "seq": seq,
        "prompt": row.get("prompt", ""),
        "chinese_prompt": row.get("chinese_prompt", ""),
        "source_reference_videos": [str(path) for path in source_videos],
        "video_reference_extraction": extraction_metadata,
        "image_items": image_items,
        "reference_roles": {},
        "person_detection_prompts": ["person"],
        "human_entity_mode": "single_human" if human_refs else "none",
        "human_entity_count": 1 if human_refs else 0,
        "source_human_entity_mode": "single_human" if human_refs else "none",
        "source_human_entity_count": 1 if human_refs else 0,
    }
    for index, item in enumerate(image_items, start=1):
        sample[f"image{index}"] = item["path"]
    if human_refs:
        sample["human_entities"] = [
            {
                "entity_id": "person_1",
                "person_id": f"{sample_id}_reference_video_subject",
                "role": "main_subject",
                "reference_images": human_refs,
                "relative_reference_images": [],
                "detection_prompts": ["person"],
            }
        ]
    return sample


def build_samples_from_csv(
    csv_path: Path,
    output_root: Path,
    detector_args: argparse.Namespace | SimpleNamespace,
    board_filter: str = "",
    num_reference_frames: int = 16,
    selected_reference_count: int = 3,
    force: bool = False,
) -> dict[str, Any]:
    csv_path = csv_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    samples_root = output_root / "samples"
    cache_root = output_root / "video_reference_cache"
    detector = build_detector(detector_args)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_sample: dict[str, dict[str, Any]] = {}
    video_cache: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = normalize_sample_id(row.get("sample_id") or row.get("sample") or row.get("index") or "")
        if not sample_id:
            continue
        board = board_from_sample(sample_id, row)
        if board_filter and board != board_filter:
            continue
        seq = seq_from_sample(sample_id, row)
        entry = by_sample.setdefault(sample_id, {"row": row, "board": board, "seq": seq, "reference_images": [], "reference_videos": []})
        for column in REFERENCE_IMAGE_COLUMNS:
            for item in parse_json_list(row.get(column)):
                value = path_from_item(item)
                if value:
                    path = abs_path(value, csv_path.parent)
                    if path not in entry["reference_images"]:
                        entry["reference_images"].append(path)
        for column in REFERENCE_VIDEO_COLUMNS:
            for item in parse_json_list(row.get(column)):
                value = path_from_item(item)
                if value:
                    path = abs_path(value, csv_path.parent)
                    if path not in entry["reference_videos"]:
                        entry["reference_videos"].append(path)

    written = []
    for sample_id, entry in sorted(by_sample.items()):
        output = samples_root / f"{entry['board']}_{entry['seq']}.json"
        if output.exists() and not force:
            written.append(str(output))
            continue
        video_ref_images: list[Path] = []
        extraction_metadata = []
        for video_path in entry["reference_videos"]:
            cache_key = str(video_path)
            if cache_key not in video_cache:
                video_cache[cache_key] = extract_reference_video(
                    video_path,
                    cache_root,
                    detector,
                    num_frames=num_reference_frames,
                    select_count=selected_reference_count,
                    min_gap=max(1, num_reference_frames // max(1, selected_reference_count + 2)),
                )
            metadata = video_cache[cache_key]
            extraction_metadata.append(metadata)
            video_ref_images.extend(Path(path) for path in metadata.get("selected_reference_images") or [])
        sample = build_sample_json(
            sample_id=sample_id,
            board=entry["board"],
            seq=entry["seq"],
            row=entry["row"],
            reference_images=entry["reference_images"],
            video_reference_images=video_ref_images,
            source_videos=entry["reference_videos"],
            extraction_metadata=extraction_metadata,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(str(output))

    manifest = {
        "csv_path": str(csv_path),
        "samples_root": str(samples_root),
        "cache_root": str(cache_root),
        "sample_count": len(written),
        "unique_reference_videos": len(video_cache),
        "num_reference_frames": num_reference_frames,
        "selected_reference_count": selected_reference_count,
        "sample_jsons": written,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EF sample JSONs by extracting reference images from reference videos.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--board", choices=["", "board1", "board2", "board3"], default="")
    parser.add_argument("--num-reference-frames", type=int, default=16)
    parser.add_argument("--selected-reference-count", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--detector", choices=["yolo-world", "face-expanded"], default="yolo-world")
    parser.add_argument("--detector-device", default="cuda:0")
    parser.add_argument("--person-score-threshold", type=float, default=0.15)
    parser.add_argument("--yoloworld-config", default=str(FINAL_EF_ROOT / "eval_py/yolo_world_clip_fix.py"))
    parser.add_argument("--yoloworld-checkpoint", default="ef_eval/model_assets/OpenS2V-Weight/yolo_world_v2_l_image_prompt_adapter-719a7afb.pth")
    parser.add_argument("--yoloworld-repo", default=str(DEFAULT_WORKSPACE / "OpenS2V-Nexus/eval/utils/yoloworld"))
    parser.add_argument("--person-detection-texts", default="person")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.person_detection_texts = [item.strip() for item in args.person_detection_texts.split(",") if item.strip()] or ["person"]
    manifest = build_samples_from_csv(
        csv_path=args.csv,
        output_root=args.output_root,
        detector_args=args,
        board_filter=args.board,
        num_reference_frames=args.num_reference_frames,
        selected_reference_count=args.selected_reference_count,
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
