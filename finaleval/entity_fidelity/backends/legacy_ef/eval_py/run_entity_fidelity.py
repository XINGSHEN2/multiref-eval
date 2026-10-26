#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from body_branch import BodyBranch, body_pack_top2
from cartoon_face_detector import best_face_for_box, build_cartoon_face_detector
from condition_pack import load_condition_references
from face_branch import FaceBranch, face_pack_component_top2, face_pack_top2
from frame_box_visualization import clean_label, save_annotated_frame
from frame_sampler import sample_video_frames
from multi_entity_fidelity import MultiEntityFidelityEvaluator
from object_branch import ObjectBranch
from reference_pack import load_human_reference_pack, load_reference_pack
from scene_branch import SceneBranch
from subject_detector import Box, FaceExpandedBodyDetector, YOLOWorldPersonDetector, crop_rgb

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HUMAN_ROOT = DEFAULT_ROOT / "datasets" / "human"
DEFAULT_REFERENCE_BODY_MASK_ROOT = DEFAULT_ROOT / "datasets" / "human_masks_sam3" / "masks"
DEFAULT_MIN_FRAME_EF = 0.1
DEFAULT_CARTOON_MIN_BODY_EF = 0.6


def build_detector(args):
    if args.detector == "face-expanded":
        return FaceExpandedBodyDetector()
    texts = getattr(args, "person_detection_texts", None)
    return YOLOWorldPersonDetector(
        config=args.yoloworld_config,
        checkpoint=args.yoloworld_checkpoint,
        device=args.detector_device,
        repo_root=args.yoloworld_repo,
        score_threshold=args.person_score_threshold,
        texts=texts,
    )


def build_text_detector(args, prompt: str):
    return YOLOWorldPersonDetector(
        config=args.yoloworld_config,
        checkpoint=args.yoloworld_checkpoint,
        device=args.detector_device,
        repo_root=args.yoloworld_repo,
        score_threshold=args.object_score_threshold,
        texts=[prompt],
    )


def has_cartoon_human(pack) -> bool:
    entities = getattr(pack, "entities", None)
    if entities is not None:
        return any("cartoon" in str(entity.person_id or entity.raw_entity).lower() for entity in entities)
    return is_cartoon_reference_pack(pack)


def reference_top_k(args) -> int:
    return max(1, int(getattr(args, "reference_top_k", 1)))


def min_frame_ef(args) -> float:
    return float(getattr(args, "min_frame_ef", DEFAULT_MIN_FRAME_EF))


def cartoon_min_body_ef(args) -> float:
    return float(getattr(args, "cartoon_min_body_ef", DEFAULT_CARTOON_MIN_BODY_EF))


def passes_min_frame_ef(face_like_ef: float | None, body_ef: float | None, threshold: float) -> bool:
    if face_like_ef is not None and face_like_ef < threshold:
        return False
    if body_ef is not None and body_ef < threshold:
        return False
    return True


def _contains_cartoon(value: Any) -> bool:
    if isinstance(value, str):
        return "cartoon" in value.lower()
    if isinstance(value, dict):
        return any(_contains_cartoon(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_cartoon(item) for item in value)
    return False


def is_cartoon_reference_pack(pack) -> bool:
    if any("cartoon" in str(ref.path).lower() for ref in pack.references):
        return True
    raw = getattr(pack, "raw_sample", {}) or {}
    image_values = [raw.get(key) for key in ("image1", "image2", "image3", "person_id")]
    image_values.extend(raw.get("image_items") or [])
    image_values.extend(raw.get("human_entities") or [])
    return _contains_cartoon(image_values)


def quality_weight(face_valid: bool, body_valid: bool, face_score: float, body_score: float) -> float:
    if not face_valid and not body_valid:
        return 0.0
    base = 0.5
    if face_valid:
        base += 0.25 * max(0.0, min(1.0, face_score))
    if body_valid:
        base += 0.25 * max(0.0, min(1.0, body_score))
    return base


def head_box_from_body_box(body_box: Box, head_ratio: float = 0.45) -> Box:
    height = max(1.0, body_box.y2 - body_box.y1)
    return Box(
        x1=body_box.x1,
        y1=body_box.y1,
        x2=body_box.x2,
        y2=body_box.y1 + height * head_ratio,
        score=body_box.score,
        label="cartoon_head",
    )


def reference_head_crop(image: Image.Image, head_ratio: float = 0.55) -> Image.Image:
    width, height = image.size
    return image.convert("RGB").crop((0, 0, width, max(1, int(round(height * head_ratio)))))


def reference_mask_path(image_path: Path, human_root: Path, mask_root: Path) -> Path | None:
    try:
        rel = image_path.resolve().relative_to(human_root.resolve())
    except ValueError:
        return None
    base = mask_root / rel.with_suffix("")
    return base.with_name(f"{base.name}.mask.png")


def apply_reference_body_mask(image: Image.Image, mask_path: Path) -> Image.Image:
    image_rgb = image.convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != image_rgb.size:
        mask = mask.resize(image_rgb.size, Image.Resampling.NEAREST)
    image_np = np.asarray(image_rgb, dtype=np.uint8)
    mask_np = (np.asarray(mask, dtype=np.float32) / 255.0)[..., None]
    masked = (image_np.astype(np.float32) * mask_np).clip(0, 255).astype(np.uint8)
    return Image.fromarray(masked, mode="RGB")


def reference_body_image(ref, args) -> tuple[Image.Image, str | None]:
    image = ref.pil()
    if args.disable_reference_body_mask:
        return image, None
    mask_path = reference_mask_path(
        ref.path,
        human_root=Path(args.reference_human_root),
        mask_root=Path(args.reference_body_mask_root),
    )
    if mask_path is None or not mask_path.exists():
        return image, None
    return apply_reference_body_mask(image, mask_path), str(mask_path)


def reference_cartoon_face_image(ref, cartoon_face_detector) -> Image.Image | None:
    if cartoon_face_detector is None:
        return None
    image = ref.pil()
    image_np = np.asarray(image.convert("RGB"))
    detection = best_face_for_box(cartoon_face_detector.detect(image_np))
    if detection is None:
        return None
    return crop_rgb(image_np, detection.box)


def frame_box_root(args) -> Path | None:
    if args.disable_frame_box_export:
        return None
    if not args.save_frame_boxes and not args.frame_box_dir:
        return None
    if args.frame_box_dir:
        return Path(args.frame_box_dir)
    output_path = Path(args.output_json)
    return output_path.with_name(f"{output_path.stem}_frame_boxes")


def single_subject_label(pack) -> str:
    raw_entities = pack.raw_sample.get("human_entities") or []
    if raw_entities:
        label = raw_entities[0].get("person_id")
        if label:
            return clean_label(label)
    if pack.raw_sample.get("person_id"):
        return clean_label(pack.raw_sample.get("person_id"))
    if pack.references:
        return clean_label(pack.references[0].path.stem)
    return "subject"


def best_reference_label(similarities: dict[str, float], labels: dict[str, str], fallback: str) -> str:
    if not similarities:
        return fallback
    best_key = max(similarities, key=similarities.get)
    return labels.get(best_key, fallback)


def score_label(label: str, ef: float | None, face: float | None, body: float | None) -> str:
    parts = [label]
    if ef is not None:
        parts.append(f"EF:{ef:.2f}")
    if face is not None:
        parts.append(f"F:{face:.2f}")
    if body is not None:
        parts.append(f"B:{body:.2f}")
    return " ".join(parts)


def body_score_label(label: str, ef: float | None, body: float | None) -> str:
    parts = [label]
    if ef is not None:
        parts.append(f"EF:{ef:.2f}")
    if body is not None:
        parts.append(f"B:{body:.2f}")
    return " ".join(parts)


def face_score_label(label: str, face: float | None, *, cartoon: bool = False) -> str:
    parts = [label]
    if face is not None:
        parts.append(f"{'Head' if cartoon else 'F'}:{face:.2f}")
    return " ".join(parts)


def build_results_summary(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": results["sample_id"],
        "mode": results.get("mode", "single_human"),
        "condition_types": results.get("condition_types", ["human"]),
        "videos": [
            {
                "video_name": Path(video["video_path"]).name,
                "EF": video.get("EF"),
                "Overall_EF": video.get("Overall_EF"),
                "Human_EF": video.get("Human_EF"),
                "Object_EF": video.get("Object_EF"),
                "Scene_EF": video.get("Scene_EF"),
                "branch_weights": video.get("branch_weights"),
                "EF_min": video.get("EF_min"),
                "EF_visible_weighted": video.get("EF_visible_weighted"),
                "Face_EF": video.get("Face_EF"),
                "Face_Recognition": video.get("Face_Recognition"),
                "face_recognition_model": video.get("face_recognition_model"),
                "Face_ArcFace": video.get("Face_ArcFace"),
                "Face_CurricularFace": video.get("Face_CurricularFace"),
                "Cartoon_Head_EF": video.get("Cartoon_Head_EF"),
                "Body_EF": video.get("Body_EF"),
                "Object": {
                    "Object_EF": (video.get("Object") or {}).get("Object_EF"),
                    "valid_object_ratio": (video.get("Object") or {}).get("valid_object_ratio"),
                    "objects": {
                        key: {
                            "Object_EF": obj.get("Object_EF"),
                            "valid_object_ratio": obj.get("valid_object_ratio"),
                            "prompt": obj.get("prompt"),
                        }
                        for key, obj in ((video.get("Object") or {}).get("objects") or {}).items()
                    },
                }
                if video.get("Object")
                else None,
                "Scene": {
                    "Scene_EF": (video.get("Scene") or {}).get("Scene_EF"),
                    "valid_scene_ratio": (video.get("Scene") or {}).get("valid_scene_ratio"),
                }
                if video.get("Scene")
                else None,
                "entities": {
                    entity_id: {
                        "EF": entity_result.get("EF"),
                        "visible_ratio": entity_result.get("visible_ratio"),
                        "Face_EF": entity_result.get("Face_EF"),
                        "Face_Recognition": entity_result.get("Face_Recognition"),
                        "Face_ArcFace": entity_result.get("Face_ArcFace"),
                        "Face_CurricularFace": entity_result.get("Face_CurricularFace"),
                        "Cartoon_Head_EF": entity_result.get("Cartoon_Head_EF"),
                        "Body_EF": entity_result.get("Body_EF"),
                    }
                    for entity_id, entity_result in (video.get("entities") or {}).items()
                },
            }
            for video in results["videos"]
        ],
    }


def combine_video_branches(video: dict[str, Any], scene_result: dict[str, Any] | None, object_result: dict[str, Any] | None) -> dict[str, Any]:
    human_score = video.get("EF")
    video["Human_EF"] = human_score
    if scene_result is not None:
        video["Scene"] = scene_result
        video["Scene_EF"] = scene_result.get("Scene_EF")
    if object_result is not None:
        video["Object"] = object_result
        video["Object_EF"] = object_result.get("Object_EF")

    base_weights = {"human": 0.50, "object": 0.25, "scene": 0.25}
    branch_scores: dict[str, float] = {}
    if human_score is not None:
        branch_scores["human"] = float(human_score)
    if object_result is not None and object_result.get("Object_EF") is not None:
        branch_scores["object"] = float(object_result["Object_EF"])
    if scene_result is not None and scene_result.get("Scene_EF") is not None:
        branch_scores["scene"] = float(scene_result["Scene_EF"])

    weight_sum = sum(base_weights[key] for key in branch_scores)
    branch_weights = {
        key: base_weights[key] / weight_sum
        for key in branch_scores
    } if weight_sum else {}
    overall = sum(branch_weights[key] * branch_scores[key] for key in branch_scores) if branch_scores else None
    video["branch_scores"] = branch_scores
    video["branch_weights"] = branch_weights
    video["Overall_EF"] = overall
    if overall is not None:
        video["EF"] = overall
    return video


def object_frame_box_dir(args, video_path: Path) -> Path | None:
    root = frame_box_root(args)
    if root is None:
        return None
    return root / f"{video_path.stem}_object"


def reference_cache_key(args, pack) -> tuple[Any, ...]:
    return (
        pack.sample_id,
        getattr(args, "face_recognition_name", "arcface"),
        str(getattr(args, "face_recognition_model", "") or ""),
        str(getattr(args, "dinov2_model", "") or ""),
        str(getattr(args, "siglip_model", "") or ""),
        str(getattr(args, "cartoon_face_detector", "") or ""),
        bool(getattr(args, "disable_body_ef", False)),
        bool(getattr(args, "disable_cartoon_head_ef", False)),
        bool(getattr(args, "disable_reference_body_mask", False)),
        tuple((ref.key, str(ref.path)) for ref in pack.references),
    )


def build_reference_embeddings(args, pack, face_branch, body_branch, cache: dict[tuple[Any, ...], dict[str, Any]] | None = None, cartoon_face_detector=None) -> dict[str, Any]:
    key = reference_cache_key(args, pack)
    if cache is not None and key in cache:
        return cache[key]

    is_cartoon_reference = is_cartoon_reference_pack(pack)
    ref_faces = []
    ref_bodies = []
    ref_heads = []
    ref_body_mask_paths: dict[str, str | None] = {}
    for ref in pack.references:
        image = ref.pil()
        if not is_cartoon_reference:
            face = face_branch.embed_reference(ref.key, image, ref.roles)
            if face is not None:
                ref_faces.append(face)
        if is_cartoon_reference and body_branch is not None and not getattr(args, "disable_cartoon_head_ef", False):
            cartoon_face = reference_cartoon_face_image(ref, cartoon_face_detector)
            if cartoon_face is not None:
                ref_heads.append(body_branch.embed(ref.key, cartoon_face))
        if body_branch is not None and not getattr(args, "disable_body_ef", False):
            body_image, mask_path = reference_body_image(ref, args)
            ref_body_mask_paths[ref.key] = mask_path
            ref_bodies.append(body_branch.embed(ref.key, body_image))
        else:
            ref_body_mask_paths[ref.key] = None

    value = {
        "ref_faces": ref_faces,
        "ref_bodies": ref_bodies,
        "ref_heads": ref_heads,
        "is_cartoon_reference": is_cartoon_reference,
        "ref_body_mask_paths": ref_body_mask_paths,
    }
    if cache is not None:
        cache[key] = value
    return value


def evaluate_video(video_path: Path, args, pack, face_branch, body_branch, detector, reference_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None, cartoon_face_detector=None) -> dict[str, Any]:
    frames = sample_video_frames(video_path, num_frames=args.num_frames)
    box_output_dir = frame_box_root(args)
    if box_output_dir is not None:
        box_output_dir = box_output_dir / video_path.stem
    subject_label = single_subject_label(pack)
    ref_labels = {ref.key: clean_label(ref.path.stem, ref.key) for ref in pack.references}
    refs = build_reference_embeddings(args, pack, face_branch, body_branch, reference_cache, cartoon_face_detector)
    ref_faces = refs["ref_faces"]
    ref_bodies = refs["ref_bodies"]
    ref_heads = refs["ref_heads"]
    is_cartoon_reference = refs["is_cartoon_reference"]
    ref_body_mask_paths = refs["ref_body_mask_paths"]
    frame_score_threshold = min_frame_ef(args)
    cartoon_body_threshold = cartoon_min_body_ef(args)

    frame_rows = []

    for sampled in frames:
        frame_face = face_branch.embed_frame(sampled.image_rgb)
        cartoon_face_detections = cartoon_face_detector.detect(sampled.image_rgb) if is_cartoon_reference and cartoon_face_detector is not None else []
        face_valid = (not is_cartoon_reference) and frame_face is not None and bool(ref_faces)
        face_pack_score = 0.0
        face_sims: dict[str, float] = {}
        face_components = {}

        if face_valid and frame_face is not None:
            top_k = reference_top_k(args)
            face_pack_score, face_sims = face_pack_top2(frame_face, ref_faces, k=top_k)
            face_components = face_pack_component_top2(frame_face, ref_faces, k=top_k)

        face_ef = face_pack_score if face_valid else None

        body_boxes = [] if detector is None or getattr(args, "disable_body_ef", False) else detector.detect(sampled.image_rgb, frame_face.bbox if frame_face else None)
        body_valid = bool(body_boxes)
        body_score = 0.0
        body_dinov2_score = None
        body_siglip_score = None
        body_sims: dict[str, float] = {}
        if body_valid and body_branch is not None:
            body_crop = crop_rgb(sampled.image_rgb, body_boxes[0])
            frame_body = body_branch.embed("frame", body_crop)
            body_result = body_pack_top2(frame_body, ref_bodies, k=reference_top_k(args))
            body_score = body_result.overall_score
            body_sims = body_result.overall_sims
            body_dinov2_score = body_result.dinov2_score
            body_siglip_score = body_result.siglip_score

        cartoon_head_ef = None
        cartoon_head_sims: dict[str, float] = {}
        if (
            face_ef is None
            and body_valid
            and body_branch is not None
            and ref_heads
            and is_cartoon_reference
            and not getattr(args, "disable_cartoon_head_ef", False)
        ):
            cartoon_face = best_face_for_box(cartoon_face_detections, body_boxes[0])
            if cartoon_face is not None:
                head_crop = crop_rgb(sampled.image_rgb, cartoon_face.box)
                frame_head = body_branch.embed("frame_cartoon_face", head_crop)
                head_result = body_pack_top2(frame_head, ref_heads, k=reference_top_k(args))
                cartoon_head_ef = head_result.dinov2_score
                cartoon_head_sims = head_result.dinov2_sims

        face_like_ef = face_ef if face_ef is not None else cartoon_head_ef

        if not is_cartoon_reference and face_ef is None:
            continue
        if face_like_ef is None and not body_valid:
            continue
        if is_cartoon_reference and body_valid and body_score < cartoon_body_threshold:
            continue
        if face_like_ef is None:
            ef_t = body_score
        elif not body_valid:
            ef_t = face_like_ef
        else:
            ef_t = args.face_weight * face_like_ef + (1.0 - args.face_weight) * body_score

        if not passes_min_frame_ef(face_like_ef, body_score if body_valid else None, frame_score_threshold):
            continue

        frame_box_export = None
        if box_output_dir is not None:
            body_box_label = best_reference_label(body_sims, ref_labels, subject_label)
            face_box_label = best_reference_label(face_sims, ref_labels, subject_label)
            cartoon_head_box_label = best_reference_label(cartoon_head_sims, ref_labels, subject_label)
            boxes_to_draw = []
            if body_valid:
                boxes_to_draw.append((
                    body_boxes[0],
                    body_box_label,
                    body_score_label(body_box_label, ef_t, body_score),
                ))
            if face_valid and frame_face is not None:
                boxes_to_draw.append((
                    frame_face.bbox,
                    face_box_label,
                    face_score_label(face_box_label, face_ef),
                ))
            elif cartoon_head_ef is not None and body_valid:
                cartoon_face = best_face_for_box(cartoon_face_detections, body_boxes[0])
                boxes_to_draw.append((
                    cartoon_face.box if cartoon_face is not None else body_boxes[0],
                    cartoon_head_box_label,
                    face_score_label(cartoon_head_box_label, cartoon_head_ef, cartoon=True),
                ))
            frame_box_export = save_annotated_frame(
                sampled.image_rgb,
                boxes_to_draw,
                box_output_dir,
                sampled.frame_index,
                metadata={
                    "video_path": str(video_path),
                    "subject_mode": "single_human",
                    "subject_label": subject_label,
                    "box_label": body_box_label if body_valid else None,
                    "body_box_label": body_box_label if body_valid else None,
                    "face_box_label": face_box_label if face_valid else None,
                    "cartoon_head_box_label": cartoon_head_box_label if cartoon_head_ef is not None else None,
                },
            )

        head_valid = face_valid or cartoon_head_ef is not None
        head_score = frame_face.det_score if face_valid and frame_face else (body_boxes[0].score if cartoon_head_ef is not None and body_boxes else 0.0)
        q_t = quality_weight(head_valid, body_valid, head_score, body_boxes[0].score if body_boxes else 0.0)

        frame_rows.append(
            {
                "frame_index": sampled.frame_index,
                "EF_t": ef_t,
                "Face_EF_t": face_ef,
                "Face_Recognition_t": face_components.get("arcface_score"),
                "Face_ArcFace_t": face_components.get("arcface_score"),
                "Face_CurricularFace_t": face_components.get("curricular_score"),
                "Cartoon_Head_EF_t": cartoon_head_ef,
                "Body_EF_t": body_score if body_valid else None,
                "Body_DINOv2_t": body_dinov2_score if body_valid else None,
                "Body_SigLIP_t": body_siglip_score if body_valid else None,
                "face_valid": face_valid,
                "cartoon_head_valid": cartoon_head_ef is not None,
                "body_valid": body_valid,
                "quality_weight": q_t,
                "frame_box_export": frame_box_export,
            }
        )

    if not frame_rows:
        return {
            "sample_id": pack.sample_id,
            "video_path": str(video_path),
            "EF": 0.5,
            "error": "No valid frame had face or body evidence.",
            "frame_scores": [],
        }

    weights = np.asarray([row["quality_weight"] for row in frame_rows], dtype=np.float32)
    if float(weights.sum()) == 0.0:
        weights = np.ones_like(weights)
    ef_values = np.asarray([row["EF_t"] for row in frame_rows], dtype=np.float32)

    face_values = [row["Face_EF_t"] for row in frame_rows if row["Face_EF_t"] is not None]
    face_recognition_values = [row["Face_Recognition_t"] for row in frame_rows if row.get("Face_Recognition_t") is not None]
    face_arc_values = [row["Face_ArcFace_t"] for row in frame_rows if row.get("Face_ArcFace_t") is not None]
    face_curricular_values = [row["Face_CurricularFace_t"] for row in frame_rows if row.get("Face_CurricularFace_t") is not None]
    cartoon_head_values = [row["Cartoon_Head_EF_t"] for row in frame_rows if row.get("Cartoon_Head_EF_t") is not None]
    body_values = [row["Body_EF_t"] for row in frame_rows if row["Body_EF_t"] is not None]
    body_dinov2_values = [row["Body_DINOv2_t"] for row in frame_rows if row["Body_DINOv2_t"] is not None]
    body_siglip_values = [row["Body_SigLIP_t"] for row in frame_rows if row["Body_SigLIP_t"] is not None]

    human_ef = float(np.sum(weights * ef_values) / np.sum(weights))
    if human_ef == 0.0:
        human_ef = 0.5

    return {
        "sample_id": pack.sample_id,
        "video_path": str(video_path),
        "EF": human_ef,
        "Face_EF": float(np.mean(face_values)) if face_values else None,
        "Face_Recognition": float(np.mean(face_recognition_values)) if face_recognition_values else None,
        "face_recognition_model": args.face_recognition_name,
        "is_cartoon_reference": is_cartoon_reference,
        "Face_ArcFace": float(np.mean(face_arc_values)) if face_arc_values else None,
        "Face_CurricularFace": float(np.mean(face_curricular_values)) if face_curricular_values else None,
        "Cartoon_Head_EF": float(np.mean(cartoon_head_values)) if cartoon_head_values else None,
        "Body_EF": float(np.mean(body_values)) if body_values else None,
        "Body_DINOv2": float(np.mean(body_dinov2_values)) if body_dinov2_values else None,
        "Body_SigLIP": float(np.mean(body_siglip_values)) if body_siglip_values else None,
        "valid_face_ratio": float(len(face_values) / len(frames)) if frames else 0.0,
        "valid_cartoon_head_ratio": float(len(cartoon_head_values) / len(frames)) if frames else 0.0,
        "valid_body_ratio": float(len(body_values) / len(frames)) if frames else 0.0,
        "sampled_frames": len(frames),
        "used_frames": len(frame_rows),
        "reference_body_masks": ref_body_mask_paths,
        "frame_scores": frame_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-json", required=True)
    parser.add_argument("--video-folder", required=True)
    parser.add_argument("--video-glob", default="*.mp4")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--face-weight", type=float, default=0.35)
    parser.add_argument("--reference-top-k", type=int, default=1, help="Top-k reference views to average for face/body similarity.")
    parser.add_argument("--min-frame-ef", type=float, default=DEFAULT_MIN_FRAME_EF, help="Drop frames whose available face/cartoon-head or body EF is below this threshold.")
    parser.add_argument("--cartoon-min-body-ef", type=float, default=DEFAULT_CARTOON_MIN_BODY_EF, help="For cartoon references, drop frame evidence when body EF is below this threshold.")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--face-model-root", required=True)
    parser.add_argument("--curricular-checkpoint")
    parser.add_argument("--face-recognition-model", help="Optional InsightFace-compatible recognition ONNX to replace the default model-pack embedding.")
    parser.add_argument("--face-recognition-name", default="arcface", help="Label stored in result JSON for the selected face recognition model.")
    parser.add_argument("--dinov2-model")
    parser.add_argument("--siglip-model")
    parser.add_argument("--reference-human-root", default=str(DEFAULT_HUMAN_ROOT))
    parser.add_argument("--reference-body-mask-root", default=str(DEFAULT_REFERENCE_BODY_MASK_ROOT))
    parser.add_argument("--disable-reference-body-mask", action="store_true")
    parser.add_argument("--disable-body-ef", action="store_true", help="Skip detector/body branch loading and compute human EF from face evidence only.")
    parser.add_argument("--save-frame-boxes", action="store_true")
    parser.add_argument("--frame-box-dir", help="Directory for sampled frames annotated with detected subject boxes.")
    parser.add_argument("--disable-frame-box-export", action="store_true")
    parser.add_argument("--draw-dropped-boxes", action="store_true")

    parser.add_argument("--detector", choices=["yolo-world", "face-expanded"], default="yolo-world")
    parser.add_argument(
        "--yoloworld-config",
        default="OpenS2V-Nexus/eval/utils/yoloworld/configs/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_image_prompt_demo.py",
    )
    parser.add_argument("--yoloworld-checkpoint")
    parser.add_argument("--yoloworld-repo", default="OpenS2V-Nexus/eval/utils/yoloworld")
    parser.add_argument("--detector-device", default="cuda:1")
    parser.add_argument("--person-score-threshold", type=float, default=0.15)
    parser.add_argument("--person-detection-text", action="append", dest="person_detection_texts", help="YOLO-World text prompt for human/body detection. Can be repeated. Defaults to sample JSON person_detection_prompts or 'person'.")
    parser.add_argument("--object-score-threshold", type=float, default=0.05)
    parser.add_argument("--disable-object-ef", action="store_true")
    parser.add_argument("--disable-scene-ef", action="store_true")
    parser.add_argument("--disable-cartoon-head-ef", action="store_true", help="Disable SigLIP head-crop fallback when InsightFace cannot detect reference faces.")
    parser.add_argument("--cartoon-face-detector", choices=["hysts-yolov3", "yolov8-anime-face", "none"], default="hysts-yolov3", help="Detector used for cartoon face EF. If no cartoon face is detected, EF falls back to body only.")
    parser.add_argument("--cartoon-face-score-threshold", type=float, default=0.15)
    parser.add_argument("--hysts-yolov3-config", default="ef_eval/model_assets/anime_face_detectors/mmdet3/mmdet_anime-face_yolov3.py")
    parser.add_argument("--hysts-yolov3-checkpoint", default="ef_eval/model_assets/anime_face_detectors/mmdet3/mmdet_anime-face_yolov3.pth")
    parser.add_argument("--yolov8-anime-face-checkpoint", default="ef_eval/model_assets/anime_face_detectors/yolov8/yolov8x6_animeface.pt")
    parser.add_argument("--yolov8-device", default="0")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.disable_body_ef and args.detector == "yolo-world" and not args.yoloworld_checkpoint:
        raise ValueError("--yoloworld-checkpoint is required when --detector yolo-world.")

    human_pack = load_human_reference_pack(args.sample_json)
    pack = load_reference_pack(args.sample_json)
    if not args.person_detection_texts:
        prompts = pack.raw_sample.get("person_detection_prompts")
        if isinstance(prompts, list) and prompts:
            args.person_detection_texts = [str(item) for item in prompts if str(item).strip()]
    object_refs = [] if args.disable_object_ef or args.disable_body_ef else load_condition_references(args.sample_json, "object")
    scene_refs = [] if args.disable_scene_ef or args.disable_body_ef else load_condition_references(args.sample_json, "scene")
    detector = None if args.disable_body_ef else build_detector(args)
    face_branch = FaceBranch(
        args.face_model_root,
        args.curricular_checkpoint,
        recognition_model=args.face_recognition_model,
        recognition_name=args.face_recognition_name,
        device=args.device,
    )
    body_branch = None if args.disable_body_ef else BodyBranch(args.dinov2_model, args.siglip_model, device=args.device)
    scene_branch = SceneBranch(body_branch, scene_refs) if scene_refs else None
    object_branch = (
        ObjectBranch(body_branch, object_refs, detector_factory=lambda prompt: build_text_detector(args, prompt))
        if object_refs
        else None
    )
    cartoon_face_detector = (
        build_cartoon_face_detector(args)
        if body_branch is not None and not args.disable_cartoon_head_ef and (has_cartoon_human(human_pack) or has_cartoon_human(pack))
        else None
    )

    video_folder = Path(args.video_folder)
    video_paths = sorted(video_folder.glob(args.video_glob))
    if not video_paths:
        raise FileNotFoundError(f"No videos matched {args.video_glob!r} under {video_folder}")
    if len(human_pack.entities) > 1:
        multi_evaluator = MultiEntityFidelityEvaluator(
            face_branch,
            body_branch,
            detector,
            face_weight=args.face_weight,
            reference_top_k=reference_top_k(args),
            reference_body_image_fn=lambda ref: reference_body_image(ref, args),
            cartoon_face_detector=cartoon_face_detector,
        )
        videos = []
        for video_path in video_paths:
            human_video = multi_evaluator.evaluate_video(video_path, args, human_pack)
            scene_result = scene_branch.evaluate_video(video_path, args.num_frames) if scene_branch else None
            object_result = (
                object_branch.evaluate_video(video_path, args.num_frames, object_frame_box_dir(args, video_path))
                if object_branch
                else None
            )
            videos.append(combine_video_branches(human_video, scene_result, object_result))

        results = {
            "sample_id": human_pack.sample_id,
            "mode": "multi_human",
            "condition_types": ["human"] + (["object"] if object_refs else []) + (["scene"] if scene_refs else []),
            "human_entities": [
                {
                    "entity_id": entity.entity_id,
                    "person_id": entity.person_id,
                    "reference_images": [str(ref.path) for ref in entity.references],
                }
                for entity in human_pack.entities
            ],
            "object_references": [
                {"key": ref.key, "path": str(ref.path), "description": ref.description, "style": ref.style}
                for ref in object_refs
            ],
            "scene_references": [
                {"key": ref.key, "path": str(ref.path), "description": ref.description, "style": ref.style}
                for ref in scene_refs
            ],
            "videos": videos,
        }
    else:
        videos = []
        for video_path in video_paths:
            human_video = evaluate_video(video_path, args, pack, face_branch, body_branch, detector, cartoon_face_detector=cartoon_face_detector)
            scene_result = scene_branch.evaluate_video(video_path, args.num_frames) if scene_branch else None
            object_result = (
                object_branch.evaluate_video(video_path, args.num_frames, object_frame_box_dir(args, video_path))
                if object_branch
                else None
            )
            videos.append(combine_video_branches(human_video, scene_result, object_result))

        results = {
            "sample_id": pack.sample_id,
            "mode": "single_human",
            "condition_types": ["human"] + (["object"] if object_refs else []) + (["scene"] if scene_refs else []),
            "reference_roles": {ref.key: ref.roles for ref in pack.references},
            "object_references": [
                {"key": ref.key, "path": str(ref.path), "description": ref.description, "style": ref.style}
                for ref in object_refs
            ],
            "scene_references": [
                {"key": ref.key, "path": str(ref.path), "description": ref.description, "style": ref.style}
                for ref in scene_refs
            ],
            "videos": videos,
        }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = output_path.parent / "summary.json" if output_path.name == "result.json" else output_path.with_name(f"{output_path.stem}_summary{output_path.suffix}")
    summary_path.write_text(json.dumps(build_results_summary(results), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved EF results to {output_path}")
    print(f"Saved EF summary to {summary_path}")


if __name__ == "__main__":
    main()
