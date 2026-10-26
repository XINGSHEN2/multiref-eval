from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from body_branch import BodyBranch, body_pack_top2
from condition_pack import ConditionReference, object_prompt
from frame_box_visualization import save_annotated_frame
from frame_sampler import sample_video_frames
from subject_detector import Box, YOLOWorldPersonDetector, crop_rgb


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _box_size(box: Box) -> tuple[float, float]:
    return max(0.0, box.x2 - box.x1), max(0.0, box.y2 - box.y1)


def _is_valid_embedding_box(box: Box, min_side: float = 8.0, min_area: float = 256.0) -> bool:
    width, height = _box_size(box)
    return width >= min_side and height >= min_side and width * height >= min_area


class ObjectBranch:
    def __init__(
        self,
        body_branch: BodyBranch,
        refs: list[ConditionReference],
        detector_factory: Callable[[str], YOLOWorldPersonDetector],
    ) -> None:
        self.body_branch = body_branch
        self.refs = refs
        self.ref_embeddings = {
            ref.key: body_branch.embed(ref.key, Image.open(ref.path).convert("RGB"))
            for ref in refs
        }
        self.prompts = {ref.key: object_prompt(ref) for ref in refs}
        self.detectors = {ref.key: detector_factory(self.prompts[ref.key]) for ref in refs}

    def _evaluate_ref_video(self, ref: ConditionReference, video_path: Path, frames) -> dict[str, Any]:
        detector = self.detectors[ref.key]
        ref_embedding = self.ref_embeddings[ref.key]
        rows = []
        for sampled in frames:
            boxes = detector.detect(sampled.image_rgb, None)
            if not boxes:
                rows.append(
                    {
                        "frame_index": sampled.frame_index,
                        "EF_t": None,
                        "Object_EF_t": None,
                        "Object_DINOv2_t": None,
                        "Object_SigLIP_t": None,
                        "valid": False,
                        "object_box": None,
                    }
                )
                continue

            valid_boxes = [box for box in boxes if _is_valid_embedding_box(box)]
            if not valid_boxes:
                rows.append(
                    {
                        "frame_index": sampled.frame_index,
                        "EF_t": None,
                        "Object_EF_t": None,
                        "Object_DINOv2_t": None,
                        "Object_SigLIP_t": None,
                        "valid": False,
                        "object_box": None,
                        "skipped_reason": "no_valid_object_box",
                    }
                )
                continue

            box = valid_boxes[0]
            crop = crop_rgb(sampled.image_rgb, box)
            frame_embedding = self.body_branch.embed(f"{ref.key}_frame", crop)
            result = body_pack_top2(frame_embedding, [ref_embedding])
            rows.append(
                {
                    "frame_index": sampled.frame_index,
                    "EF_t": result.overall_score,
                    "Object_EF_t": result.overall_score,
                    "Object_DINOv2_t": result.dinov2_score,
                    "Object_SigLIP_t": result.siglip_score,
                    "valid": True,
                    "object_box": box.__dict__,
                    "similarities": result.overall_sims,
                }
            )

        valid_rows = [row for row in rows if row.get("EF_t") is not None]
        values = [row["EF_t"] for row in valid_rows]
        dinov2_values = [row["Object_DINOv2_t"] for row in valid_rows if row["Object_DINOv2_t"] is not None]
        siglip_values = [row["Object_SigLIP_t"] for row in valid_rows if row["Object_SigLIP_t"] is not None]
        return {
            "key": ref.key,
            "path": str(ref.path),
            "description": ref.description,
            "style": ref.style,
            "prompt": self.prompts[ref.key],
            "Object_EF": _mean(values),
            "Object_DINOv2": _mean(dinov2_values),
            "Object_SigLIP": _mean(siglip_values),
            "valid_object_ratio": float(len(valid_rows) / len(frames)) if frames else 0.0,
            "used_frames": len(valid_rows),
            "frame_scores": rows,
        }

    def evaluate_video(self, video_path: Path, num_frames: int, frame_box_dir: Path | None = None) -> dict[str, Any]:
        frames = sample_video_frames(video_path, num_frames=num_frames)
        objects = {
            ref.key: self._evaluate_ref_video(ref, video_path, frames)
            for ref in self.refs
        }
        valid_scores = [
            obj["Object_EF"] for obj in objects.values() if obj.get("Object_EF") is not None
        ]

        aggregate_rows = []
        if frames:
            for sampled in frames:
                object_boxes = []
                per_frame = [
                    obj_row["EF_t"]
                    for obj in objects.values()
                    for obj_row in obj.get("frame_scores") or []
                    if obj_row.get("frame_index") == sampled.frame_index and obj_row.get("EF_t") is not None
                ]
                for obj in objects.values():
                    for obj_row in obj.get("frame_scores") or []:
                        if obj_row.get("frame_index") != sampled.frame_index or not obj_row.get("object_box"):
                            continue
                        box_dict = obj_row["object_box"]
                        object_boxes.append(
                            (
                                Box(
                                    x1=box_dict["x1"],
                                    y1=box_dict["y1"],
                                    x2=box_dict["x2"],
                                    y2=box_dict["y2"],
                                    score=box_dict.get("score", 1.0),
                                    label=box_dict.get("label", "object"),
                                ),
                                obj.get("prompt") or obj.get("key") or "object",
                            )
                        )
                frame_box_export = (
                    save_annotated_frame(
                        sampled.image_rgb,
                        object_boxes,
                        frame_box_dir,
                        sampled.frame_index,
                        metadata={
                            "video_path": str(video_path),
                            "subject_mode": "object",
                            "detected_object_count": len(object_boxes),
                        },
                    )
                    if frame_box_dir is not None
                    else None
                )
                aggregate_rows.append(
                    {
                        "frame_index": sampled.frame_index,
                        "EF_t": float(np.mean(per_frame)) if per_frame else None,
                        "Object_EF_t": float(np.mean(per_frame)) if per_frame else None,
                        "valid_object_count": len(per_frame),
                        "frame_box_export": frame_box_export,
                    }
                )

        valid_ratios = [obj["valid_object_ratio"] for obj in objects.values()]
        return {
            "Object_EF": _mean(valid_scores),
            "Object_EF_min": float(np.min(valid_scores)) if valid_scores else None,
            "valid_object_ratio": _mean(valid_ratios) or 0.0,
            "object_references": [
                {
                    "key": ref.key,
                    "path": str(ref.path),
                    "description": ref.description,
                    "style": ref.style,
                    "prompt": self.prompts[ref.key],
                }
                for ref in self.refs
            ],
            "objects": objects,
            "frame_scores": aggregate_rows,
        }
