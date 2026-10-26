from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from body_branch import BodyBranch, body_pack_top2
from cartoon_face_detector import best_face_for_box
from face_branch import FaceBranch, FaceEmbedding, face_pack_component_top2, face_pack_top2
from frame_box_visualization import clean_label, save_annotated_frame
from frame_sampler import sample_video_frames
from reference_pack import HumanEntityPack, HumanReferencePack
from subject_detector import Box, crop_rgb, expand_face_to_body

DEFAULT_CARTOON_MIN_BODY_EF = 0.6


def _box_iou(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0.0 else 0.0


def _contains_center(outer: Box, inner: Box) -> bool:
    cx = (inner.x1 + inner.x2) / 2.0
    cy = (inner.y1 + inner.y2) / 2.0
    return outer.x1 <= cx <= outer.x2 and outer.y1 <= cy <= outer.y2


def _body_box_for_face(face_box: Box, body_boxes: list[Box], image_shape: tuple[int, int, int]) -> Box:
    candidates = [box for box in body_boxes if _contains_center(box, face_box)]
    if not candidates:
        return expand_face_to_body(face_box, image_shape)
    return max(candidates, key=lambda box: (_box_iou(box, expand_face_to_body(face_box, image_shape)), box.score))


def _quality_weight(face_score: float, body_score: float) -> float:
    return 0.5 + 0.25 * max(0.0, min(1.0, face_score)) + 0.25 * max(0.0, min(1.0, body_score))


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _passes_min_frame_ef(face_like_ef: float | None, body_ef: float | None, threshold: float) -> bool:
    if face_like_ef is not None and face_like_ef < threshold:
        return False
    if body_ef is not None and body_ef < threshold:
        return False
    return True


def _cartoon_min_body_ef(args) -> float:
    return float(getattr(args, "cartoon_min_body_ef", DEFAULT_CARTOON_MIN_BODY_EF))


def _entity_label(entity: HumanEntityPack) -> str:
    if entity.person_id:
        return clean_label(entity.person_id, entity.entity_id)
    if entity.references:
        return clean_label(entity.references[0].path.stem, entity.entity_id)
    return entity.entity_id


def _contains_cartoon(value: Any) -> bool:
    if isinstance(value, str):
        return "cartoon" in value.lower()
    if isinstance(value, dict):
        return any(_contains_cartoon(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_cartoon(item) for item in value)
    return False


def _is_cartoon_entity(entity: HumanEntityPack) -> bool:
    if entity.person_id and "cartoon" in entity.person_id.lower():
        return True
    if any("cartoon" in str(ref.path).lower() for ref in entity.references):
        return True
    return _contains_cartoon(entity.raw_entity)


def _head_box_from_body_box(body_box: Box, head_ratio: float = 0.45) -> Box:
    height = max(1.0, body_box.y2 - body_box.y1)
    return Box(
        x1=body_box.x1,
        y1=body_box.y1,
        x2=body_box.x2,
        y2=body_box.y1 + height * head_ratio,
        score=body_box.score,
        label="cartoon_head",
    )


def _reference_head_crop(image: Image.Image, head_ratio: float = 0.55) -> Image.Image:
    width, height = image.size
    return image.convert("RGB").crop((0, 0, width, max(1, int(round(height * head_ratio)))))


def _frame_box_root(args) -> Path | None:
    if getattr(args, "disable_frame_box_export", False):
        return None
    frame_box_dir = getattr(args, "frame_box_dir", None)
    if not getattr(args, "save_frame_boxes", False) and not frame_box_dir:
        return None
    if frame_box_dir:
        return Path(frame_box_dir)
    output_path = Path(args.output_json)
    return output_path.with_name(f"{output_path.stem}_frame_boxes")


def _best_reference_label(similarities: dict[str, float], labels: dict[str, str], fallback: str) -> str:
    if not similarities:
        return fallback
    best_key = max(similarities, key=similarities.get)
    return labels.get(best_key, fallback)


def _score_label(label: str, ef: float | None, face: float | None, body: float | None) -> str:
    parts = [label]
    if ef is not None:
        parts.append(f"EF:{ef:.2f}")
    if face is not None:
        parts.append(f"F:{face:.2f}")
    if body is not None:
        parts.append(f"B:{body:.2f}")
    return " ".join(parts)


def _body_score_label(label: str, ef: float | None, body: float | None) -> str:
    parts = [label]
    if ef is not None:
        parts.append(f"EF:{ef:.2f}")
    if body is not None:
        parts.append(f"B:{body:.2f}")
    return " ".join(parts)


def _face_score_label(label: str, face: float | None, *, cartoon: bool = False) -> str:
    parts = [label]
    if face is not None:
        parts.append(f"{'Head' if cartoon else 'F'}:{face:.2f}")
    return " ".join(parts)


class MultiEntityFidelityEvaluator:
    def __init__(
        self,
        face_branch: FaceBranch,
        body_branch: BodyBranch,
        detector,
        face_weight: float = 0.6,
        reference_top_k: int = 1,
        reference_body_image_fn: Callable[[Any], tuple[Image.Image, str | None]] | None = None,
        cartoon_face_detector=None,
    ) -> None:
        self.face_branch = face_branch
        self.body_branch = body_branch
        self.detector = detector
        self.face_weight = face_weight
        self.reference_top_k = max(1, int(reference_top_k))
        self.reference_body_image_fn = reference_body_image_fn
        self.cartoon_face_detector = cartoon_face_detector

    def _embed_entity_faces(self, entity: HumanEntityPack) -> list[FaceEmbedding]:
        if _is_cartoon_entity(entity):
            return []
        faces: list[FaceEmbedding] = []
        for ref in entity.references:
            face = self.face_branch.embed_reference(ref.key, ref.pil(), ref.roles)
            if face is not None:
                faces.append(face)
        return faces

    def _embed_entity_heads(self, entity: HumanEntityPack):
        if not _is_cartoon_entity(entity):
            return []
        heads = []
        for ref in entity.references:
            if self.cartoon_face_detector is None:
                continue
            image = ref.pil()
            image_np = np.asarray(image.convert("RGB"))
            face = best_face_for_box(self.cartoon_face_detector.detect(image_np))
            if face is not None:
                heads.append(self.body_branch.embed(ref.key, crop_rgb(image_np, face.box)))
        return heads

    def _embed_entity_bodies(self, entity: HumanEntityPack):
        bodies = []
        for ref in entity.references:
            image = self.reference_body_image_fn(ref)[0] if self.reference_body_image_fn else ref.pil()
            bodies.append(self.body_branch.embed(ref.key, image))
        return bodies

    def evaluate_video(self, video_path: Path, args, human_pack: HumanReferencePack) -> dict[str, Any]:
        frames = sample_video_frames(video_path, num_frames=args.num_frames)
        entity_refs: dict[str, dict[str, Any]] = {}
        for entity in human_pack.entities:
            entity_refs[entity.entity_id] = {
                "entity": entity,
                "label": _entity_label(entity),
                "ref_labels": {ref.key: clean_label(ref.path.stem, ref.key) for ref in entity.references},
                "is_cartoon_reference": _is_cartoon_entity(entity),
                "face_refs": self._embed_entity_faces(entity),
                "head_refs": self._embed_entity_heads(entity),
                "body_refs": self._embed_entity_bodies(entity),
            }
        box_output_dir = _frame_box_root(args)
        if box_output_dir is not None:
            box_output_dir = box_output_dir / video_path.stem

        per_entity_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        frame_rows: list[dict[str, Any]] = []
        identity_switch_count = 0
        previous_detection_assignment: dict[str, int] = {}
        frame_score_threshold = float(getattr(args, "min_frame_ef", 0.1))
        cartoon_body_threshold = _cartoon_min_body_ef(args)

        for sampled in frames:
            frame_faces = self.face_branch.embed_frame_faces(sampled.image_rgb)
            cartoon_faces = self.cartoon_face_detector.detect(sampled.image_rgb) if self.cartoon_face_detector is not None else []
            body_boxes = self.detector.detect(sampled.image_rgb, None)
            detections = []
            used_face_indices: set[int] = set()
            for body_index, body_box in enumerate(body_boxes):
                candidate_faces = [
                    (face_index, face)
                    for face_index, face in enumerate(frame_faces)
                    if _contains_center(body_box, face.bbox)
                ]
                face_index, frame_face = max(candidate_faces, key=lambda item: item[1].det_score) if candidate_faces else (None, None)
                if face_index is not None:
                    used_face_indices.add(face_index)
                body_crop = crop_rgb(sampled.image_rgb, body_box)
                cartoon_face = best_face_for_box(cartoon_faces, body_box)
                detections.append(
                    {
                        "detection_index": body_index,
                        "face_index": face_index,
                        "face": frame_face,
                        "cartoon_face": cartoon_face,
                        "body_box": body_box,
                        "head": self.body_branch.embed(f"frame_cartoon_face_{body_index}", crop_rgb(sampled.image_rgb, cartoon_face.box)) if cartoon_face is not None else None,
                        "body": self.body_branch.embed(f"frame_face_{face_index}", body_crop),
                    }
                )
            for face_index, frame_face in enumerate(frame_faces):
                if face_index in used_face_indices:
                    continue
                body_box = _body_box_for_face(frame_face.bbox, body_boxes, sampled.image_rgb.shape)
                body_crop = crop_rgb(sampled.image_rgb, body_box)
                cartoon_face = best_face_for_box(cartoon_faces, body_box)
                detections.append(
                    {
                        "detection_index": len(detections),
                        "face_index": face_index,
                        "face": frame_face,
                        "cartoon_face": cartoon_face,
                        "body_box": body_box,
                        "head": self.body_branch.embed(f"frame_cartoon_face_{face_index}", crop_rgb(sampled.image_rgb, cartoon_face.box)) if cartoon_face is not None else None,
                        "body": self.body_branch.embed(f"frame_face_{face_index}", body_crop),
                    }
                )

            candidates = []
            for entity_id, refs in entity_refs.items():
                face_refs = refs["face_refs"]
                head_refs = refs["head_refs"]
                body_refs = refs["body_refs"]
                is_cartoon_reference = refs["is_cartoon_reference"]
                if not face_refs and not head_refs and not body_refs:
                    continue
                for det in detections:
                    face_pack_score = 0.0
                    face_sims: dict[str, float] = {}
                    face_components = {}
                    face_ef = None
                    cartoon_head_ef = None
                    cartoon_head_sims: dict[str, float] = {}
                    if not is_cartoon_reference and det["face"] is not None and face_refs:
                        face_pack_score, face_sims = face_pack_top2(det["face"], face_refs, k=self.reference_top_k)
                        face_components = face_pack_component_top2(det["face"], face_refs, k=self.reference_top_k)
                        face_ef = face_pack_score
                    if is_cartoon_reference and head_refs and det["head"] is not None:
                        head_result = body_pack_top2(det["head"], head_refs, k=self.reference_top_k)
                        cartoon_head_ef = head_result.dinov2_score
                        cartoon_head_sims = head_result.dinov2_sims
                    if not is_cartoon_reference and face_ef is None:
                        continue
                    face_like_ef = face_ef if face_ef is not None else cartoon_head_ef

                    body_result = body_pack_top2(det["body"], body_refs, k=self.reference_top_k) if body_refs else None
                    body_score = body_result.overall_score if body_result else None
                    body_sims = body_result.overall_sims if body_result else {}

                    if face_like_ef is None and body_score is None:
                        continue
                    if is_cartoon_reference and body_score is not None and body_score < cartoon_body_threshold:
                        continue
                    if not _passes_min_frame_ef(face_like_ef, body_score, frame_score_threshold):
                        continue
                    if face_like_ef is None:
                        ef_t = float(body_score)
                    elif body_score is None:
                        ef_t = float(face_like_ef)
                    else:
                        ef_t = self.face_weight * face_like_ef + (1.0 - self.face_weight) * body_score
                    match_score = 0.4 * (face_like_ef or 0.0) + 0.6 * (body_score or 0.0)
                    body_box_label = _best_reference_label(body_sims, refs["ref_labels"], refs["label"])
                    face_box_label = _best_reference_label(face_sims, refs["ref_labels"], refs["label"])
                    cartoon_head_box_label = _best_reference_label(cartoon_head_sims, refs["ref_labels"], refs["label"])
                    candidates.append(
                        {
                            "entity_id": entity_id,
                            "entity_label": refs["label"],
                            "box_label": body_box_label,
                            "body_box_label": body_box_label,
                            "face_box_label": face_box_label if face_ef is not None else None,
                            "cartoon_head_box_label": cartoon_head_box_label if cartoon_head_ef is not None else None,
                            "detection_index": det["detection_index"],
                            "face_index": det["face_index"],
                            "match_score": float(match_score),
                            "EF_t": float(ef_t),
                            "Face_EF_t": float(face_ef) if face_ef is not None else None,
                            "Face_Recognition_t": face_components.get("arcface_score"),
                            "Face_ArcFace_t": face_components.get("arcface_score"),
                            "Face_CurricularFace_t": face_components.get("curricular_score"),
                            "Cartoon_Head_EF_t": float(cartoon_head_ef) if cartoon_head_ef is not None else None,
                            "Body_EF_t": float(body_score) if body_score is not None else None,
                            "quality_weight": _quality_weight(
                                det["face"].det_score if face_ef is not None and det["face"] else det["cartoon_face"].box.score if cartoon_head_ef is not None and det["cartoon_face"] is not None else 0.0,
                                det["body_box"].score,
                            ),
                            "face_box": det["face"].bbox.__dict__ if det["face"] else None,
                            "cartoon_face_box": det["cartoon_face"].box.__dict__ if det["cartoon_face"] is not None else None,
                            "body_box": det["body_box"].__dict__,
                        }
                    )

            used_entities: set[str] = set()
            used_detections: set[int] = set()
            assignments = []
            dropped_assignments = []
            best_candidate_by_detection: dict[int, dict[str, Any]] = {}
            for candidate in sorted(candidates, key=lambda item: item["match_score"], reverse=True):
                best_candidate_by_detection.setdefault(candidate["detection_index"], candidate)
            for candidate in sorted(candidates, key=lambda item: item["match_score"], reverse=True):
                if candidate["entity_id"] in used_entities or candidate["detection_index"] in used_detections:
                    continue
                if len(assignments) >= len(human_pack.entities):
                    dropped_assignments.append({**candidate, "drop_reason": "more_detections_than_json_entities"})
                    used_detections.add(candidate["detection_index"])
                    continue
                used_entities.add(candidate["entity_id"])
                used_detections.add(candidate["detection_index"])
                assignments.append(candidate)
                per_entity_rows[candidate["entity_id"]].append(
                    {"frame_index": sampled.frame_index, **candidate}
                )

                previous = previous_detection_assignment.get(candidate["entity_id"])
                if previous is not None and previous != candidate["detection_index"]:
                    identity_switch_count += 1
                previous_detection_assignment[candidate["entity_id"]] = candidate["detection_index"]

            assignment_by_detection = {item["detection_index"]: item for item in assignments}
            dropped_by_detection = {item["detection_index"]: item for item in dropped_assignments}
            boxes_to_draw = []
            for det in detections:
                assignment = assignment_by_detection.get(det["detection_index"])
                dropped = dropped_by_detection.get(det["detection_index"])
                best_candidate = best_candidate_by_detection.get(det["detection_index"])
                if assignment:
                    body_label = assignment.get("body_box_label") or assignment["box_label"]
                    boxes_to_draw.append((
                        det["body_box"],
                        body_label,
                        _body_score_label(body_label, assignment.get("EF_t"), assignment.get("Body_EF_t")),
                    ))
                    if assignment.get("Face_EF_t") is not None and assignment.get("face_box"):
                        face_label = assignment.get("face_box_label") or assignment["entity_label"]
                        boxes_to_draw.append((
                            Box(**assignment["face_box"]),
                            face_label,
                            _face_score_label(face_label, assignment.get("Face_EF_t")),
                        ))
                    elif assignment.get("Cartoon_Head_EF_t") is not None:
                        head_label = assignment.get("cartoon_head_box_label") or assignment["entity_label"]
                        boxes_to_draw.append((
                            Box(**assignment["cartoon_face_box"]) if assignment.get("cartoon_face_box") else det["body_box"],
                            head_label,
                            _face_score_label(head_label, assignment.get("Cartoon_Head_EF_t"), cartoon=True),
                        ))
                elif getattr(args, "draw_dropped_boxes", False):
                    label = (
                        f"drop_{dropped['box_label']}" if dropped else
                        best_candidate["box_label"] if best_candidate else f"unmatched_{det['detection_index'] + 1}"
                    )
                    boxes_to_draw.append((det["body_box"], label))

            frame_box_export = None
            if box_output_dir is not None and boxes_to_draw:
                frame_box_export = save_annotated_frame(
                    sampled.image_rgb,
                    boxes_to_draw,
                    box_output_dir,
                    sampled.frame_index,
                    metadata={
                        "video_path": str(video_path),
                        "subject_mode": "multi_human",
                        "assigned_entities": [
                            {"entity_id": item["entity_id"], "label": item["box_label"]}
                            for item in assignments
                        ],
                        "dropped_entities": [
                            {
                                "entity_id": item["entity_id"],
                                "label": item["box_label"],
                                "reason": item["drop_reason"],
                            }
                            for item in dropped_assignments
                        ],
                    },
                )

            frame_rows.append(
                {
                    "frame_index": sampled.frame_index,
                    "detected_face_count": len(frame_faces),
                    "detected_body_count": len(detections),
                    "assigned_entity_count": len(assignments),
                    "assignments": assignments,
                    "dropped_assignments": dropped_assignments,
                    "unmatched_entities": [
                        entity.entity_id for entity in human_pack.entities if entity.entity_id not in used_entities
                    ],
                    "frame_box_export": frame_box_export,
                }
            )

        entity_results = {}
        for entity in human_pack.entities:
            rows = per_entity_rows.get(entity.entity_id, [])
            weights = np.asarray([row["quality_weight"] for row in rows], dtype=np.float32)
            if len(weights) and float(weights.sum()) == 0.0:
                weights = np.ones_like(weights)
            ef_values = np.asarray([row["EF_t"] for row in rows], dtype=np.float32)
            entity_ef = float(np.sum(weights * ef_values) / np.sum(weights)) if len(rows) else None
            if entity_ef == 0.0:
                entity_ef = 0.5
            entity_results[entity.entity_id] = {
                "person_id": entity.person_id,
                "EF": entity_ef,
                "Face_EF": _mean([row["Face_EF_t"] for row in rows if row["Face_EF_t"] is not None]),
                "Face_Recognition": _mean([row["Face_Recognition_t"] for row in rows if row.get("Face_Recognition_t") is not None]),
                "Face_ArcFace": _mean([row["Face_ArcFace_t"] for row in rows if row.get("Face_ArcFace_t") is not None]),
                "Face_CurricularFace": _mean([row["Face_CurricularFace_t"] for row in rows if row.get("Face_CurricularFace_t") is not None]),
                "Cartoon_Head_EF": _mean([row["Cartoon_Head_EF_t"] for row in rows if row.get("Cartoon_Head_EF_t") is not None]),
                "Body_EF": _mean([row["Body_EF_t"] for row in rows if row["Body_EF_t"] is not None]),
                "is_cartoon_reference": _is_cartoon_entity(entity),
                "visible_ratio": float(len(rows) / len(frames)) if frames else 0.0,
                "used_frames": len(rows),
                "frame_scores": rows,
            }

        entity_scores_with_missing = [
            value["EF"] if value.get("EF") is not None else 0.5 for value in entity_results.values()
        ]
        valid_entity_scores = [
            value["EF"] for value in entity_results.values() if value.get("EF") is not None
        ]
        visible_weights = np.asarray(
            [value["visible_ratio"] for value in entity_results.values() if value.get("EF") is not None],
            dtype=np.float32,
        )
        visible_scores = np.asarray(valid_entity_scores, dtype=np.float32)
        if len(visible_weights) and float(visible_weights.sum()) == 0.0:
            visible_weights = np.ones_like(visible_weights)

        return {
            "sample_id": human_pack.sample_id,
            "video_path": str(video_path),
            "mode": "multi_human",
            "EF": float(np.mean(entity_scores_with_missing)) if entity_scores_with_missing else 0.0,
            "EF_min": float(np.min(entity_scores_with_missing)) if entity_scores_with_missing else 0.0,
            "EF_visible_weighted": float(np.sum(visible_weights * visible_scores) / np.sum(visible_weights))
            if len(valid_entity_scores)
            else 0.0,
            "identity_switch_count": identity_switch_count,
            "face_recognition_model": getattr(args, "face_recognition_name", "arcface"),
            "sampled_frames": len(frames),
            "entities": entity_results,
            "frame_scores": frame_rows,
        }
