from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    label: str = "person"

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def clamp_box(box: Box, width: int, height: int) -> Box:
    return Box(
        x1=max(0.0, min(float(width - 1), box.x1)),
        y1=max(0.0, min(float(height - 1), box.y1)),
        x2=max(0.0, min(float(width), box.x2)),
        y2=max(0.0, min(float(height), box.y2)),
        score=box.score,
        label=box.label,
    )


def crop_rgb(image_rgb: np.ndarray, box: Box) -> Image.Image:
    h, w = image_rgb.shape[:2]
    box = clamp_box(box, w, h)
    x1, y1, x2, y2 = [int(round(v)) for v in (box.x1, box.y1, box.x2, box.y2)]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop box: {box}")
    return Image.fromarray(image_rgb[y1:y2, x1:x2]).convert("RGB")


def expand_face_to_body(face_box: Box, image_shape: tuple[int, int, int], scale_x: float = 3.2, scale_y: float = 5.8) -> Box:
    h, w = image_shape[:2]
    cx = (face_box.x1 + face_box.x2) / 2.0
    cy = (face_box.y1 + face_box.y2) / 2.0
    fw = max(1.0, face_box.x2 - face_box.x1)
    fh = max(1.0, face_box.y2 - face_box.y1)
    body_w = fw * scale_x
    body_h = fh * scale_y
    return clamp_box(
        Box(
            x1=cx - body_w / 2.0,
            y1=cy - fh * 0.8,
            x2=cx + body_w / 2.0,
            y2=cy - fh * 0.8 + body_h,
            score=face_box.score,
            label="person_from_face",
        ),
        w,
        h,
    )


class FaceExpandedBodyDetector:
    """Fallback detector for debugging when YOLO-World weights are unavailable."""

    def detect(self, image_rgb: np.ndarray, face_box: Box | None = None) -> list[Box]:
        if face_box is None:
            return []
        return [expand_face_to_body(face_box, image_rgb.shape)]


class YOLOWorldPersonDetector:
    """YOLO-World person detector adapter.

    The OpenS2V repo vendors YOLO-World source code but not necessarily weights. Pass a
    checkpoint path on the target machine. This adapter does not download anything.
    """

    def __init__(
        self,
        config: str | Path,
        checkpoint: str | Path,
        device: str = "cuda:0",
        repo_root: str | Path = "OpenS2V-Nexus/eval/utils/yoloworld",
        score_threshold: float = 0.15,
        texts: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        repo_root = Path(repo_root).resolve()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        try:
            import yolo_world  # noqa: F401
            from mmdet.apis import inference_detector, init_detector
            from mmengine.dataset import Compose
        except ImportError as exc:
            raise RuntimeError(
                "YOLO-World dependencies are not available. Follow OpenS2V-Nexus/eval/README.md "
                "to create the opens2v_yoloworld environment."
            ) from exc

        self._inference_detector = inference_detector
        self.texts = list(texts or ["person"])
        # Avoid building the demo config's COCO dataset just to infer a palette.
        self.model = init_detector(str(config), str(checkpoint), device=device, palette="random")
        if hasattr(self.model, "reparameterize"):
            self.model.reparameterize([[text] for text in self.texts])
        pipeline_cfg = copy.deepcopy(self.model.cfg.test_dataloader.dataset.pipeline)
        pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
        self.pipeline = Compose(pipeline_cfg)
        self.score_threshold = score_threshold

    def detect(self, image_rgb: np.ndarray, face_box: Box | None = None) -> list[Box]:
        data_info = {"img_id": 0, "img": image_rgb, "texts": [[text] for text in self.texts]}
        data_info = self.pipeline(data_info)
        data_batch = {
            "inputs": data_info["inputs"].unsqueeze(0),
            "data_samples": [data_info["data_samples"]],
        }

        with torch.no_grad():
            result = self.model.test_step(data_batch)[0]

        pred = getattr(result, "pred_instances", None)
        if pred is None:
            return []

        bboxes = pred.bboxes.detach().cpu().numpy()
        scores = pred.scores.detach().cpu().numpy()
        boxes: list[Box] = []
        labels = getattr(pred, "labels", None)
        label_values = labels.detach().cpu().numpy() if labels is not None else [0] * len(scores)
        for bbox, score, label_index in zip(bboxes, scores, label_values):
            score = float(score)
            if score < self.score_threshold:
                continue
            label = self.texts[int(label_index)] if int(label_index) < len(self.texts) else self.texts[0]
            boxes.append(Box(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), score, label))
        boxes.sort(key=lambda b: (b.score, b.area), reverse=True)
        return boxes
