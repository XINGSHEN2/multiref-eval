from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from subject_detector import Box, clamp_box


@dataclass
class CartoonFaceDetection:
    box: Box
    source: str


class CartoonFaceDetector:
    def detect(self, image_rgb: np.ndarray) -> list[CartoonFaceDetection]:
        raise NotImplementedError


class HystsAnimeFaceDetector(CartoonFaceDetector):
    def __init__(
        self,
        config: str | Path,
        checkpoint: str | Path,
        device: str,
        score_threshold: float,
    ) -> None:
        try:
            from mmdet.apis import inference_detector, init_detector
        except ImportError as exc:
            raise RuntimeError("mmdet is required for hysts anime-face-detector.") from exc
        self.inference_detector = inference_detector
        self.model = init_detector(str(config), str(checkpoint), device=device)
        self.score_threshold = score_threshold

    def detect(self, image_rgb: np.ndarray) -> list[CartoonFaceDetection]:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        result = self.inference_detector(self.model, image_bgr)
        boxes: list[CartoonFaceDetection] = []
        if hasattr(result, "pred_instances"):
            instances = result.pred_instances
            bboxes = instances.bboxes.detach().cpu().numpy()
            scores = instances.scores.detach().cpu().numpy()
            for bbox, score in zip(bboxes, scores):
                if float(score) >= self.score_threshold:
                    boxes.append(CartoonFaceDetection(Box(*[float(v) for v in bbox], score=float(score), label="cartoon_face"), "hysts"))
        else:
            arrays = result if isinstance(result, (list, tuple)) else [result]
            for array in arrays:
                if array is None:
                    continue
                arr = np.asarray(array)
                if arr.ndim != 2 or arr.shape[1] < 5:
                    continue
                for row in arr:
                    score = float(row[4])
                    if score >= self.score_threshold:
                        boxes.append(CartoonFaceDetection(Box(*[float(v) for v in row[:4]], score=score, label="cartoon_face"), "hysts"))
        h, w = image_rgb.shape[:2]
        return sorted(
            [CartoonFaceDetection(clamp_box(item.box, w, h), item.source) for item in boxes],
            key=lambda item: item.box.score,
            reverse=True,
        )


class YOLOv8AnimeFaceDetector(CartoonFaceDetector):
    def __init__(self, weights: str | Path, device: str, score_threshold: float) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for YOLOv8 anime face detector.") from exc
        self.model = YOLO(str(weights))
        self.device = device
        self.score_threshold = score_threshold

    def detect(self, image_rgb: np.ndarray) -> list[CartoonFaceDetection]:
        results = self.model.predict(image_rgb, conf=self.score_threshold, device=self.device, verbose=False)
        detections: list[CartoonFaceDetection] = []
        for result in results:
            if result.boxes is None:
                continue
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            conf = result.boxes.conf.detach().cpu().numpy()
            for bbox, score in zip(xyxy, conf):
                detections.append(CartoonFaceDetection(Box(*[float(v) for v in bbox], score=float(score), label="cartoon_face"), "yolov8"))
        h, w = image_rgb.shape[:2]
        return sorted(
            [CartoonFaceDetection(clamp_box(item.box, w, h), item.source) for item in detections],
            key=lambda item: item.box.score,
            reverse=True,
        )


def build_cartoon_face_detector(args: Any) -> CartoonFaceDetector | None:
    detector_name = getattr(args, "cartoon_face_detector", "hysts-yolov3")
    if detector_name in (None, "none"):
        return None
    score_threshold = float(getattr(args, "cartoon_face_score_threshold", 0.15))
    if detector_name == "hysts-yolov3":
        return HystsAnimeFaceDetector(
            getattr(args, "hysts_yolov3_config", "ef_eval/model_assets/anime_face_detectors/mmdet3/mmdet_anime-face_yolov3.py"),
            getattr(args, "hysts_yolov3_checkpoint", "ef_eval/model_assets/anime_face_detectors/mmdet3/mmdet_anime-face_yolov3.pth"),
            getattr(args, "detector_device", "cuda:0"),
            score_threshold,
        )
    if detector_name == "yolov8-anime-face":
        return YOLOv8AnimeFaceDetector(
            getattr(args, "yolov8_anime_face_checkpoint", "ef_eval/model_assets/anime_face_detectors/yolov8/yolov8x6_animeface.pt"),
            getattr(args, "yolov8_device", "0"),
            score_threshold,
        )
    raise ValueError(f"Unsupported cartoon face detector: {detector_name}")


def contains_center(outer: Box, inner: Box) -> bool:
    cx = (inner.x1 + inner.x2) / 2.0
    cy = (inner.y1 + inner.y2) / 2.0
    return outer.x1 <= cx <= outer.x2 and outer.y1 <= cy <= outer.y2


def best_face_for_box(detections: list[CartoonFaceDetection], body_box: Box | None = None) -> CartoonFaceDetection | None:
    candidates = detections
    if body_box is not None:
        candidates = [item for item in detections if contains_center(body_box, item.box)]
    return max(candidates, key=lambda item: item.box.score) if candidates else None
