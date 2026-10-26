from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn

from similarity import cosine, topk_average
from subject_detector import Box


@dataclass
class FaceEmbedding:
    key: str
    arcface: np.ndarray
    bbox: Box | None
    roles: dict[str, Any]


@dataclass
class FrameFace:
    arcface: np.ndarray
    bbox: Box
    det_score: float


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResNet(nn.Module):
    def __init__(self, layers: list[int]):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.features = nn.BatchNorm1d(512, eps=1e-05)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes * IBasicBlock.expansion, stride),
            nn.BatchNorm2d(planes * IBasicBlock.expansion, eps=1e-05),
        )
        layers = [IBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * IBasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(IBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)
        return x


class TorchFaceRecognizer:
    def __init__(self, checkpoint: Path, device: str):
        self.device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.model = IResNet([3, 13, 30, 3])
        state = torch.load(str(checkpoint), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def get(self, image_rgb: np.ndarray, face) -> np.ndarray:
        from insightface.utils import face_align

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        aligned_bgr = face_align.norm_crop(image_bgr, landmark=face.kps, image_size=112)
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(aligned_rgb.transpose(2, 0, 1)).float()
        tensor = (tensor - 127.5) / 128.0
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor).squeeze(0).detach().cpu().numpy()
        return feat.astype(np.float32)


class FaceBranch:
    def __init__(
        self,
        face_model_root: str | Path,
        curricular_checkpoint: str | Path | None = None,
        recognition_model: str | Path | None = None,
        recognition_name: str = "arcface",
        device: str = "cuda",
    ) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError("insightface is required for Face branch.") from exc

        self.device = device
        self.recognition_name = recognition_name
        self.face_model = FaceAnalysis(
            root=str(face_model_root),
            providers=["CUDAExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"],
        )
        ctx_id = 0 if device.startswith("cuda") else -1
        self.face_model.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.recognition_model = None
        if recognition_model:
            recognition_model = Path(recognition_model)
            if not recognition_model.exists():
                raise FileNotFoundError(f"Face recognition model not found: {recognition_model}")
            if recognition_model.suffix.lower() == ".onnx":
                from insightface.model_zoo import model_zoo

                self.recognition_model = model_zoo.get_model(
                    str(recognition_model),
                    providers=["CUDAExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"],
                )
                if self.recognition_model is None:
                    raise RuntimeError(f"Failed to load face recognition model: {recognition_model}")
                self.recognition_model.prepare(ctx_id)
            elif recognition_model.suffix.lower() in {".pth", ".pt"}:
                self.recognition_model = TorchFaceRecognizer(recognition_model, device)
            else:
                raise ValueError(f"Unsupported face recognition model format: {recognition_model}")

    def _detect_largest_face(self, image_rgb: np.ndarray):
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        faces = self.face_model.get(image_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def _detect_faces(self, image_rgb: np.ndarray):
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        faces = self.face_model.get(image_bgr)
        return sorted(
            faces,
            key=lambda f: (float(f.det_score), (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
            reverse=True,
        )

    def _embedding(self, image_rgb: np.ndarray, face) -> np.ndarray:
        if self.recognition_model is None:
            return np.asarray(face.embedding, dtype=np.float32)
        if isinstance(self.recognition_model, TorchFaceRecognizer):
            return self.recognition_model.get(image_rgb, face)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return np.asarray(self.recognition_model.get(image_bgr, face), dtype=np.float32)

    def embed_reference(self, key: str, image: Image.Image, roles: dict[str, Any]) -> FaceEmbedding | None:
        image_rgb = np.asarray(image.convert("RGB"))
        face = self._detect_largest_face(image_rgb)
        if face is None:
            return None
        bbox = Box(*[float(v) for v in face.bbox], score=float(face.det_score), label="face")
        return FaceEmbedding(
            key=key,
            arcface=self._embedding(image_rgb, face),
            bbox=bbox,
            roles=roles,
        )

    def embed_frame(self, image_rgb: np.ndarray) -> FrameFace | None:
        face = self._detect_largest_face(image_rgb)
        if face is None:
            return None
        bbox = Box(*[float(v) for v in face.bbox], score=float(face.det_score), label="face")
        return FrameFace(
            arcface=self._embedding(image_rgb, face),
            bbox=bbox,
            det_score=float(face.det_score),
        )

    def embed_frame_faces(self, image_rgb: np.ndarray) -> list[FrameFace]:
        frame_faces: list[FrameFace] = []
        for face in self._detect_faces(image_rgb):
            bbox = Box(*[float(v) for v in face.bbox], score=float(face.det_score), label="face")
            frame_faces.append(
                FrameFace(
                    arcface=self._embedding(image_rgb, face),
                    bbox=bbox,
                    det_score=float(face.det_score),
                )
            )
        return frame_faces


def face_similarity(frame_face: FrameFace, ref_face: FaceEmbedding) -> float:
    return max(0.0, cosine(frame_face.arcface, ref_face.arcface))


def face_component_similarities(frame_face: FrameFace, ref_face: FaceEmbedding) -> tuple[float, float | None]:
    arc = max(0.0, cosine(frame_face.arcface, ref_face.arcface))
    return arc, None


def face_pack_top2(frame_face: FrameFace, ref_faces: list[FaceEmbedding], k: int = 1) -> tuple[float, dict[str, float]]:
    sims = {ref.key: face_similarity(frame_face, ref) for ref in ref_faces}
    return topk_average(list(sims.values()), k=k), sims


def face_pack_component_top2(frame_face: FrameFace, ref_faces: list[FaceEmbedding], k: int = 1) -> dict[str, object]:
    arc_sims: dict[str, float] = {}
    curricular_sims: dict[str, float] = {}
    for ref in ref_faces:
        arc, cur = face_component_similarities(frame_face, ref)
        arc_sims[ref.key] = arc
        if cur is not None:
            curricular_sims[ref.key] = cur
    return {
        "arcface_score": topk_average(list(arc_sims.values()), k=k) if arc_sims else None,
        "curricular_score": topk_average(list(curricular_sims.values()), k=k) if curricular_sims else None,
        "arcface_sims": arc_sims,
        "curricular_sims": curricular_sims,
    }
