from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from similarity import cosine, topk_average


class VisionEmbedder:
    """Generic image embedding wrapper for DINOv2 / SigLIP style HF models."""

    def __init__(self, model_path: str | Path, device: str = "cuda") -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("transformers is required for Body branch image embeddings.") from exc

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if hasattr(self.model, "get_image_features"):
            emb = self.model.get_image_features(**inputs)
        else:
            outputs = self.model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            else:
                emb = outputs.last_hidden_state.mean(dim=1)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.detach().cpu().numpy()[0]


@dataclass
class BodyEmbedding:
    key: str
    dinov2: np.ndarray | None
    siglip: np.ndarray | None


@dataclass
class BodyPackResult:
    overall_score: float
    overall_sims: dict[str, float]
    dinov2_score: float | None
    dinov2_sims: dict[str, float]
    siglip_score: float | None
    siglip_sims: dict[str, float]


class BodyBranch:
    def __init__(
        self,
        dinov2_model: str | Path | None,
        siglip_model: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        if not dinov2_model and not siglip_model:
            raise ValueError("At least one of --dinov2-model or --siglip-model is required.")
        self.dinov2 = VisionEmbedder(dinov2_model, device=device) if dinov2_model else None
        self.siglip = VisionEmbedder(siglip_model, device=device) if siglip_model else None

    def embed(self, key: str, image: Image.Image) -> BodyEmbedding:
        return BodyEmbedding(
            key=key,
            dinov2=self.dinov2.embed(image) if self.dinov2 else None,
            siglip=self.siglip.embed(image) if self.siglip else None,
        )


def body_similarity(frame_body: BodyEmbedding, ref_body: BodyEmbedding) -> float:
    if frame_body.siglip is not None and ref_body.siglip is not None:
        return max(0.0, cosine(frame_body.siglip, ref_body.siglip))
    if frame_body.dinov2 is not None and ref_body.dinov2 is not None:
        return max(0.0, cosine(frame_body.dinov2, ref_body.dinov2))
    return 0.0


def _component_sims(
    frame_body: BodyEmbedding,
    ref_bodies: list[BodyEmbedding],
    attr: str,
) -> dict[str, float]:
    frame_vec = getattr(frame_body, attr)
    if frame_vec is None:
        return {}

    sims: dict[str, float] = {}
    for ref in ref_bodies:
        ref_vec = getattr(ref, attr)
        if ref_vec is None:
            continue
        sims[ref.key] = max(0.0, cosine(frame_vec, ref_vec))
    return sims


def body_pack_top2(frame_body: BodyEmbedding, ref_bodies: list[BodyEmbedding], k: int = 1) -> BodyPackResult:
    overall_sims = {ref.key: body_similarity(frame_body, ref) for ref in ref_bodies}
    dinov2_sims = _component_sims(frame_body, ref_bodies, "dinov2")
    siglip_sims = _component_sims(frame_body, ref_bodies, "siglip")

    return BodyPackResult(
        overall_score=topk_average(list(overall_sims.values()), k=k),
        overall_sims=overall_sims,
        dinov2_score=topk_average(list(dinov2_sims.values()), k=k) if dinov2_sims else None,
        dinov2_sims=dinov2_sims,
        siglip_score=topk_average(list(siglip_sims.values()), k=k) if siglip_sims else None,
        siglip_sims=siglip_sims,
    )
