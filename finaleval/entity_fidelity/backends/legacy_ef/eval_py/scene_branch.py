from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from body_branch import BodyBranch, body_pack_top2
from condition_pack import ConditionReference
from frame_sampler import sample_video_frames


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid_rows = [row for row in rows if row.get("EF_t") is not None]
    return max(valid_rows, key=lambda row: float(row["EF_t"])) if valid_rows else None


class SceneBranch:
    def __init__(self, body_branch: BodyBranch, refs: list[ConditionReference]) -> None:
        self.body_branch = body_branch
        self.refs = refs
        self.ref_embeddings = [
            body_branch.embed(ref.key, Image.open(ref.path).convert("RGB"))
            for ref in refs
        ]

    def evaluate_video(self, video_path: Path, num_frames: int) -> dict[str, Any]:
        frames = sample_video_frames(video_path, num_frames=num_frames)
        rows = []
        for sampled in frames:
            frame_image = Image.fromarray(sampled.image_rgb).convert("RGB")
            frame_emb = self.body_branch.embed("scene_frame", frame_image)
            result = body_pack_top2(frame_emb, self.ref_embeddings)
            scene_score = result.siglip_score if result.siglip_score is not None else result.overall_score
            rows.append(
                {
                    "frame_index": sampled.frame_index,
                    "EF_t": scene_score,
                    "Scene_EF_t": scene_score,
                    "Scene_DINOv2_t": result.dinov2_score,
                    "Scene_SigLIP_t": result.siglip_score,
                    "valid": True,
                    "similarities": result.siglip_sims or result.overall_sims,
                    "diagnostic_overall_score": result.overall_score,
                }
            )

        values = [row["EF_t"] for row in rows]
        best_row = _best_row(rows)
        return {
            "Scene_EF": float(best_row["EF_t"]) if best_row else 0.0,
            "Scene_DINOv2": best_row.get("Scene_DINOv2_t") if best_row else None,
            "Scene_SigLIP": best_row.get("Scene_SigLIP_t") if best_row else None,
            "Scene_EF_mean": _mean(values),
            "best_frame_index": best_row.get("frame_index") if best_row else None,
            "aggregation": "best_frame",
            "valid_scene_ratio": 1.0 if frames else 0.0,
            "scene_references": [
                {
                    "key": ref.key,
                    "path": str(ref.path),
                    "description": ref.description,
                    "style": ref.style,
                }
                for ref in self.refs
            ],
            "frame_scores": rows,
        }
