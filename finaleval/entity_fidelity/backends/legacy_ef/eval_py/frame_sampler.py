from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SampledFrame:
    frame_index: int
    image_rgb: np.ndarray


def sample_video_frames(video_path: str | Path, num_frames: int = 32) -> list[SampledFrame]:
    """Uniformly sample RGB frames from a video with decord.

    This follows OpenS2V-Nexus/eval/get_facesim.py: linspace over the whole video.
    """
    try:
        import decord
    except ImportError as exc:
        raise RuntimeError("decord is required for video frame sampling.") from exc

    video_path = str(video_path)
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    if total_frames == 0:
        return []

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    return [
        SampledFrame(frame_index=int(idx), image_rgb=vr[int(idx)].asnumpy())
        for idx in frame_indices
    ]

