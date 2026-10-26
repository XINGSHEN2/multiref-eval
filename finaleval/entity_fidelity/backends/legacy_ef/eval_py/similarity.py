from __future__ import annotations

import numpy as np


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def topk_average(values: list[float], k: int = 2) -> float:
    if not values:
        return 0.0
    values = sorted(values, reverse=True)
    return float(np.mean(values[: min(k, len(values))]))

