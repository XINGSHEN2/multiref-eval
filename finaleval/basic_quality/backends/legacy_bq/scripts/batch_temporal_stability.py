#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import mean

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def init_worker() -> None:
    cv2.setNumThreads(1)


def iter_videos(video_dir: Path) -> list[Path]:
    return sorted(
        p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def sample_frame_indices(total_frames: int, num_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames <= num_frames:
        return list(range(total_frames))
    step = (total_frames - 1) / float(num_frames - 1)
    return sorted({min(total_frames - 1, round(i * step)) for i in range(num_frames)})


def sample_frame_indices_by_fps(
    total_frames: int,
    video_fps: float,
    sample_fps: float,
    max_seconds: float | None = None,
) -> list[int]:
    if total_frames <= 0 or video_fps <= 0.0 or sample_fps <= 0.0:
        return []
    duration = total_frames / video_fps
    if max_seconds and max_seconds > 0.0:
        duration = min(duration, max_seconds)
    sample_count = max(2, int(np.floor(duration * sample_fps)) + 1)
    indices = []
    for sample_index in range(sample_count):
        timestamp = sample_index / sample_fps
        frame_index = round(timestamp * video_fps)
        indices.append(min(total_frames - 1, frame_index))
    return sorted(set(indices))


def load_frames(
    video_path: Path,
    num_frames: int,
    size: int,
    sample_fps: float | None = None,
    max_seconds: float | None = None,
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    if sample_fps and sample_fps > 0.0:
        indices = sample_frame_indices_by_fps(total_frames, video_fps, sample_fps, max_seconds)
    else:
        indices = sample_frame_indices(total_frames, num_frames)
    target_indices = set(indices)
    max_index = indices[-1] if indices else -1
    frame_index = 0
    while frame_index <= max_index:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index in target_indices:
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        frame_index += 1
    cap.release()
    return frames


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 1.0
    return float(np.dot(a, b) / denom)


def score_video(
    video_path: Path,
    num_frames: int,
    size: int,
    sample_fps: float | None = None,
    skip_flow: bool = False,
    max_seconds: float | None = None,
) -> dict:
    frames = load_frames(video_path, num_frames, size, sample_fps, max_seconds)
    if len(frames) < 2:
        return {
            "file": str(video_path),
            "filename": video_path.name,
            "error": "fewer than two readable frames",
        }

    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

    luma_diffs = []
    hist_diffs = []
    frame_similarities = []
    flow_magnitudes = []
    for prev, curr in zip(gray_frames, gray_frames[1:]):
        luma_diffs.append(float(np.mean(np.abs(curr.astype(np.float32) - prev.astype(np.float32))) / 255.0))

        prev_hist = cv2.calcHist([prev], [0], None, [64], [0, 256])
        curr_hist = cv2.calcHist([curr], [0], None, [64], [0, 256])
        cv2.normalize(prev_hist, prev_hist)
        cv2.normalize(curr_hist, curr_hist)
        hist_diffs.append(float(cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)))

        frame_similarities.append(cosine_similarity(prev, curr))

        if not skip_flow:
            flow = cv2.calcOpticalFlowFarneback(
                prev,
                curr,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitudes.append(float(np.mean(magnitude)))

    flow_jitter = None
    if len(flow_magnitudes) > 1:
        flow_jitter = float(np.mean(np.abs(np.diff(flow_magnitudes))))

    return {
        "file": str(video_path),
        "filename": video_path.name,
        "num_frames": len(frames),
        "sample_fps": sample_fps,
        "max_seconds": max_seconds,
        "temporal_stability": float(mean(frame_similarities)),
        "frame_similarity_mean": float(mean(frame_similarities)),
        "luma_flicker_mean": float(mean(luma_diffs)),
        "histogram_diff_mean": float(mean(hist_diffs)),
        "flow_magnitude_mean": float(mean(flow_magnitudes)) if flow_magnitudes else None,
        "flow_jitter": flow_jitter,
        "error": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch temporal stability scoring")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--sample_fps", type=float, default=0.0, help="Sample frames at this real-time FPS instead of fixed num_frames.")
    parser.add_argument("--max_seconds", type=float, default=0.0, help="Only score the first N seconds when positive.")
    parser.add_argument("--skip_flow", action="store_true", help="Skip optical-flow diagnostics; temporal_stability is unchanged.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    video_dir = Path(args.video_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    videos = iter_videos(video_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found in {video_dir}")

    sample_fps = args.sample_fps if args.sample_fps > 0.0 else None
    max_seconds = args.max_seconds if args.max_seconds > 0.0 else None
    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as executor:
            futures = [
                executor.submit(
                    score_video,
                    video,
                    args.num_frames,
                    args.size,
                    sample_fps,
                    args.skip_flow,
                    max_seconds,
                )
                for video in videos
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if index == 1 or index % 50 == 0 or index == len(videos):
                    print(f"Processed {index}/{len(videos)} videos", flush=True)
    else:
        init_worker()
        for index, video in enumerate(videos, start=1):
            results.append(score_video(video, args.num_frames, args.size, sample_fps, args.skip_flow, max_seconds))
            if index == 1 or index % 50 == 0 or index == len(videos):
                print(f"Processed {index}/{len(videos)} videos", flush=True)
    valid = [item for item in results if item.get("error") is None]
    flow_jitters = [item["flow_jitter"] for item in valid if item.get("flow_jitter") is not None]

    payload = {
        "metric": "temporal_stability",
        "summary": {
            "mean_temporal_stability": float(mean([item["temporal_stability"] for item in valid])) if valid else 0.0,
            "mean_luma_flicker": float(mean([item["luma_flicker_mean"] for item in valid])) if valid else 0.0,
            "mean_flow_jitter": float(mean(flow_jitters)) if flow_jitters else None,
            "total_samples": len(results),
            "successful_samples": len(valid),
        },
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
