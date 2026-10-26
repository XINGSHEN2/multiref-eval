#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import cv2
import numpy as np


FPS = 25
AUDIO_SAMPLE_RATE = 16000
VAD_MODE = 2
VAD_FRAME_MS = 30
MIN_SPEECH_SEC = 0.30
MERGE_SPEECH_GAP_SEC = 0.20
ENERGY_VAD_MIN_RMS = 0.003
VAD_BACKEND = "silero"
SILERO_THRESHOLD = 0.5
WINDOW_SEC = 2.0
MIN_FACE_CONFIDENCE = 0.90
MIN_FACE_SIZE_PX = 64
TRACK_IOU_THRESHOLD = 0.30
MAX_TRACK_GAP_FRAMES = 5
MIN_TRACK_FRAMES = 20
MIN_TRACK_COVERAGE = 0.80
FACE_CROP_SCALE = 1.30
SYNCNET_VSHIFT = 15
MIN_SYNC_CONFIDENCE = 1.50
MAX_RELIABLE_ABS_OFFSET = 5
MAX_SELECTION_ABS_OFFSET = 10
MAX_AV_DURATION_DIFF_SEC = 0.08
METRIC_NAME = "Visible Speech-Lip Synchronization"
METRIC_VERSION = "vsls_v3_combined_lse_track_select"

DEFAULT_LATENTSYNC_DIR = Path(os.getenv("LATENTSYNC_DIR", "LatentSync"))
DEFAULT_MODEL_PATH = DEFAULT_LATENTSYNC_DIR / "checkpoints/auxiliary/syncnet_v2.model"
_SILERO_MODEL: Any | None = None


def progress(message: str) -> None:
    print(f"[vsls] {message}", flush=True)


@dataclass(frozen=True)
class TimeRange:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class FaceDetection:
    frame_index: int
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass
class FaceTrack:
    track_id: int
    detections: dict[int, FaceDetection] = field(default_factory=dict)
    last_frame: int = -1
    missed_frames: int = 0

    @property
    def start_frame(self) -> int:
        return min(self.detections)

    @property
    def end_frame(self) -> int:
        return max(self.detections)

    @property
    def detection_count(self) -> int:
        return len(self.detections)


@dataclass(frozen=True)
class TrackWindow:
    track_id: int
    start_frame: int
    end_frame: int
    coverage: float
    mean_face_size: float
    bboxes: dict[int, tuple[float, float, float, float]]


@dataclass(frozen=True)
class SyncNetResult:
    track_id: int
    crop_path: str
    confidence: float
    offset_frames: float
    offset_seconds: float
    min_distance: float
    reliable: bool
    score_0_1: float | None
    mos_1_5: float | None
    lse_c_norm: float
    lse_d_norm: float
    combined_lse_score_0_1: float
    combined_lse_score_1_5: float


def run_command(command: list[str], description: str) -> None:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"{description} failed: {stderr[-2000:]}")


def resolve_ffmpeg(explicit_path: str | None) -> str:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ffmpeg not found: {path}")
        return str(path)
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("ffmpeg is unavailable; install ffmpeg or imageio-ffmpeg") from exc


def expose_ffmpeg_on_path(ffmpeg_bin: str, work_dir: Path) -> Path:
    bin_dir = work_dir / ".bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    link_path = bin_dir / "ffmpeg"
    target = Path(ffmpeg_bin).resolve()
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target)
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    return link_path


def media_duration(path: Path, ffmpeg_bin: str) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())

    result = subprocess.run(
        [ffmpeg_bin, "-i", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    for line in result.stderr.splitlines():
        if "Duration:" not in line:
            continue
        value = line.split("Duration:", 1)[1].split(",", 1)[0].strip()
        hours, minutes, seconds = value.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise RuntimeError(f"Unable to determine media duration: {path}")


def standardize_video(video_path: Path, work_dir: Path, ffmpeg_bin: str) -> tuple[Path, Path, float]:
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")

    standard_dir = work_dir / "standard"
    standard_dir.mkdir(parents=True, exist_ok=True)
    standard_video = standard_dir / "video_25fps.mp4"
    audio_wav = standard_dir / "audio_16k.wav"

    run_command(
        [
            ffmpeg_bin,
            "-y",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"fps={FPS}",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(standard_video),
        ],
        "video standardization",
    )
    run_command(
        [
            ffmpeg_bin,
            "-y",
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(AUDIO_SAMPLE_RATE),
            "-sample_fmt",
            "s16",
            "-c:a",
            "pcm_s16le",
            str(audio_wav),
        ],
        "audio standardization",
    )

    video_duration = media_duration(standard_video, ffmpeg_bin)
    audio_duration = media_duration(audio_wav, ffmpeg_bin)
    if audio_duration - video_duration > MAX_AV_DURATION_DIFF_SEC:
        progress(
            f"standardized full audio track is longer than video; trim full audio track "
            f"video={video_duration:.3f}s audio={audio_duration:.3f}s"
        )
        trimmed_audio = standard_dir / "audio_16k_trimmed.wav"
        run_command(
            [
                ffmpeg_bin,
                "-y",
                "-i",
                str(audio_wav),
                "-t",
                f"{video_duration:.6f}",
                "-ac",
                "1",
                "-ar",
                str(AUDIO_SAMPLE_RATE),
                "-sample_fmt",
                "s16",
                "-c:a",
                "pcm_s16le",
                str(trimmed_audio),
            ],
            "audio duration trim",
        )
        trimmed_audio.replace(audio_wav)
        audio_duration = media_duration(audio_wav, ffmpeg_bin)
        progress(f"trimmed full audio track duration={audio_duration:.3f}s")
    elif video_duration - audio_duration > MAX_AV_DURATION_DIFF_SEC:
        progress(
            f"standardized video is longer than audio; using shared duration "
            f"video={video_duration:.3f}s audio={audio_duration:.3f}s"
        )
    return standard_video, audio_wav, min(video_duration, audio_duration)


def merge_time_ranges(ranges: Iterable[TimeRange], max_gap: float) -> list[TimeRange]:
    sorted_ranges = sorted(ranges, key=lambda item: item.start)
    if not sorted_ranges:
        return []
    merged = [sorted_ranges[0]]
    for current in sorted_ranges[1:]:
        previous = merged[-1]
        if current.start - previous.end <= max_gap:
            merged[-1] = TimeRange(previous.start, max(previous.end, current.end))
        else:
            merged.append(current)
    return merged


def detect_speech_segments(
    audio_wav: Path,
    backend: str = VAD_BACKEND,
    vad_mode: int = VAD_MODE,
    frame_ms: int = VAD_FRAME_MS,
    min_speech_duration: float = MIN_SPEECH_SEC,
    merge_gap: float = MERGE_SPEECH_GAP_SEC,
) -> list[TimeRange]:
    with wave.open(str(audio_wav), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        audio = handle.readframes(handle.getnframes())

    if channels != 1 or sample_width != 2 or sample_rate != AUDIO_SAMPLE_RATE:
        raise ValueError("VAD requires 16 kHz mono PCM16 WAV")

    samples_per_frame = sample_rate * frame_ms // 1000
    bytes_per_frame = samples_per_frame * sample_width
    backend = backend.lower()
    if backend == "webrtcvad":
        import webrtcvad

        vad = webrtcvad.Vad(vad_mode)
        flags = []
        for offset in range(0, len(audio) - bytes_per_frame + 1, bytes_per_frame):
            frame = audio[offset : offset + bytes_per_frame]
            flags.append(vad.is_speech(frame, sample_rate))
    elif backend == "silero":
        return detect_speech_segments_by_silero(
            audio_wav,
            min_speech_duration=min_speech_duration,
            merge_gap=merge_gap,
        )
    elif backend == "energy":
        progress("using energy VAD; this detects active audio, not speech specifically")
        flags = detect_speech_flags_by_energy(audio, sample_rate, samples_per_frame)
    else:
        raise ValueError(f"unsupported VAD backend: {backend}; use silero, webrtcvad, or energy")

    raw_segments: list[TimeRange] = []
    start_index: int | None = None
    for index, is_speech in enumerate(flags + [False]):
        if is_speech and start_index is None:
            start_index = index
        elif not is_speech and start_index is not None:
            start = start_index * frame_ms / 1000.0
            end = index * frame_ms / 1000.0
            if end - start >= min_speech_duration:
                raw_segments.append(TimeRange(start, end))
            start_index = None
    return merge_time_ranges(raw_segments, merge_gap)


def load_cached_silero_model() -> Any:
    global _SILERO_MODEL
    if _SILERO_MODEL is None:
        progress("load Silero VAD model")
        from silero_vad import load_silero_vad

        _SILERO_MODEL = load_silero_vad()
    return _SILERO_MODEL


def detect_speech_segments_by_silero(
    audio_wav: Path,
    min_speech_duration: float = MIN_SPEECH_SEC,
    merge_gap: float = MERGE_SPEECH_GAP_SEC,
) -> list[TimeRange]:
    from silero_vad import get_speech_timestamps, read_audio

    model = load_cached_silero_model()
    audio = read_audio(str(audio_wav), sampling_rate=AUDIO_SAMPLE_RATE)
    timestamps = get_speech_timestamps(
        audio,
        model,
        threshold=SILERO_THRESHOLD,
        sampling_rate=AUDIO_SAMPLE_RATE,
        min_speech_duration_ms=int(min_speech_duration * 1000),
        min_silence_duration_ms=int(merge_gap * 1000),
        speech_pad_ms=30,
        return_seconds=True,
    )
    segments = [
        TimeRange(float(item["start"]), float(item["end"]))
        for item in timestamps
        if float(item["end"]) - float(item["start"]) >= min_speech_duration
    ]
    preview = ", ".join(f"{segment.start:.2f}-{segment.end:.2f}s" for segment in segments[:8])
    if len(segments) > 8:
        preview += f", ... +{len(segments) - 8} more"
    progress(f"Silero VAD speech segments={len(segments)} [{preview}]")
    return merge_time_ranges(segments, merge_gap)


def detect_speech_flags_by_energy(
    audio: bytes,
    sample_rate: int,
    samples_per_frame: int,
) -> list[bool]:
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    frame_count = len(samples) // samples_per_frame
    if frame_count <= 0:
        return []

    rms_values = []
    for frame_index in range(frame_count):
        start = frame_index * samples_per_frame
        frame = samples[start : start + samples_per_frame]
        rms_values.append(float(np.sqrt(np.mean(frame * frame))))
    rms = np.asarray(rms_values, dtype=np.float32)
    if rms.size == 0 or float(np.max(rms)) < 1e-5:
        return [False] * frame_count

    noise_floor = float(np.percentile(rms, 20))
    high_energy = float(np.percentile(rms, 90))
    threshold = max(ENERGY_VAD_MIN_RMS, noise_floor * 2.5, high_energy * 0.15)
    progress(
        "energy VAD threshold="
        f"{threshold:.5f} noise_floor={noise_floor:.5f} high_energy={high_energy:.5f}"
    )
    return [value >= threshold for value in rms_values]


def split_speech_windows(
    segments: list[TimeRange],
    window_sec: float = WINDOW_SEC,
) -> list[TimeRange]:
    windows: list[TimeRange] = []
    for segment in segments:
        cursor = segment.start
        while cursor < segment.end:
            end = min(cursor + window_sec, segment.end)
            windows.append(TimeRange(cursor, end))
            cursor = end
    return windows


def bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def detect_faces(
    video_path: Path,
    detector: Any,
    min_confidence: float = MIN_FACE_CONFIDENCE,
    progress_label: str | None = None,
    progress_every: int = 100,
) -> tuple[dict[int, list[FaceDetection]], int, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open standardized video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    faces_by_frame: dict[int, list[FaceDetection]] = {}
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected = detector.detect_faces(rgb, conf_th=min_confidence, scales=[0.25])
        faces_by_frame[frame_index] = [
            FaceDetection(
                frame_index=frame_index,
                bbox=tuple(float(value) for value in bbox[:4]),
                confidence=float(bbox[4]),
            )
            for bbox in detected
            if float(bbox[4]) >= min_confidence
        ]
        frame_index += 1
        if progress_label and progress_every > 0 and frame_index % progress_every == 0:
            progress(f"{progress_label}: detected faces through frame {frame_index}")
    capture.release()
    return faces_by_frame, frame_index, width, height


def build_face_tracks(
    faces_by_frame: dict[int, list[FaceDetection]],
    iou_threshold: float = TRACK_IOU_THRESHOLD,
    max_gap_frames: int = MAX_TRACK_GAP_FRAMES,
) -> list[FaceTrack]:
    active: list[FaceTrack] = []
    completed: list[FaceTrack] = []
    next_track_id = 0

    for frame_index in sorted(faces_by_frame):
        detections = faces_by_frame[frame_index]
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(active):
            last_detection = track.detections[track.last_frame]
            for detection_index, detection in enumerate(detections):
                iou = bbox_iou(last_detection.bbox, detection.bbox)
                if iou >= iou_threshold:
                    candidates.append((iou, track_index, detection_index))

        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        for _, track_index, detection_index in sorted(candidates, reverse=True):
            if track_index in assigned_tracks or detection_index in assigned_detections:
                continue
            track = active[track_index]
            detection = detections[detection_index]
            track.detections[frame_index] = detection
            track.last_frame = frame_index
            track.missed_frames = 0
            assigned_tracks.add(track_index)
            assigned_detections.add(detection_index)

        still_active: list[FaceTrack] = []
        for track_index, track in enumerate(active):
            if track_index not in assigned_tracks:
                track.missed_frames += 1
            if track.missed_frames > max_gap_frames:
                completed.append(track)
            else:
                still_active.append(track)
        active = still_active

        for detection_index, detection in enumerate(detections):
            if detection_index in assigned_detections:
                continue
            active.append(
                FaceTrack(
                    track_id=next_track_id,
                    detections={frame_index: detection},
                    last_frame=frame_index,
                )
            )
            next_track_id += 1

    completed.extend(active)
    return completed


def interpolate_bbox(
    detections: dict[int, FaceDetection],
    frame_index: int,
) -> tuple[float, float, float, float] | None:
    if frame_index in detections:
        return detections[frame_index].bbox
    previous = [index for index in detections if index < frame_index]
    following = [index for index in detections if index > frame_index]
    if not previous or not following:
        return None
    left = max(previous)
    right = min(following)
    ratio = (frame_index - left) / (right - left)
    left_bbox = np.asarray(detections[left].bbox, dtype=np.float64)
    right_bbox = np.asarray(detections[right].bbox, dtype=np.float64)
    return tuple((left_bbox + ratio * (right_bbox - left_bbox)).tolist())


def smooth_bboxes(
    bboxes: dict[int, tuple[float, float, float, float]],
    radius: int = 2,
) -> dict[int, tuple[float, float, float, float]]:
    frame_indices = sorted(bboxes)
    values = np.asarray([bboxes[index] for index in frame_indices], dtype=np.float64)
    smoothed = []
    for index in range(len(frame_indices)):
        start = max(0, index - radius)
        end = min(len(frame_indices), index + radius + 1)
        smoothed.append(np.median(values[start:end], axis=0))
    return {
        frame_index: tuple(float(value) for value in bbox)
        for frame_index, bbox in zip(frame_indices, smoothed)
    }


def get_candidate_tracks(
    tracks: list[FaceTrack],
    window: TimeRange,
    fps: int = FPS,
    min_track_frames: int = MIN_TRACK_FRAMES,
    min_coverage: float = MIN_TRACK_COVERAGE,
    min_face_size: float = MIN_FACE_SIZE_PX,
) -> tuple[list[FaceTrack], list[TrackWindow]]:
    start_frame = max(0, int(math.floor(window.start * fps)))
    end_frame = max(start_frame + 1, int(math.ceil(window.end * fps)))
    total_frames = end_frame - start_frame
    intersecting: list[FaceTrack] = []
    candidates: list[TrackWindow] = []

    for track in tracks:
        if track.end_frame < start_frame or track.start_frame >= end_frame:
            continue
        intersecting.append(track)
        bboxes: dict[int, tuple[float, float, float, float]] = {}
        for frame_index in range(start_frame, end_frame):
            bbox = interpolate_bbox(track.detections, frame_index)
            if bbox is not None:
                bboxes[frame_index] = bbox
        coverage = len(bboxes) / total_frames
        if not bboxes:
            continue
        face_sizes = [max(bbox[2] - bbox[0], bbox[3] - bbox[1]) for bbox in bboxes.values()]
        mean_face_size = float(mean(face_sizes))
        if len(bboxes) < min_track_frames or coverage < min_coverage or mean_face_size < min_face_size:
            continue
        candidates.append(
            TrackWindow(
                track_id=track.track_id,
                start_frame=start_frame,
                end_frame=end_frame,
                coverage=coverage,
                mean_face_size=mean_face_size,
                bboxes=smooth_bboxes(bboxes),
            )
        )
    return intersecting, candidates


def expanded_crop(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    scale: float = FACE_CROP_SCALE,
) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * scale
    crop_x1 = int(math.floor(center_x - side / 2.0))
    crop_y1 = int(math.floor(center_y - side * 0.45))
    crop_x2 = int(math.ceil(center_x + side / 2.0))
    crop_y2 = int(math.ceil(crop_y1 + side))

    pad_left = max(0, -crop_x1)
    pad_top = max(0, -crop_y1)
    pad_right = max(0, crop_x2 - width)
    pad_bottom = max(0, crop_y2 - height)
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        frame = cv2.copyMakeBorder(
            frame,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(110, 110, 110),
        )
        crop_x1 += pad_left
        crop_x2 += pad_left
        crop_y1 += pad_top
        crop_y2 += pad_top
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        raise RuntimeError("empty face crop")
    return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)


def crop_track_segment(
    video_path: Path,
    audio_wav: Path,
    track_window: TrackWindow,
    window: TimeRange,
    output_path: Path,
    ffmpeg_bin: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    silent_path = output_path.with_name(output_path.stem + "_silent.mp4")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video for cropping: {video_path}")
    writer = cv2.VideoWriter(
        str(silent_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (224, 224),
    )
    capture.set(cv2.CAP_PROP_POS_FRAMES, track_window.start_frame)
    for frame_index in range(track_window.start_frame, track_window.end_frame):
        ok, frame = capture.read()
        if not ok:
            break
        bbox = track_window.bboxes.get(frame_index)
        if bbox is None:
            raise RuntimeError(f"missing interpolated bbox for frame {frame_index}")
        writer.write(expanded_crop(frame, bbox))
    writer.release()
    capture.release()

    run_command(
        [
            ffmpeg_bin,
            "-y",
            "-i",
            str(silent_path),
            "-ss",
            f"{window.start:.6f}",
            "-to",
            f"{window.end:.6f}",
            "-i",
            str(audio_wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-ar",
            str(AUDIO_SAMPLE_RATE),
            "-ac",
            "1",
            "-shortest",
            str(output_path),
        ],
        "face crop audio mux",
    )
    silent_path.unlink(missing_ok=True)
    duration = media_duration(output_path, ffmpeg_bin)
    if abs(duration - window.duration) > MAX_AV_DURATION_DIFF_SEC:
        raise RuntimeError(
            f"crop duration mismatch: expected={window.duration:.3f}s, actual={duration:.3f}s"
        )
    return output_path


def load_models(latentsync_dir: Path, model_path: Path, device: str) -> tuple[Any, Any]:
    if not latentsync_dir.is_dir():
        raise FileNotFoundError(f"LatentSync directory not found: {latentsync_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"SyncNet model not found: {model_path}")
    sys.path.insert(0, str(latentsync_dir))
    os.chdir(latentsync_dir)

    import torch
    from eval.detectors import S3FD
    from eval.syncnet.syncnet_eval import SyncNetEval

    resolved_device = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    progress(f"resolved torch device={resolved_device} requested={device}")
    detector = S3FD(device=resolved_device)
    syncnet = SyncNetEval(device=resolved_device)
    syncnet.loadParameters(str(model_path))
    return detector, syncnet


def syncnet_to_score(confidence: float, offset_frames: float) -> float:
    conf_score = np.clip((confidence - MIN_SYNC_CONFIDENCE) / (7.0 - MIN_SYNC_CONFIDENCE), 0.0, 1.0)
    offset_score = 1.0 - min(abs(offset_frames), 5.0) / 5.0
    return float(np.clip(0.8 * conf_score + 0.2 * offset_score, 0.0, 1.0))


def combined_lse_score(lse_c: float, lse_d: float) -> tuple[float, float, float, float]:
    c_norm = float(np.clip((lse_c - 1.5) / (7.0 - 1.5), 0.0, 1.0))
    d_norm = float(np.clip((13.0 - lse_d) / (13.0 - 7.0), 0.0, 1.0))
    score_0_1 = 0.60 * c_norm + 0.40 * d_norm
    return c_norm, d_norm, score_0_1, 1.0 + 4.0 * score_0_1


def is_reliable_sync(confidence: float, offset_frames: float) -> bool:
    return confidence >= MIN_SYNC_CONFIDENCE and abs(offset_frames) <= MAX_RELIABLE_ABS_OFFSET


def select_best_sync_track(track_results: list[SyncNetResult]) -> SyncNetResult | None:
    selectable = [result for result in track_results if abs(result.offset_frames) < MAX_SELECTION_ABS_OFFSET]
    if not selectable:
        return None
    return max(
        selectable,
        key=lambda result: (
            result.combined_lse_score_0_1,
            result.confidence,
            -result.min_distance,
            -abs(result.offset_frames),
        ),
    )


def run_syncnet(
    syncnet: Any,
    crop_video_path: Path,
    temp_dir: Path,
    track_id: int,
) -> SyncNetResult:
    offset, min_distance, confidence = syncnet.evaluate(
        video_path=str(crop_video_path),
        temp_dir=str(temp_dir),
        vshift=SYNCNET_VSHIFT,
    )
    offset_frames = float(offset)
    confidence_value = float(confidence)
    reliable = is_reliable_sync(confidence_value, offset_frames)
    score = syncnet_to_score(confidence_value, offset_frames)
    c_norm, d_norm, combined_0_1, combined_1_5 = combined_lse_score(confidence_value, float(min_distance))
    return SyncNetResult(
        track_id=track_id,
        crop_path=str(crop_video_path),
        confidence=confidence_value,
        offset_frames=offset_frames,
        offset_seconds=offset_frames / FPS,
        min_distance=float(min_distance),
        reliable=reliable,
        score_0_1=score,
        mos_1_5=(1.0 + 4.0 * score) if score is not None else None,
        lse_c_norm=c_norm,
        lse_d_norm=d_norm,
        combined_lse_score_0_1=combined_0_1,
        combined_lse_score_1_5=combined_1_5,
    )


def weighted_mean(values: list[float], weights: list[float]) -> float | None:
    total_weight = sum(weights)
    if not values or total_weight <= 0:
        return None
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def evaluate_window(
    window_id: int,
    total_windows: int,
    window: TimeRange,
    tracks: list[FaceTrack],
    standard_video: Path,
    audio_wav: Path,
    crops_dir: Path,
    sync_temp_dir: Path,
    syncnet: Any,
    ffmpeg_bin: str,
    progress_label: str,
) -> dict[str, Any]:
    intersecting, candidates = get_candidate_tracks(tracks, window)
    progress(
        f"{progress_label}: window {window_id + 1}/{total_windows} "
        f"{window.start:.2f}-{window.end:.2f}s, "
        f"intersecting_tracks={len(intersecting)}, candidates={len(candidates)}"
    )
    base: dict[str, Any] = {
        "segment_id": window_id,
        "start": window.start,
        "end": window.end,
        "duration": window.duration,
        "num_intersecting_tracks": len(intersecting),
        "num_candidate_tracks": len(candidates),
        "num_reliable_tracks": 0,
        "num_selectable_tracks": 0,
        "best_track_id": None,
        "best_confidence": None,
        "best_offset_frames": None,
        "best_lse_d": None,
        "best_combined_lse_score_0_1": None,
        "best_combined_lse_score_1_5": None,
        "score_0_1": 0.0,
        "mos_1_5": 1.0,
        "mapping_version": METRIC_VERSION,
        "all_track_scores": [],
    }
    if not intersecting:
        base["status"] = "offscreen_voice"
        progress(f"{progress_label}: window {window_id + 1}/{total_windows} -> offscreen_voice")
        return base
    if not candidates:
        base["status"] = "insufficient_track"
        progress(f"{progress_label}: window {window_id + 1}/{total_windows} -> insufficient_track")
        return base

    track_results: list[SyncNetResult] = []
    errors: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        crop_path = crops_dir / f"segment_{window_id:04d}_track_{candidate.track_id:04d}.mp4"
        try:
            progress(
                f"{progress_label}: window {window_id + 1}/{total_windows}, "
                f"track {candidate_index}/{len(candidates)} id={candidate.track_id} crop"
            )
            crop_track_segment(
                standard_video,
                audio_wav,
                candidate,
                window,
                crop_path,
                ffmpeg_bin,
            )
            progress(
                f"{progress_label}: window {window_id + 1}/{total_windows}, "
                f"track {candidate_index}/{len(candidates)} id={candidate.track_id} syncnet"
            )
            result = run_syncnet(
                syncnet,
                crop_path,
                sync_temp_dir / f"segment_{window_id:04d}_track_{candidate.track_id:04d}",
                candidate.track_id,
            )
            track_results.append(result)
            progress(
                f"{progress_label}: window {window_id + 1}/{total_windows}, "
                f"track id={candidate.track_id} conf={result.confidence:.3f} "
                f"offset={result.offset_frames:.1f} reliable={result.reliable}"
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"track_id": candidate.track_id, "error": str(exc)})
            progress(
                f"{progress_label}: window {window_id + 1}/{total_windows}, "
                f"track id={candidate.track_id} error={exc}"
            )

    base["all_track_scores"] = [asdict(result) for result in track_results]
    if errors:
        base["track_errors"] = errors
    if not track_results:
        base["status"] = "syncnet_error"
        progress(f"{progress_label}: window {window_id + 1}/{total_windows} -> syncnet_error")
        return base

    base["num_reliable_tracks"] = sum(result.reliable for result in track_results)
    base["num_selectable_tracks"] = sum(abs(result.offset_frames) < MAX_SELECTION_ABS_OFFSET for result in track_results)
    if base["num_selectable_tracks"] == 0:
        base["status"] = "unreliable_sync"
        progress(
            f"{progress_label}: window {window_id + 1}/{total_windows} -> unreliable_sync "
            f"(no track with abs(offset)<{MAX_SELECTION_ABS_OFFSET})"
        )
        return base
    best = select_best_sync_track(track_results)
    if best is None:
        base["status"] = "unreliable_sync"
        progress(f"{progress_label}: window {window_id + 1}/{total_windows} -> unreliable_sync")
        return base
    base.update(
        {
            "status": "scored",
            "best_track_id": best.track_id,
            "best_confidence": best.confidence,
            "best_offset_frames": best.offset_frames,
            "best_lse_d": best.min_distance,
            "best_combined_lse_score_0_1": best.combined_lse_score_0_1,
            "best_combined_lse_score_1_5": best.combined_lse_score_1_5,
            "score_0_1": best.score_0_1,
            "mos_1_5": best.mos_1_5,
        }
    )
    progress(
        f"{progress_label}: window {window_id + 1}/{total_windows} -> scored "
        f"best_track={best.track_id} score={best.score_0_1:.4f} "
        f"combined_lse={best.combined_lse_score_0_1:.4f} "
        f"conf={best.confidence:.3f} lse_d={best.min_distance:.3f} "
        f"offset={best.offset_frames:.1f}"
    )
    return base


def evaluate_video(
    video_path: Path,
    latentsync_dir: Path,
    model_path: Path,
    work_dir: Path,
    output_json: Path,
    device: str,
    ffmpeg_bin: str,
    keep_crops: bool = True,
    keep_temp: bool = True,
    detector: Any | None = None,
    syncnet: Any | None = None,
    vad_backend: str = VAD_BACKEND,
) -> dict[str, Any]:
    progress(f"start video={video_path}")
    work_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    progress("prepare ffmpeg path")
    expose_ffmpeg_on_path(ffmpeg_bin, work_dir)
    if detector is None or syncnet is None:
        progress(f"load models device={device}")
        detector, syncnet = load_models(latentsync_dir, model_path, device)
    else:
        progress("reuse loaded models")
    progress("standardize video/audio")
    standard_video, audio_wav, duration = standardize_video(video_path, work_dir, ffmpeg_bin)
    progress(f"standardized duration={duration:.3f}s video={standard_video} audio={audio_wav}")
    progress(f"detect speech with VAD backend={vad_backend}")
    speech_segments = detect_speech_segments(audio_wav, backend=vad_backend)
    windows = split_speech_windows(speech_segments)
    speech_duration = sum(segment.duration for segment in speech_segments)
    progress(
        f"speech detected segments={len(speech_segments)} windows={len(windows)} "
        f"speech_duration={speech_duration:.3f}s"
    )

    thresholds = {
        "min_sync_confidence": MIN_SYNC_CONFIDENCE,
        "max_reliable_abs_offset_frames": MAX_RELIABLE_ABS_OFFSET,
        "max_selection_abs_offset_frames": MAX_SELECTION_ABS_OFFSET,
        "min_track_coverage": MIN_TRACK_COVERAGE,
        "min_track_frames": MIN_TRACK_FRAMES,
        "min_face_size_px": MIN_FACE_SIZE_PX,
    }
    if not speech_segments:
        payload = {
            "metric": METRIC_NAME,
            "metric_version": METRIC_VERSION,
            "video_path": str(video_path),
            "video_status": "no_speech",
            "fps": FPS,
            "audio_sample_rate": AUDIO_SAMPLE_RATE,
            "duration": duration,
            "thresholds": thresholds,
            "speech_segments": [],
            "segments": [],
            "video_score_0_1": None,
            "video_mos_1_5": None,
        }
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        progress(f"finish video_status=no_speech output={output_json}")
        if not keep_temp:
            progress("delete transient work directories")
            shutil.rmtree(work_dir / "standard", ignore_errors=True)
            shutil.rmtree(work_dir / "syncnet_temp", ignore_errors=True)
        return payload

    progress("detect faces")
    faces_by_frame, frame_count, width, height = detect_faces(
        standard_video,
        detector,
        progress_label="face detection",
    )
    detected_face_count = sum(len(faces) for faces in faces_by_frame.values())
    progress(
        f"faces done frames={frame_count} size={width}x{height} "
        f"detections={detected_face_count}"
    )
    progress("build face tracks")
    tracks = build_face_tracks(faces_by_frame)
    progress(f"tracks built count={len(tracks)}")
    crops_dir = work_dir / "crops"
    sync_temp_dir = work_dir / "syncnet_temp"
    progress(f"evaluate windows count={len(windows)}")
    segments = []
    for index, window in enumerate(windows):
        segments.append(
            evaluate_window(
                index,
                len(windows),
                window,
                tracks,
                standard_video,
                audio_wav,
                crops_dir,
                sync_temp_dir,
                syncnet,
                ffmpeg_bin,
                "window eval",
            )
        )

    scored_segments = [segment for segment in segments if segment["status"] == "scored"]
    score = weighted_mean(
        [float(segment["score_0_1"]) for segment in scored_segments],
        [float(segment["duration"]) for segment in scored_segments],
    )
    status_durations: dict[str, float] = {}
    for segment in segments:
        status = str(segment["status"])
        status_durations[status] = status_durations.get(status, 0.0) + float(segment["duration"])
    scored_duration = status_durations.get("scored", 0.0)

    payload = {
        "metric": METRIC_NAME,
        "metric_version": METRIC_VERSION,
        "video_path": str(video_path),
        "video_status": "scored",
        "fps": FPS,
        "audio_sample_rate": AUDIO_SAMPLE_RATE,
        "duration": duration,
        "frame_count": frame_count,
        "frame_width": width,
        "frame_height": height,
        "window_policy": "non_overlapping_2s",
        "thresholds": thresholds,
        "num_face_tracks": len(tracks),
        "num_speech_segments": len(speech_segments),
        "num_windows": len(segments),
        "num_scored_windows": len(scored_segments),
        "speech_duration": speech_duration,
        "scored_speech_duration": scored_duration,
        "offscreen_speech_duration": status_durations.get("offscreen_voice", 0.0),
        "unreliable_speech_duration": (
            status_durations.get("no_active_speaker", 0.0)
            + status_durations.get("insufficient_track", 0.0)
            + status_durations.get("syncnet_error", 0.0)
            + status_durations.get("unreliable_sync", 0.0)
        ),
        "scored_speech_ratio": scored_duration / speech_duration if speech_duration > 0 else 0.0,
        "video_score_0_1": score,
        "video_mos_1_5": (1.0 + 4.0 * score) if score is not None else None,
        "speech_segments": [asdict(segment) for segment in speech_segments],
        "status_counts": {
            status: sum(segment["status"] == status for segment in segments)
            for status in (
                "scored",
                "offscreen_voice",
                "insufficient_track",
                "no_active_speaker",
                "syncnet_error",
                "unreliable_sync",
            )
        },
        "segments": segments,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    progress(
        f"finish video_status={payload['video_status']} score={payload['video_score_0_1']} "
        f"scored_ratio={payload['scored_speech_ratio']:.3f} output={output_json}"
    )
    if not keep_temp:
        progress("delete transient work directories")
        shutil.rmtree(work_dir / "standard", ignore_errors=True)
        shutil.rmtree(sync_temp_dir, ignore_errors=True)
    if not keep_crops:
        progress("delete crop directory")
        shutil.rmtree(crops_dir, ignore_errors=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Visible Speech-Lip Synchronization with SyncNet.")
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--latentsync-dir", type=Path, default=DEFAULT_LATENTSYNC_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ffmpeg-bin", default=None)
    parser.add_argument("--vad-backend", choices=["silero", "webrtcvad", "energy"], default=VAD_BACKEND)
    parser.add_argument("--delete-crops", action="store_true")
    parser.add_argument("--delete-temp", action="store_true", help="Delete standard/syncnet_temp work files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ffmpeg_bin = resolve_ffmpeg(args.ffmpeg_bin)
    try:
        payload = evaluate_video(
            video_path=args.video_path.expanduser().resolve(),
            latentsync_dir=args.latentsync_dir.expanduser().resolve(),
            model_path=args.model_path.expanduser().resolve(),
            work_dir=args.work_dir.expanduser().resolve(),
            output_json=args.output_json.expanduser().resolve(),
            device=args.device,
            ffmpeg_bin=ffmpeg_bin,
            keep_crops=not args.delete_crops,
            keep_temp=not args.delete_temp,
            vad_backend=args.vad_backend,
        )
    except Exception as exc:  # noqa: BLE001
        output_json = args.output_json.expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metric": METRIC_NAME,
            "metric_version": METRIC_VERSION,
            "video_path": str(args.video_path.expanduser().resolve()),
            "video_status": "error",
            "video_score_0_1": None,
            "video_mos_1_5": None,
            "error": str(exc),
        }
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
