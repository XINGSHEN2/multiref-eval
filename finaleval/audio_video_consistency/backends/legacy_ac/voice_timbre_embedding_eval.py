#!/usr/bin/env python3
"""Speaker-embedding voice timbre evaluation for AC outputs.

This script scores voice timbre similarity with ECAPA speaker embeddings.
It is calibrated for short generated video speech:

- generated and reference speech are VAD-trimmed with a simple energy VAD;
- both sides must contain at least --min-speech-sec speech, default 2.5s;
- scores are mapped from cosine similarity using the final short-video
  calibration: cosine >=0.300 -> 5, >=0.225 -> 4, >=0.150 -> 3,
  >=0.075 -> 2, else 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio


SAMPLE_RATE = 16000
DEFAULT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


@dataclass
class SpeechAudio:
    waveform: torch.Tensor
    speech_sec: float
    total_sec: float
    segments: list[tuple[float, float]]


def die_missing_speechbrain() -> None:
    raise SystemExit(
        "Missing dependency: speechbrain. Install it in the target env, e.g.\n"
        "  pip install speechbrain\n"
        "The model weights are downloaded from HuggingFace on first use:\n"
        f"  {DEFAULT_MODEL}\n"
    )


def run_ffmpeg_extract(video_path: Path, wav_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        str(wav_path),
    ]
    subprocess.run(cmd, check=True)


def load_audio(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.numel() == 0:
        return torch.zeros(1, 0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.float()


def load_video_audio(video_path: Path, tmp_dir: Path) -> torch.Tensor:
    wav_path = tmp_dir / f"{video_path.stem}_audio.wav"
    run_ffmpeg_extract(video_path, wav_path)
    return load_audio(wav_path)


def energy_vad(
    wav: torch.Tensor,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    threshold_ratio: float = 0.08,
    min_segment_sec: float = 0.18,
    pad_sec: float = 0.08,
) -> SpeechAudio:
    """Return speech-like regions using a conservative energy threshold.

    This is not as strong as Silero/pyannote VAD, but avoids extra dependencies
    and is good enough to reject silence/music-only failures in this small run.
    """

    wav = wav[:1]
    n = wav.shape[1]
    total_sec = n / SAMPLE_RATE
    if n == 0:
        return SpeechAudio(wav, 0.0, 0.0, [])

    x = wav.squeeze(0)
    frame = max(1, int(SAMPLE_RATE * frame_ms / 1000.0))
    hop = max(1, int(SAMPLE_RATE * hop_ms / 1000.0))
    if n < frame:
        rms = torch.sqrt(torch.mean(x * x) + 1e-12).item()
        if rms <= 1e-4:
            return SpeechAudio(torch.zeros(1, 0), 0.0, total_sec, [])
        return SpeechAudio(wav, total_sec, total_sec, [(0.0, total_sec)])

    frames = x.unfold(0, frame, hop)
    rms = torch.sqrt(torch.mean(frames * frames, dim=1) + 1e-12)
    # Use robust threshold: not too low, but adapts to quiet speech.
    peak = float(torch.quantile(rms, 0.95).item())
    median = float(torch.median(rms).item())
    thr = max(1e-4, median * 2.5, peak * threshold_ratio)
    mask = (rms >= thr).cpu().tolist()

    segments: list[tuple[int, int]] = []
    start: int | None = None
    for i, active in enumerate(mask):
        if active and start is None:
            start = i
        if (not active or i == len(mask) - 1) and start is not None:
            end_i = i if not active else i + 1
            s = max(0, start * hop - int(pad_sec * SAMPLE_RATE))
            e = min(n, end_i * hop + frame + int(pad_sec * SAMPLE_RATE))
            if (e - s) / SAMPLE_RATE >= min_segment_sec:
                segments.append((s, e))
            start = None

    # Merge close regions.
    merged: list[tuple[int, int]] = []
    gap = int(0.18 * SAMPLE_RATE)
    for s, e in segments:
        if not merged or s - merged[-1][1] > gap:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], e)

    if not merged:
        return SpeechAudio(torch.zeros(1, 0), 0.0, total_sec, [])

    speech = torch.cat([wav[:, s:e] for s, e in merged], dim=1)
    speech_sec = speech.shape[1] / SAMPLE_RATE
    return SpeechAudio(
        speech,
        speech_sec,
        total_sec,
        [(s / SAMPLE_RATE, e / SAMPLE_RATE) for s, e in merged],
    )


def encode_embedding(classifier: Any, wav: torch.Tensor, device: str) -> torch.Tensor:
    if wav.shape[1] == 0:
        raise ValueError("empty waveform")
    wav = wav.to(device)
    with torch.no_grad():
        emb = classifier.encode_batch(wav)
    return emb.squeeze().detach().cpu()


def score_from_cosine(cosine: float) -> int:
    if cosine >= 0.300:
        return 5
    if cosine >= 0.225:
        return 4
    if cosine >= 0.150:
        return 3
    if cosine >= 0.075:
        return 2
    return 1


def load_tasks(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def voice_applicable(task: dict[str, Any]) -> bool:
    spec = task.get("ac_eval_spec") or {}
    dims = spec.get("applicable_dimensions") or {}
    ref = spec.get("reference_audio") or {}
    return bool(dims.get("Voice Timbre Similarity") and ref.get("path"))


def evaluate_task(
    task: dict[str, Any],
    classifier: Any,
    device: str,
    tmp_dir: Path,
    min_speech_sec: float,
) -> dict[str, Any]:
    spec = task.get("ac_eval_spec") or {}
    ref_info = spec.get("reference_audio") or {}
    ref_path = Path(ref_info.get("path", ""))
    video_path = Path(task.get("video_path", ""))
    row = {
        "task_id": task.get("task_id", ""),
        "sample_id": task.get("sample_id", ""),
        "source_board": spec.get("source_board", ""),
        "model_name": task.get("model_name", ""),
        "reference_audio": str(ref_path),
        "video_path": str(video_path),
        "ref_speech_sec": "",
        "gen_speech_sec": "",
        "cosine_similarity": "",
        "score": "",
        "confidence": "",
        "status": "",
        "mixed_speaker_risk": "",
        "ref_segments": "",
        "gen_segments": "",
        "error": "",
    }

    try:
        if not ref_path.exists():
            row["status"] = "missing_reference_audio"
            return row
        if not video_path.exists():
            row["status"] = "missing_video"
            return row

        ref = energy_vad(load_audio(ref_path))
        gen = energy_vad(load_video_audio(video_path, tmp_dir))
        row["ref_speech_sec"] = f"{ref.speech_sec:.3f}"
        row["gen_speech_sec"] = f"{gen.speech_sec:.3f}"
        row["ref_segments"] = json.dumps(ref.segments, ensure_ascii=False)
        row["gen_segments"] = json.dumps(gen.segments, ensure_ascii=False)

        if ref.speech_sec < min_speech_sec:
            row["status"] = "too_short_reference"
            return row
        if gen.speech_sec < min_speech_sec:
            row["status"] = "too_short_generated"
            return row

        ref_emb = encode_embedding(classifier, ref.waveform, device)
        gen_emb = encode_embedding(classifier, gen.waveform, device)
        cosine = float(F.cosine_similarity(ref_emb, gen_emb, dim=0).item())
        row["cosine_similarity"] = f"{cosine:.6f}"
        row["score"] = str(score_from_cosine(cosine))
        row["confidence"] = "low" if min(ref.speech_sec, gen.speech_sec) < 3.0 else "normal"
        row["mixed_speaker_risk"] = "true" if len(spec.get("speech_prompt") or []) > 1 else "false"
        row["status"] = "scored"
        return row
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = repr(exc)
        return row


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "voice_timbre_embedding_scores.csv"
    json_path = output_dir / "voice_timbre_embedding_scores.json"
    md_path = output_dir / "voice_timbre_embedding_summary.md"

    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    scored = [
        r
        for r in rows
        if r["status"] == "scored" and r["score"] and r["cosine_similarity"]
    ]
    by_model: dict[str, list[dict[str, float]]] = {}
    for r in scored:
        by_model.setdefault(r["model_name"], []).append(
            {
                "score": float(r["score"]),
                "cosine": float(r["cosine_similarity"]),
            }
        )

    lines = [
        "# Voice Timbre Embedding Summary",
        "",
        "- Model: `speechbrain/spkrec-ecapa-voxceleb`",
        "- VAD: built-in energy VAD",
        "- Minimum speech duration: controlled by `--min-speech-sec`, default 2.5s",
        "- Score mapping: cosine >=0.300 -> 5, >=0.225 -> 4, >=0.150 -> 3, >=0.075 -> 2, else 1",
        "",
        "## Model Means",
        "",
        "| Model | Raw Cosine Mean | Raw Cosine Std | Raw Cosine Min | Raw Cosine Max | Mean Score | Count |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in sorted(by_model):
        vals = by_model[model]
        scores = [v["score"] for v in vals]
        cosines = [v["cosine"] for v in vals]
        mean_cos = sum(cosines) / len(cosines)
        if len(cosines) > 1:
            var = sum((v - mean_cos) ** 2 for v in cosines) / (len(cosines) - 1)
            std_cos = math.sqrt(var)
        else:
            std_cos = 0.0
        lines.append(
            f"| {model} | {mean_cos:.6f} | {std_cos:.6f} | "
            f"{min(cosines):.6f} | {max(cosines):.6f} | "
            f"{sum(scores)/len(scores):.4f} | {len(scores)} |"
        )
    lines.extend(
        [
            "",
            "## Score Distribution",
            "",
            "| Model | Score | Count |",
            "|---|---:|---:|",
        ]
    )
    score_counts: dict[tuple[str, int], int] = {}
    for r in scored:
        key = (r["model_name"], int(float(r["score"])))
        score_counts[key] = score_counts.get(key, 0) + 1
    for (model, score), count in sorted(score_counts.items()):
        lines.append(f"| {model} | {score} | {count} |")
    lines.extend(
        [
            "",
            "## Status Counts",
            "",
            "| Status | Count |",
            "|---|---:|",
        ]
    )
    status_counts: dict[str, int] = {}
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")
    five_rows = [
        r
        for r in scored
        if int(float(r["score"])) == 5
    ]
    if five_rows:
        lines.extend(
            [
                "",
                "## Score 5 Rows",
                "",
                "| Sample | Model | Cosine | Ref Speech Sec | Gen Speech Sec |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for r in sorted(five_rows, key=lambda x: (x["model_name"], x["sample_id"])):
            lines.append(
                f"| {r['sample_id']} | {r['model_name']} | "
                f"{float(r['cosine_similarity']):.6f} | "
                f"{float(r['ref_speech_sec']):.2f} | "
                f"{float(r['gen_speech_sec']):.2f} |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Raw cosine is the embedding similarity before score mapping.",
            "- This metric should be interpreted as embedding-based relative timbre similarity, not strict speaker verification.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-speech-sec", type=float, default=2.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--model-source", default=DEFAULT_MODEL)
    parser.add_argument("--savedir", default="pretrained_models/spkrec-ecapa-voxceleb")
    args = parser.parse_args()

    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception:
        die_missing_speechbrain()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(
        source=args.model_source,
        savedir=args.savedir,
        run_opts={"device": device},
    )

    tasks = [t for t in load_tasks(args.tasks) if voice_applicable(t)]
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="voice_timbre_embedding_") as tmp:
        tmp_dir = Path(tmp)
        for i, task in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] {task.get('sample_id')}::{task.get('model_name')}", flush=True)
            rows.append(evaluate_task(task, classifier, device, tmp_dir, args.min_speech_sec))
    write_outputs(rows, args.output_dir)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
