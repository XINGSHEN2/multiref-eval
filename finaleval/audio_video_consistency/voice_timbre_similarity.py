from __future__ import annotations

import json
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from finaleval.common.media import SAMPLE_RATE, extract_video_audio, load_audio_16k
from finaleval.common.output_writer import write_metric_outputs
from finaleval.common.schema import InputSample


DEFAULT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
METRIC_NAME = "voice_timbre_similarity"


@dataclass
class SpeechAudio:
    waveform: torch.Tensor
    speech_sec: float
    total_sec: float
    segments: list[tuple[float, float]]


def energy_vad(
    wav: torch.Tensor,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    threshold_ratio: float = 0.08,
    min_segment_sec: float = 0.18,
    pad_sec: float = 0.08,
) -> SpeechAudio:
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
    peak = float(torch.quantile(rms, 0.95).item())
    median = float(torch.median(rms).item())
    threshold = max(1e-4, median * 2.5, peak * threshold_ratio)
    mask = (rms >= threshold).cpu().tolist()

    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        if (not active or index == len(mask) - 1) and start is not None:
            end_i = index if not active else index + 1
            s = max(0, start * hop - int(pad_sec * SAMPLE_RATE))
            e = min(n, end_i * hop + frame + int(pad_sec * SAMPLE_RATE))
            if (e - s) / SAMPLE_RATE >= min_segment_sec:
                segments.append((s, e))
            start = None

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
    return SpeechAudio(
        speech,
        speech.shape[1] / SAMPLE_RATE,
        total_sec,
        [(s / SAMPLE_RATE, e / SAMPLE_RATE) for s, e in merged],
    )


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


def encode_embedding(classifier: Any, wav: torch.Tensor, device: str) -> torch.Tensor:
    if wav.shape[1] == 0:
        raise ValueError("empty waveform")
    wav = wav.to(device)
    with torch.no_grad():
        emb = classifier.encode_batch(wav)
    return emb.squeeze().detach().cpu()


def base_row(sample: InputSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "model_name": sample.model_name,
        "video_path": sample.video_path,
        "metric_name": METRIC_NAME,
        "score": "",
        "status": "",
        "reason": "",
        "raw_score": "",
        "cosine_similarity": "",
        "ref_speech_sec": "",
        "gen_speech_sec": "",
        "ref_segments": "",
        "gen_segments": "",
    }


def evaluate_sample(
    sample: InputSample,
    classifier: Any,
    device: str,
    tmp_dir: Path,
    min_speech_sec: float,
) -> dict[str, Any]:
    row = base_row(sample)
    try:
        if not sample.reference_audio_path:
            row.update(status="not_applicable", reason="missing reference_audio_path")
            return row
        if not sample.video_path:
            row.update(status="not_applicable", reason="missing video_path")
            return row

        ref_path = Path(sample.reference_audio_path)
        video_path = Path(sample.video_path)
        if not ref_path.exists():
            row.update(status="missing_reference_audio", reason=str(ref_path))
            return row
        if not video_path.exists():
            row.update(status="missing_video", reason=str(video_path))
            return row

        video_wav = tmp_dir / f"{sample.sample_id}_{sample.model_name}_audio.wav"
        extract_video_audio(video_path, video_wav)
        ref = energy_vad(load_audio_16k(ref_path))
        gen = energy_vad(load_audio_16k(video_wav))

        row["ref_speech_sec"] = f"{ref.speech_sec:.3f}"
        row["gen_speech_sec"] = f"{gen.speech_sec:.3f}"
        row["ref_segments"] = json.dumps(ref.segments, ensure_ascii=False)
        row["gen_segments"] = json.dumps(gen.segments, ensure_ascii=False)

        if ref.speech_sec < min_speech_sec:
            row.update(status="too_short_reference", reason=f"reference speech < {min_speech_sec}s")
            return row
        if gen.speech_sec < min_speech_sec:
            row.update(status="too_short_generated", reason=f"generated speech < {min_speech_sec}s")
            return row

        ref_emb = encode_embedding(classifier, ref.waveform, device)
        gen_emb = encode_embedding(classifier, gen.waveform, device)
        cosine = float(F.cosine_similarity(ref_emb, gen_emb, dim=0).item())
        score = score_from_cosine(cosine)
        row["cosine_similarity"] = f"{cosine:.6f}"
        row["raw_score"] = f"{cosine:.6f}"
        row["score"] = score
        row["status"] = "scored"
        row["reason"] = ""
        return row
    except Exception as exc:  # noqa: BLE001
        row.update(status="error", reason=repr(exc))
        return row


def evaluate(
    samples: list[InputSample],
    output_root: Path,
    min_speech_sec: float = 2.5,
    device: str = "auto",
    model_source: str = DEFAULT_MODEL,
    model_cache: str | None = None,
) -> Path:
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("speechbrain is required for voice_timbre_similarity") from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    savedir = model_cache or "pretrained_models/spkrec-ecapa-voxceleb"
    classifier = EncoderClassifier.from_hparams(
        source=model_source,
        savedir=savedir,
        run_opts={"device": device},
    )

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="finaleval_voice_timbre_") as tmp:
        tmp_dir = Path(tmp)
        for sample in samples:
            rows.append(evaluate_sample(sample, classifier, device, tmp_dir, min_speech_sec))

    details = {
        "metric": METRIC_NAME,
        "model": model_source,
        "min_speech_sec": min_speech_sec,
        "score_mapping": {
            ">=0.300": 5,
            ">=0.225": 4,
            ">=0.150": 3,
            ">=0.075": 2,
            "<0.075": 1,
        },
        "input_samples": [asdict(sample) for sample in samples],
    }
    return write_metric_outputs(
        output_root,
        METRIC_NAME,
        rows,
        details,
        extra_summary_fields=["cosine_similarity", "ref_speech_sec", "gen_speech_sec"],
    )

