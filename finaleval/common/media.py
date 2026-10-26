from __future__ import annotations

import subprocess
from pathlib import Path

import torch
import torchaudio


SAMPLE_RATE = 16000


def extract_video_audio(video_path: Path, wav_path: Path) -> None:
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


def load_audio_16k(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.numel() == 0:
        return torch.zeros(1, 0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.float()

