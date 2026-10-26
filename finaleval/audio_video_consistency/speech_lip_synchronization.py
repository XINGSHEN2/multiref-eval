from __future__ import annotations

from pathlib import Path

from finaleval.common.legacy_backend import AUDIO_VIDEO_BACKEND, run_shell_metric
from finaleval.common.schema import InputSample


METRIC_NAME = "speech_lip_synchronization"


def evaluate(samples: list[InputSample], output_root: Path, stable_face_filter: bool = True, **_: object) -> Path:
    metrics = "stable_speaking_frontal_face,vsls_eval" if stable_face_filter else "vsls_eval"
    return run_shell_metric(
        samples,
        output_root,
        METRIC_NAME,
        "audio_video_consistency",
        "finaleval/audio_video_consistency/backends/legacy_ac/run_pure_ac.sh vsls_eval",
        AUDIO_VIDEO_BACKEND,
        [
            "bash",
            "run_pure_ac.sh",
            "--input-csv",
            str(output_root / "_legacy" / METRIC_NAME / "input_manifest.csv"),
            "--metrics",
            metrics,
            "--run-name",
            METRIC_NAME,
            "--output-root",
            str(output_root / "_legacy" / METRIC_NAME),
        ],
        [
            "Speech-Lip Synchronization_score",
            "Stable Speaking Frontal Face_score",
            "Speech-Lip Synchronization",
            "Visible Speech-Lip Synchronization",
            "vsls_score",
            "score",
        ],
        ["vsls_results.csv", "ac_per_sample.csv"],
    )
