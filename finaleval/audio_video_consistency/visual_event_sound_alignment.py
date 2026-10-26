from __future__ import annotations

from pathlib import Path

from finaleval.common.legacy_backend import AUDIO_VIDEO_BACKEND, run_shell_metric
from finaleval.common.schema import InputSample


METRIC_NAME = "visual_event_sound_alignment"


def evaluate(samples: list[InputSample], output_root: Path, **_: object) -> Path:
    return run_shell_metric(
        samples,
        output_root,
        METRIC_NAME,
        "audio_video_consistency",
        "finaleval/audio_video_consistency/backends/legacy_ac/run_pure_ac.sh visual_event_sound_alignment",
        AUDIO_VIDEO_BACKEND,
        [
            "bash",
            "run_pure_ac.sh",
            "--input-csv",
            str(output_root / "_legacy" / METRIC_NAME / "input_manifest.csv"),
            "--metrics",
            "visual_event_sound_alignment",
            "--run-name",
            METRIC_NAME,
            "--output-root",
            str(output_root / "_legacy" / METRIC_NAME),
        ],
        ["Visual Event-Sound Alignment_score", "Visual Event-Sound Alignment", "visual_event_sound_alignment", "score"],
        ["ac_per_sample.csv"],
    )
