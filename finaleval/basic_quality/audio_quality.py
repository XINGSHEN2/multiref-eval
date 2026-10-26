from __future__ import annotations

from pathlib import Path

from finaleval.basic_quality._legacy_bq import run_basic_quality_metric
from finaleval.common.schema import InputSample


METRIC_NAME = "audio_quality"


def evaluate(samples: list[InputSample], output_root: Path, **_: object) -> Path:
    return run_basic_quality_metric(
        samples,
        output_root,
        METRIC_NAME,
        ["audio_technical_quality", "audio_quality", "audio_mean"],
        ["audio_technical_quality"],
    )
