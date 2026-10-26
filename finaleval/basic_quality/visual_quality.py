from __future__ import annotations

from pathlib import Path

from finaleval.basic_quality._legacy_bq import run_basic_quality_metric
from finaleval.common.schema import InputSample


METRIC_NAME = "visual_quality"


def evaluate(samples: list[InputSample], output_root: Path, **_: object) -> Path:
    return run_basic_quality_metric(
        samples,
        output_root,
        METRIC_NAME,
        ["visual_technical_quality_doverpp", "visual_quality", "visual_mean"],
        ["visual_technical_quality_doverpp"],
    )
