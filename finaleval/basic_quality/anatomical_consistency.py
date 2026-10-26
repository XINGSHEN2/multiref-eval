from __future__ import annotations

from pathlib import Path

from finaleval.common.legacy_backend import BASIC_QUALITY_BACKEND, run_shell_metric
from finaleval.common.schema import InputSample


METRIC_NAME = "anatomical_consistency"


def evaluate(samples: list[InputSample], output_root: Path, **_: object) -> Path:
    return run_shell_metric(
        samples,
        output_root,
        METRIC_NAME,
        "basic_quality",
        "finaleval/basic_quality/backends/legacy_bq/run_mllm_anatomical_from_xlsx.sh",
        BASIC_QUALITY_BACKEND,
        ["bash", "run_mllm_anatomical_from_xlsx.sh"],
        ["Object Integrity Score", "overall_score", "score", "mean_score"],
        ["anatomical_consistency_per_sample.csv", "anatomical_per_sample.csv"],
    )
