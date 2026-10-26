from __future__ import annotations

from pathlib import Path

from finaleval.common.legacy_backend import INSTRUCTION_FOLLOWING_BACKEND, run_shell_metric
from finaleval.common.schema import InputSample


METRIC_NAME = "temporal_order_following"


def evaluate(samples: list[InputSample], output_root: Path, **_: object) -> Path:
    return run_shell_metric(
        samples,
        output_root,
        METRIC_NAME,
        "instruction_following",
        "finaleval/instruction_following/backends/legacy_if/run_if_full_from_table.sh temporal",
        INSTRUCTION_FOLLOWING_BACKEND,
        ["bash", "run_if_full_from_table.sh", str(output_root / "_legacy" / METRIC_NAME / "input_manifest.csv"), "all"],
        ["Temporal Order Following", "temporal_order_following", "score"],
        ["if_per_sample.csv", "instruction_following_per_sample.csv"],
        env={"IF_DIMENSION": "temporal", "IF_FINAL_OUTPUT_DIR": str(output_root / "_legacy" / METRIC_NAME)},
    )
