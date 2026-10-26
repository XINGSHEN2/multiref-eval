from __future__ import annotations

from pathlib import Path

from finaleval.common.legacy_backend import ENTITY_FIDELITY_BACKEND, run_shell_metric
from finaleval.common.schema import InputSample


METRIC_NAME = "detail_preservation"


def evaluate(samples: list[InputSample], output_root: Path, **_: object) -> Path:
    return run_shell_metric(
        samples,
        output_root,
        METRIC_NAME,
        "entity_fidelity",
        "finaleval/entity_fidelity/backends/legacy_ef/run_mllm_detail_binding_from_xlsx.sh detail",
        ENTITY_FIDELITY_BACKEND,
        ["bash", "run_mllm_detail_binding_from_xlsx.sh"],
        ["Detail Preservation Score", "detail_preservation_score", "score", "mean_score"],
        ["detail_preservation_per_sample.csv"],
        env={"METRIC": "detail", "DP_OUTPUT_DIR": str(output_root / "_legacy" / METRIC_NAME)},
    )
