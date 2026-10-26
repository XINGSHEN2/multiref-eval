from __future__ import annotations

from pathlib import Path

from finaleval.common.legacy_backend import ENTITY_FIDELITY_BACKEND, run_shell_metric
from finaleval.common.schema import InputSample


METRIC_NAME = "detail_binding"


def evaluate(samples: list[InputSample], output_root: Path, **_: object) -> Path:
    return run_shell_metric(
        samples,
        output_root,
        METRIC_NAME,
        "entity_fidelity",
        "finaleval/entity_fidelity/backends/legacy_ef/run_mllm_detail_binding_from_xlsx.sh binding",
        ENTITY_FIDELITY_BACKEND,
        ["bash", "run_mllm_detail_binding_from_xlsx.sh"],
        ["Binding Score", "binding_score", "score", "mean_score"],
        ["binding_per_sample.csv"],
        env={"METRIC": "binding", "BINDING_OUTPUT_DIR": str(output_root / "_legacy" / METRIC_NAME)},
    )
