from __future__ import annotations

import os
from pathlib import Path

from finaleval.common.legacy_backend import (
    ENTITY_FIDELITY_BACKEND,
    python_bin,
    run_command,
    run_shell_metric,
    write_backend_error,
    write_manifest,
)
from finaleval.common.schema import InputSample


METRIC_NAME = "entity_fidelity"


def run_paste_judge(samples: list[InputSample], output_root: Path) -> Path:
    legacy_run = output_root / "_legacy" / METRIC_NAME
    manifest = write_manifest(samples, legacy_run / "input_manifest.csv")
    paste_output_dir = legacy_run / "paste_artifact"
    env = {
        "CSV": str(manifest),
        "BOARD": "all",
        "PASTE_OUTPUT_DIR": str(paste_output_dir),
        "PASTE_RUN_NAME": "paste_artifact_entity_fidelity",
        "PASTE_PYTHON": python_bin(),
    }
    run_command(["bash", "run_mllm_paste_artifact_from_xlsx.sh"], cwd=ENTITY_FIDELITY_BACKEND, env=env)
    latest_file = paste_output_dir / "latest_run.txt"
    if not latest_file.exists():
        raise FileNotFoundError(f"Paste judge did not write latest_run.txt: {latest_file}")
    latest_run = Path(latest_file.read_text(encoding="utf-8").strip())
    paste_csv = latest_run / "paste_artifact_per_sample.csv"
    if not paste_csv.exists():
        raise FileNotFoundError(f"Paste judge did not write per-sample CSV: {paste_csv}")
    return paste_csv


def evaluate(samples: list[InputSample], output_root: Path, **kwargs: object) -> Path:
    env = {}
    paste_per_sample = kwargs.get("paste_per_sample")
    auto_paste = kwargs.get("auto_paste")
    if auto_paste is None:
        auto_paste = os.getenv("MULTIREF_COMPASS_AUTO_PASTE", "1").strip().lower() not in {"0", "false", "no"}
    if paste_per_sample:
        env["PASTE_PER_SAMPLE"] = str(Path(str(paste_per_sample)).expanduser().resolve())
    elif auto_paste:
        try:
            env["PASTE_PER_SAMPLE"] = str(run_paste_judge(samples, output_root))
        except Exception as exc:
            return write_backend_error(
                samples,
                output_root,
                METRIC_NAME,
                "entity_fidelity",
                exc,
                "finaleval/entity_fidelity/backends/legacy_ef/run_mllm_paste_artifact_from_xlsx.sh",
            )
    return run_shell_metric(
        samples,
        output_root,
        METRIC_NAME,
        "entity_fidelity",
        "finaleval/entity_fidelity/backends/legacy_ef/run_entity_fidelity_from_xlsx.sh",
        ENTITY_FIDELITY_BACKEND,
        ["bash", "run_entity_fidelity_from_xlsx.sh"],
        ["final_entity_fidelity", "entity_fidelity", "ef_score", "score", "mean_ef"],
        ["summary.csv", "entity_fidelity_per_sample.csv"],
        env=env,
    )
