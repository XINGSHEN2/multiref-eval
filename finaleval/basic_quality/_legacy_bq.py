from __future__ import annotations

from pathlib import Path

from finaleval.common.legacy_backend import (
    BASIC_QUALITY_BACKEND,
    find_csv,
    normalize_csv_rows,
    python_bin,
    run_command,
    write_backend_error,
    write_manifest,
)
from finaleval.common.output_writer import write_metric_outputs
from finaleval.common.schema import InputSample


def run_basic_quality_metric(
    samples: list[InputSample],
    output_root: Path,
    metric_name: str,
    score_fields: list[str],
    extra_fields: list[str] | None = None,
) -> Path:
    legacy_run = output_root / "_legacy" / "basic_quality"
    manifest = write_manifest(samples, legacy_run / "input_manifest.csv")
    try:
        run_command(
            [
                python_bin(),
                BASIC_QUALITY_BACKEND / "run_basic_quality_from_csv.py",
                "--input-csv",
                manifest,
                "--output-dir",
                legacy_run,
                "--models",
                "auto",
                "--expected-samples-per-model",
                "auto",
                "--audio-tool",
                "audiobox",
            ],
            cwd=BASIC_QUALITY_BACKEND,
        )
        source_csv = find_csv(legacy_run, ["metric_details.csv"])
        if source_csv is None:
            raise FileNotFoundError(f"No Basic Quality metric_details.csv under {legacy_run}")
        rows = normalize_csv_rows(samples, metric_name, source_csv, score_fields, extra_fields=extra_fields)
        return write_metric_outputs(
            output_root,
            metric_name,
            rows,
            details={
                "metric": metric_name,
                "group": "basic_quality",
                "implemented": True,
                "backend": "finaleval/basic_quality/backends/legacy_bq/run_basic_quality_from_csv.py",
                "source_csv": str(source_csv),
            },
            extra_summary_fields=extra_fields,
        )
    except Exception as exc:
        return write_backend_error(samples, output_root, metric_name, "basic_quality", exc, "Basic Quality")
