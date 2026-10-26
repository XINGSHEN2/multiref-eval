from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from finaleval.common.legacy_backend import (
    INSTRUCTION_FOLLOWING_BACKEND,
    find_csv,
    first,
    normalize_csv_rows,
    read_csv,
    run_command,
    sample_error_rows,
    write_manifest,
)
from finaleval.common.schema import InputSample


METRIC_NAME = "visual_task_following"
CATEGORY_FIELDS = [
    "important_action",
    "wearing_requirement",
    "basic_requirement",
    "other_detail",
]


def _mean_or_empty(rows: list[dict[str, Any]], field: str) -> str:
    values = [float(row[field]) for row in rows if row.get(field) not in ("", None)]
    return f"{mean(values):.6f}" if values else ""


def evaluate(samples: list[InputSample], output_root: Path, **_: object) -> Path:
    metric_dir = output_root / "metrics" / METRIC_NAME
    metric_dir.mkdir(parents=True, exist_ok=True)

    legacy_run = output_root / "_legacy" / METRIC_NAME
    manifest = write_manifest(samples, legacy_run / "input_manifest.csv")
    try:
        run_command(
            ["bash", "run_if_full_from_table.sh", str(manifest), "all"],
            cwd=INSTRUCTION_FOLLOWING_BACKEND,
            env={"IF_DIMENSION": "visual", "IF_FINAL_OUTPUT_DIR": str(legacy_run)},
        )
        source_csv = find_csv(
            legacy_run,
            ["visual_final_per_sample.csv", "if_per_sample.csv", "instruction_following_per_sample.csv"],
        )
        if source_csv is None:
            raise FileNotFoundError(f"No IF per-sample CSV under {legacy_run}")
        rows = normalize_csv_rows(
            samples,
            METRIC_NAME,
            source_csv,
            ["Visual Task Following", "visual_task_following", "score"],
            extra_fields=CATEGORY_FIELDS,
        )
        raw_rows = read_csv(source_csv)
        raw_by_key = {
            (first(row, ["sample_id", "sample", "task_id"]), first(row, ["model_name", "model"])): row
            for row in raw_rows
        }
        for row in rows:
            raw = raw_by_key.get((row["sample_id"], row["model_name"]), {})
            row["important_action"] = first(raw, ["important_action", "Important Action", "important_action_score"])
            row["wearing_requirement"] = first(raw, ["wearing_requirement", "Wearing Requirement", "wearing_requirement_score"])
            row["basic_requirement"] = first(raw, ["basic_requirement", "Basic Requirement", "basic_requirement_score"])
            row["other_detail"] = first(raw, ["other_detail", "Other Detail", "other_detail_score"])
    except Exception as exc:
        rows = sample_error_rows(
            samples,
            METRIC_NAME,
            f"finaleval/instruction_following/backends/legacy_if visual backend failed: {exc}",
        )
        for row in rows:
            for field in CATEGORY_FIELDS:
                row[field] = ""

    per_sample_fields = [
        "sample_id",
        "model_name",
        "video_path",
        "metric_name",
        "score",
        *CATEGORY_FIELDS,
        "status",
        "reason",
        "raw_score",
        "legacy_source_csv",
    ]
    with (metric_dir / "per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_sample_fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["model_name"]].append(row)

    summary_rows: list[dict[str, Any]] = []
    for model_name, model_rows in sorted(grouped.items()):
        scored = [row for row in model_rows if row["status"] == "scored"]
        item = {
            "model_name": model_name,
            "metric_name": METRIC_NAME,
            "mean_score": _mean_or_empty(scored, "score"),
            "important_action": _mean_or_empty(scored, "important_action"),
            "wearing_requirement": _mean_or_empty(scored, "wearing_requirement"),
            "basic_requirement": _mean_or_empty(scored, "basic_requirement"),
            "other_detail": _mean_or_empty(scored, "other_detail"),
            "num_scored": len(scored),
            "num_failed": sum(1 for row in model_rows if row["status"] == "error"),
            "num_skipped": sum(1 for row in model_rows if row["status"] != "scored"),
        }
        summary_rows.append(item)

    summary_fields = [
        "model_name",
        "metric_name",
        "mean_score",
        *CATEGORY_FIELDS,
        "num_scored",
        "num_failed",
        "num_skipped",
    ]
    with (metric_dir / "model_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    ranked = sorted(
        [row for row in summary_rows if row["mean_score"] != ""],
        key=lambda row: float(row["mean_score"]),
        reverse=True,
    )
    lines = [
        "# Visual Task Following Ranking",
        "",
        "| Rank | Model | Overall | Important Action | Wearing Requirement | Basic Requirement | Other Detail | Scored | Failed | Skipped |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(value: object) -> str:
        if value in ("", None):
            return ""
        return f"{float(value):.4f}"

    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {row['model_name']} | {fmt(row['mean_score'])} | "
            f"{fmt(row['important_action'])} | {fmt(row['wearing_requirement'])} | "
            f"{fmt(row['basic_requirement'])} | {fmt(row['other_detail'])} | "
            f"{row['num_scored']} | {row['num_failed']} | {row['num_skipped']} |"
        )
    if not ranked:
        lines.extend(
            [
                "",
                "No scored rows yet. The output schema is fixed; the scoring backend is not migrated yet.",
            ]
        )
    (metric_dir / "ranking.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    details = {
        "metric": METRIC_NAME,
        "group": "instruction_following",
        "implemented": True,
        "backend": "finaleval/instruction_following/backends/legacy_if/run_if_full_from_table.sh visual",
        "required_columns": ["video_path", "prompt"],
        "ranking_columns": [
            "overall",
            "important_action",
            "wearing_requirement",
            "basic_requirement",
            "other_detail",
        ],
        "note": "Visual Task Following reports weighted overall plus four category breakdowns.",
    }
    (metric_dir / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    return metric_dir
