from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def write_metric_outputs(
    output_root: Path,
    metric_name: str,
    rows: list[dict[str, Any]],
    details: dict[str, Any] | None = None,
    extra_summary_fields: list[str] | None = None,
) -> Path:
    metric_dir = output_root / "metrics" / metric_name
    metric_dir.mkdir(parents=True, exist_ok=True)

    per_sample_path = metric_dir / "per_sample.csv"
    details_path = metric_dir / "details.json"
    summary_path = metric_dir / "model_summary.csv"
    ranking_path = metric_dir / "ranking.md"

    fields = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "sample_id",
        "model_name",
        "video_path",
        "metric_name",
        "score",
        "status",
        "reason",
        "raw_score",
    ]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with per_sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    details_path.write_text(json.dumps(details or {}, ensure_ascii=False, indent=2), encoding="utf-8")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("model_name", ""))].append(row)

    extra_summary_fields = extra_summary_fields or []
    summary_rows = []
    for model_name, model_rows in sorted(grouped.items()):
        scored = [r for r in model_rows if r.get("status") == "scored" and r.get("score") not in ("", None)]
        scores = [float(r["score"]) for r in scored]
        item = {
            "model_name": model_name,
            "metric_name": metric_name,
            "mean_score": f"{mean(scores):.6f}" if scores else "",
            "num_scored": len(scored),
            "num_failed": sum(1 for r in model_rows if r.get("status") == "error"),
            "num_skipped": sum(1 for r in model_rows if r.get("status") != "scored"),
        }
        for field in extra_summary_fields:
            values = [float(r[field]) for r in scored if r.get(field) not in ("", None)]
            item[field] = f"{mean(values):.6f}" if values else ""
        summary_rows.append(item)

    summary_fields = [
        "model_name",
        "metric_name",
        "mean_score",
        "num_scored",
        "num_failed",
        "num_skipped",
        *extra_summary_fields,
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    ranked = sorted(
        [row for row in summary_rows if row["mean_score"] != ""],
        key=lambda row: float(row["mean_score"]),
        reverse=True,
    )
    lines = [
        f"# {metric_name.replace('_', ' ').title()} Ranking",
        "",
        "| Rank | Model | Mean Score | Scored | Failed | Skipped |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(ranked, start=1):
        lines.append(
            f"| {idx} | {row['model_name']} | {float(row['mean_score']):.4f} | "
            f"{row['num_scored']} | {row['num_failed']} | {row['num_skipped']} |"
        )
    ranking_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metric_dir

