#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank models from VSLS result JSON files.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--id-column", default="index")
    parser.add_argument("--model-column", default="model_name")
    parser.add_argument("--min-scored-videos", type=int, default=5)
    return parser.parse_args()


def load_metadata(path: Path | None, id_column: str) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if id_column not in (reader.fieldnames or []):
            return {}
        return {row[id_column]: row for row in reader}


def infer_model(item_id: str, result: dict[str, Any], metadata: dict[str, dict[str, str]], model_column: str) -> str:
    if item_id in metadata and metadata[item_id].get(model_column):
        return metadata[item_id][model_column]
    if result.get("model_name"):
        return str(result["model_name"])
    if "::" in item_id:
        return item_id.rsplit("::", 1)[-1]
    if "/" in item_id:
        return item_id.split("/", 1)[0]
    return item_id.split("_", 1)[0]


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    metadata = load_metadata(args.metadata_csv, args.id_column)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for path in sorted(results_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        item_id = path.stem
        model = infer_model(item_id, result, metadata, args.model_column)
        groups[model].append(result)

    rows = []
    for model, results in groups.items():
        scored = [row for row in results if row.get("video_score_0_1") is not None]
        scores = [float(row["video_score_0_1"]) for row in scored]
        mos_values = [float(row["video_mos_1_5"]) for row in scored]
        ratios = [
            float(row.get("scored_speech_ratio") or 0.0)
            for row in results
            if row.get("video_status") != "error"
        ]
        status_counts: dict[str, int] = defaultdict(int)
        for result in results:
            for status, count in result.get("status_counts", {}).items():
                status_counts[status] += int(count)
        rows.append(
            {
                "model": model,
                "mean_score": mean(scores) if scores else None,
                "mean_mos": mean(mos_values) if mos_values else None,
                "mean_coverage": mean(ratios) if ratios else 0.0,
                "scored_videos": len(scored),
                "total_videos": len(results),
                "scored_windows": status_counts["scored"],
                "offscreen_windows": status_counts["offscreen_voice"],
                "no_active_speaker_windows": status_counts["no_active_speaker"],
                "insufficient_track_windows": status_counts["insufficient_track"],
                "error_videos": sum(result.get("video_status") == "error" for result in results),
            }
        )

    rows.sort(
        key=lambda row: (
            row["mean_score"] is None,
            -(row["mean_score"] or 0.0),
            -row["mean_coverage"],
            row["model"],
        )
    )

    lines = [
        "# VSLS Model Ranking",
        "",
        f"- Results: `{results_dir}`",
        f"- Minimum support warning threshold: {args.min_scored_videos} scored videos",
        "- Mean score must be interpreted together with mean scored-speech coverage.",
        "",
        "| Rank | Model | Mean VSLS | Mean MOS | Mean Coverage | Scored Videos | Total Videos | Scored Windows | Offscreen | No Active Speaker | Insufficient Track | Errors | Support |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rank = 0
    previous_score = None
    for index, row in enumerate(rows, start=1):
        if row["mean_score"] is None:
            display_rank = "N/A"
        else:
            if previous_score is None or row["mean_score"] != previous_score:
                rank = index
                previous_score = row["mean_score"]
            display_rank = str(rank)
        score = f"{row['mean_score']:.4f}" if row["mean_score"] is not None else "N/A"
        mos = f"{row['mean_mos']:.3f}" if row["mean_mos"] is not None else "N/A"
        support = "ok" if row["scored_videos"] >= args.min_scored_videos else "low_support"
        lines.append(
            f"| {display_rank} | {row['model']} | {score} | {mos} | "
            f"{row['mean_coverage']:.3f} | {row['scored_videos']} | {row['total_videos']} | "
            f"{row['scored_windows']} | {row['offscreen_windows']} | "
            f"{row['no_active_speaker_windows']} | {row['insufficient_track_windows']} | "
            f"{row['error_videos']} | {support} |"
        )

    output = args.output_md.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved VSLS ranking to {output}")


if __name__ == "__main__":
    main()
