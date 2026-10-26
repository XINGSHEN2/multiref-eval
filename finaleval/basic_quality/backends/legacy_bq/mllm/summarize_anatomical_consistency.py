#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from eval_anatomical_consistency import (
    DEFAULT_OUTPUTS_DIR,
    DIMENSIONS,
    OVERALL_DIMENSION,
    read_rows,
)


ALL_DIMENSIONS = [*DIMENSIONS, OVERALL_DIMENSION]


def parse_score(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def score_from(row: dict[str, Any], dimension: str) -> float | None:
    return parse_score(row.get("scores", {}).get(dimension, {}).get("score"))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    overall: dict[str, list[float]] = defaultdict(list)
    failures = []
    for row in rows:
        if row.get("error"):
            failures.append({"task_id": row.get("task_id"), "model_name": row.get("model_name"), "error": row.get("error")})
            continue
        for dimension in ALL_DIMENSIONS:
            score = score_from(row, dimension)
            if score is None:
                continue
            overall[dimension].append(score)
            by_model[str(row.get("model_name", "unknown"))][dimension].append(score)

    def collapse(bucket: dict[str, list[float]]) -> dict[str, Any]:
        return {
            dimension: {"mean_1_5": mean(bucket.get(dimension, [])) if bucket.get(dimension) else None, "count": len(bucket.get(dimension, []))}
            for dimension in ALL_DIMENSIONS
        }

    return {
        "scoring": "raw judge scores; no strict audit/rejudge",
        "overall": collapse(overall),
        "by_model": {model: collapse(bucket) for model, bucket in sorted(by_model.items())},
        "total_rows": len(rows),
        "successful_rows": len(rows) - len(failures),
        "failed_rows": len(failures),
        "failures": failures,
    }


def rank_rows(summary: dict[str, Any], dimension: str) -> list[tuple[int, str, float, int]]:
    values = []
    for model, metrics in summary["by_model"].items():
        metric = metrics.get(dimension, {})
        score = metric.get("mean_1_5")
        count = metric.get("count", 0)
        if score is None:
            continue
        values.append((model, float(score), int(count or 0)))
    values.sort(key=lambda item: (-item[1], item[0]))
    ranked = []
    previous_score = None
    previous_rank = 0
    for idx, (model, score, count) in enumerate(values, start=1):
        rank = previous_rank if previous_score is not None and score == previous_score else idx
        ranked.append((rank, model, score, count))
        previous_score = score
        previous_rank = rank
    return ranked


def write_ranking_md(summary: dict[str, Any], path: Path, run_dir: Path) -> None:
    lines = [
        "# Anatomical Consistency Ranking",
        "",
        "Scoring: raw judge scores; no strict audit/rejudge.",
        "",
        f"Run directory: `{run_dir}`",
        "",
        f"Total rows: {summary.get('total_rows', 0)}",
        f"Successful rows: {summary.get('successful_rows', 0)}",
        f"Failed rows: {summary.get('failed_rows', 0)}",
        "",
    ]
    for title, dimension in [("Overall Ranking", OVERALL_DIMENSION), *[(d, d) for d in DIMENSIONS]]:
        lines.extend([f"## {title}", "", "| Rank | Model | Score | Count |", "|---:|---|---:|---:|"])
        for rank, model, score, count in rank_rows(summary, dimension):
            lines.append(f"| {rank} | {model} | {score:.4f} | {count} |")
        lines.append("")
    failures = summary.get("failures") or []
    if failures:
        lines.extend(["## Failures", "", "| Task | Model | Error |", "|---|---|---|"])
        for row in failures[:100]:
            error = str(row.get("error", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row.get('task_id')} | {row.get('model_name')} | {error} |")
        if len(failures) > 100:
            lines.append(f"| ... | ... | {len(failures) - 100} more failures omitted |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_model_summary(summary: dict[str, Any], path: Path) -> None:
    fields = ["model_name"]
    fields += [f"{dimension}_mean_1_5" for dimension in ALL_DIMENSIONS]
    fields += [f"{dimension}_count" for dimension in ALL_DIMENSIONS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, metrics in summary["by_model"].items():
            row = {"model_name": model}
            for dimension in ALL_DIMENSIONS:
                row[f"{dimension}_mean_1_5"] = metrics[dimension]["mean_1_5"]
                row[f"{dimension}_count"] = metrics[dimension]["count"]
            writer.writerow(row)


def write_per_sample(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["task_id", "sample_id", "model_name", "video_path", "error", *ALL_DIMENSIONS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {field: row.get(field) for field in fields[:5]}
            for dimension in ALL_DIMENSIONS:
                output[dimension] = score_from(row, dimension)
            writer.writerow(output)


def write_detail(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["task_id", "sample_id", "model_name", "video_path", "dimension", "score", "reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for dimension in ALL_DIMENSIONS:
                metric = row.get("scores", {}).get(dimension, {})
                writer.writerow(
                    {
                        "task_id": row.get("task_id"),
                        "sample_id": row.get("sample_id"),
                        "model_name": row.get("model_name"),
                        "video_path": row.get("video_path"),
                        "dimension": dimension,
                        "score": metric.get("score"),
                        "reason": metric.get("reason"),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize anatomical consistency results using raw judge scores.")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        latest = DEFAULT_OUTPUTS_DIR / "latest_run.txt"
        run_dir = Path(latest.read_text(encoding="utf-8").strip()).resolve()

    rows = read_rows(run_dir / "anatomical_results.json")
    summary = summarize(rows)
    (run_dir / "anatomical_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_model_summary(summary, run_dir / "anatomical_model_summary.csv")
    write_per_sample(rows, run_dir / "anatomical_per_sample.csv")
    write_detail(rows, run_dir / "anatomical_item_detail.csv")
    write_ranking_md(summary, run_dir / "anatomical_ranking.md", run_dir)
    print(f"Saved anatomical summary outputs to {run_dir}")
    print(f"Saved ranking md to {run_dir / 'anatomical_ranking.md'}")


if __name__ == "__main__":
    main()
