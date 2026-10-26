#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUTS_DIR = SCRIPT_DIR / "outputs"
DEFAULT_LATEST_RUN = DEFAULT_OUTPUTS_DIR / "latest_run.txt"
DIMENSIONS = [
    "Visual Task Following",
    "Audio Task Following",
    "Speech Content Accuracy",
    "Temporal Order Following",
    "Instruction Following",
]
SKIPPED_DIMENSIONS_BY_MODEL = {
    "foley": {"Audio Task Following", "Speech Content Accuracy"},
    "omni": {"Audio Task Following", "Speech Content Accuracy"},
    "reference2v": {"Audio Task Following", "Speech Content Accuracy"},
    "skyreels-v3": {"Audio Task Following", "Speech Content Accuracy"},
    "skyreels-official-r2v": {"Audio Task Following", "Speech Content Accuracy"},
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {path}")
        return data
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def score_from(row: dict[str, Any], dim: str) -> float | None:
    if dim == "Instruction Following":
        values = [score_from(row, item) for item in DIMENSIONS if item != "Instruction Following"]
        values = [value for value in values if value is not None]
        return sum(values) / len(values) if values else None
    if dim in SKIPPED_DIMENSIONS_BY_MODEL.get(str(row.get("model_name", "")).lower(), set()):
        return None
    score = row.get("scores", {}).get(dim, {}).get("score")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def applicable_from(row: dict[str, Any], dim: str) -> bool:
    if dim in SKIPPED_DIMENSIONS_BY_MODEL.get(str(row.get("model_name", "")).lower(), set()):
        return False
    return bool(row.get("scores", {}).get(dim, {}).get("applicable"))


def board_from_sample(sample_id: str | None) -> str:
    return (sample_id or "unknown").split("_", 1)[0]


def add(bucket: dict[str, list[float]], dim: str, score: float | None) -> None:
    if score is not None:
        bucket[dim].append(score)


def collapse(bucket: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    out = {}
    for dim in DIMENSIONS:
        values = bucket.get(dim, [])
        out[dim] = {"mean_1_5": mean(values) if values else None, "count": len(values)}
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_board: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    overall: dict[str, list[float]] = defaultdict(list)
    failures = []

    for row in rows:
        if row.get("error"):
            failures.append(
                {
                    "task_id": row.get("task_id"),
                    "sample_id": row.get("sample_id"),
                    "model_name": row.get("model_name"),
                    "error": row.get("error"),
                }
            )
            continue
        model = row.get("model_name", "unknown")
        board = board_from_sample(row.get("sample_id"))
        for dim in DIMENSIONS:
            score = score_from(row, dim)
            add(overall, dim, score)
            add(by_model[model], dim, score)
            add(by_board[board], dim, score)

    return {
        "overall": collapse(overall),
        "by_model": {model: collapse(bucket) for model, bucket in sorted(by_model.items())},
        "by_board": {board: collapse(bucket) for board, bucket in sorted(by_board.items())},
        "total_rows": len(rows),
        "successful_rows": len([row for row in rows if not row.get("error")]),
        "failed_rows": len(failures),
        "failures": failures,
    }


def write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["group_type", "group", "dimension", "mean_1_5", "count"]
    out_rows = []
    for group_type, groups in [
        ("overall", {"overall": summary["overall"]}),
        ("model", summary["by_model"]),
        ("board", summary["by_board"]),
    ]:
        for group, metrics in groups.items():
            for dim, stats in metrics.items():
                out_rows.append(
                    {
                        "group_type": group_type,
                        "group": group,
                        "dimension": dim,
                        "mean_1_5": stats["mean_1_5"],
                        "count": stats["count"],
                    }
                )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def write_model_summary_csv(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_name"] + [f"{dim}_mean_1_5" for dim in DIMENSIONS] + [f"{dim}_count" for dim in DIMENSIONS]
    out_rows = []
    for model_name, metrics in summary["by_model"].items():
        row = {"model_name": model_name}
        for dim in DIMENSIONS:
            row[f"{dim}_mean_1_5"] = metrics.get(dim, {}).get("mean_1_5")
            row[f"{dim}_count"] = metrics.get(dim, {}).get("count")
        out_rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def write_per_sample_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["task_id", "sample_id", "model_name", "error"]
        + DIMENSIONS
        + [f"{dim}_applicable" for dim in DIMENSIONS if dim != "Instruction Following"]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {
                "task_id": row.get("task_id"),
                "sample_id": row.get("sample_id"),
                "model_name": row.get("model_name"),
                "error": row.get("error"),
            }
            for dim in DIMENSIONS:
                out[dim] = score_from(row, dim)
            for dim in DIMENSIONS:
                if dim != "Instruction Following":
                    out[f"{dim}_applicable"] = applicable_from(row, dim)
            writer.writerow(out)


def write_item_detail_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "sample_id",
        "model_name",
        "dimension",
        "question",
        "item_result",
        "item_score",
        "item_reason",
    ]
    detail_rows = []
    for row in rows:
        scores = row.get("scores", {})
        for dim in [
            "Visual Task Following",
            "Audio Task Following",
            "Speech Content Accuracy",
            "Temporal Order Following",
        ]:
            if dim in SKIPPED_DIMENSIONS_BY_MODEL.get(str(row.get("model_name", "")).lower(), set()):
                continue
            dim_score = scores.get(dim, {})
            for item in dim_score.get("item_results") or []:
                if not isinstance(item, dict):
                    continue
                detail_rows.append(
                    {
                        "task_id": row.get("task_id"),
                        "sample_id": row.get("sample_id"),
                        "model_name": row.get("model_name"),
                        "dimension": dim,
                        "question": item.get("question"),
                        "item_result": item.get("result"),
                        "item_score": item.get("score"),
                        "item_reason": item.get("reason"),
                    }
                )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)


def safe_folder_name(value: str | None) -> str:
    name = value or "unknown"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)


def write_sample_outputs(rows: list[dict[str, Any]], run_dir: Path) -> None:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[row.get("sample_id") or "unknown"].append(row)

    samples_dir = run_dir / "samples"
    for sample_id, sample_rows in sorted(by_sample.items()):
        sample_dir = samples_dir / safe_folder_name(sample_id)
        write_rows(sample_dir / "if_results.json", sample_rows)


def resolve_run_dir(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    if not DEFAULT_LATEST_RUN.exists():
        raise FileNotFoundError(f"Latest run file not found: {DEFAULT_LATEST_RUN}")
    return Path(DEFAULT_LATEST_RUN.read_text(encoding="utf-8").strip()).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Instruction Following eval results.")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument(
        "--output-prefix",
        default="if",
        help="Output file prefix. Default writes if_summary.csv/json; use another prefix for separate summaries.",
    )
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    results = run_dir / "if_results.json"
    if not results.exists():
        raise FileNotFoundError(f"Results file not found: {results}")
    rows = read_rows(results)
    summary = summarize(rows)
    prefix = safe_folder_name(args.output_prefix or "if")
    (run_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(summary, run_dir / f"{prefix}_summary.csv")
    write_model_summary_csv(summary, run_dir / f"{prefix}_model_summary.csv")
    write_per_sample_csv(rows, run_dir / f"{prefix}_per_sample.csv")
    write_item_detail_csv(rows, run_dir / f"{prefix}_item_detail.csv")
    if prefix == "if":
        write_sample_outputs(rows, run_dir)
    print(f"Saved summary outputs to {run_dir}")


if __name__ == "__main__":
    main()
