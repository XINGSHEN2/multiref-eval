#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUTS_DIR = SCRIPT_DIR / "outputs"
DEFAULT_INPUT_DIR = SCRIPT_DIR / "inputs" / "metadata"
TOTAL_METRIC = "Audio Consistency"


def score_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sample_type_map(input_dir: Path) -> dict[str, str]:
    mapping = {}
    for path in sorted(input_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sample_id = str(data.get("id") or path.stem)
        text = json.dumps(data, ensure_ascii=False).lower()
        mapping[sample_id] = "cartoon" if "cartoon" in text else "real"
    return mapping


def board_from_sample(sample_id: str | None) -> str:
    return (sample_id or "unknown").split("_", 1)[0]


def metric_from_csv_column(column: str) -> str | None:
    suffix = "_score"
    if not column.endswith(suffix):
        return None
    return column[: -len(suffix)]


def read_csv_rows(path: Path, types: dict[str, str]) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        metrics = [metric for column in (reader.fieldnames or []) if (metric := metric_from_csv_column(column))]
        for row in reader:
            sample_id = row.get("sample_id") or ""
            out: dict[str, Any] = {
                "task_id": row.get("task_id"),
                "sample_id": sample_id,
                "source_board": row.get("source_board") or board_from_sample(sample_id),
                "sample_type": types.get(sample_id, "unknown"),
                "model_name": row.get("model_name"),
                "video_path": row.get("video_path"),
                "error": row.get("error"),
            }
            for metric in metrics:
                out[metric] = score_float(row.get(f"{metric}_score"))
            rows.append(out)
    return rows, metrics


def read_json_rows(path: Path, types: dict[str, str]) -> tuple[list[dict[str, Any]], list[str]]:
    raw_rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list):
        raise ValueError(f"Expected JSON array in {path}")
    metric_names: set[str] = set()
    rows = []
    for row in raw_rows:
        scores = row.get("scores") or {}
        sample_id = row.get("sample_id") or ""
        out: dict[str, Any] = {
            "task_id": row.get("task_id"),
            "sample_id": sample_id,
            "source_board": row.get("source_board") or board_from_sample(sample_id),
            "sample_type": types.get(sample_id, "unknown"),
            "model_name": row.get("model_name"),
            "video_path": row.get("video_path"),
            "error": row.get("error"),
        }
        for metric, payload in scores.items():
            if not isinstance(payload, dict):
                continue
            metric_names.add(metric)
            out[metric] = score_float(payload.get("score"))
        rows.append(out)
    return rows, sorted(metric_names)


def load_run_rows(run_dir: Path, types: dict[str, str]) -> tuple[list[dict[str, Any]], list[str], str]:
    csv_path = run_dir / "ac_per_sample.csv"
    if csv_path.exists():
        rows, metrics = read_csv_rows(csv_path, types)
        return rows, metrics, csv_path.name
    json_path = run_dir / "ac_results.json"
    if json_path.exists():
        rows, metrics = read_json_rows(json_path, types)
        return rows, metrics, json_path.name
    raise FileNotFoundError(f"No ac_per_sample.csv or ac_results.json found in {run_dir}")


def grouped_summary(rows: list[dict[str, Any]], group_keys: list[str], metric: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        score = score_float(row.get(metric))
        if score is None:
            continue
        groups[tuple(row.get(key) for key in group_keys)].append(score)
    out = []
    for group, values in sorted(groups.items()):
        item = {key: value for key, value in zip(group_keys, group)}
        item["mean_1_5"] = round(sum(values) / len(values), 4)
        item["count"] = len(values)
        out.append(item)
    return sorted(out, key=lambda item: (-item["mean_1_5"], -item["count"], str(item.get("model_name", ""))))


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def score_text(value: Any) -> str:
    score = score_float(value)
    return "" if score is None else f"{score:.4f}"


def append_ranking(lines: list[str], title: str, rows: list[dict[str, Any]], group_keys: list[str], metric: str) -> None:
    ranking = grouped_summary(rows, group_keys, metric)
    lines.append(f"\n### {title}")
    if not ranking:
        lines.append("\nNo valid scores.")
        return
    table_rows = []
    for rank, row in enumerate(ranking, start=1):
        table_rows.append(
            [rank] + [row.get(key) for key in group_keys] + [score_text(row.get("mean_1_5")), row.get("count")]
        )
    lines.append("")
    lines.append(markdown_table(["Rank"] + group_keys + ["Mean 1-5", "Count"], table_rows))


def metric_order(metrics: list[str]) -> list[str]:
    unique = []
    for metric in [TOTAL_METRIC, *metrics]:
        if metric in metrics and metric not in unique:
            unique.append(metric)
    for metric in metrics:
        if metric != TOTAL_METRIC and metric not in unique:
            unique.append(metric)
    return unique


def append_group_sections(lines: list[str], rows: list[dict[str, Any]], metrics: list[str], group_key: str, title: str) -> None:
    groups = sorted({row.get(group_key) for row in rows if row.get(group_key)})
    lines.append(f"\n## {title}")
    if not groups:
        lines.append("\nNo valid groups.")
        return
    for group in groups:
        group_rows = [row for row in rows if row.get(group_key) == group]
        lines.append(f"\n### {group}")
        for metric in metric_order(metrics):
            append_ranking(lines, metric, group_rows, ["model_name"], metric)


def write_ranking(run_dir: Path, input_dir: Path, output: Path | None = None) -> Path:
    types = sample_type_map(input_dir)
    rows, metrics, source_name = load_run_rows(run_dir, types)
    metrics = metric_order(metrics)
    output_path = output or run_dir / "ac_ranking.md"
    lines = [
        "# Audio Consistency Ranking",
        "",
        f"Run: `{run_dir}`",
        f"Source: `{source_name}`",
        f"Rows: {len(rows)}",
    ]

    lines.append("\n## Overall")
    for metric in metrics:
        append_ranking(lines, metric, rows, ["model_name"], metric)
    append_group_sections(lines, rows, metrics, "source_board", "By Board")
    append_group_sections(lines, rows, metrics, "sample_type", "By Type")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return output_path


def iter_run_dirs(outputs_dir: Path) -> list[Path]:
    return sorted(path for path in outputs_dir.iterdir() if path.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Audio Consistency ranking reports.")
    parser.add_argument("--run-dir", default=None, help="Specific AC output run directory.")
    parser.add_argument("--all", action="store_true", help="Generate ranking for every run under --outputs-dir.")
    parser.add_argument("--outputs-dir", default=str(DEFAULT_OUTPUTS_DIR))
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output", default=None, help="Output path for a single run.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if args.all:
        outputs_dir = Path(args.outputs_dir).expanduser().resolve()
        written = []
        for run_dir in iter_run_dirs(outputs_dir):
            try:
                written.append(write_ranking(run_dir, input_dir))
            except FileNotFoundError as exc:
                print(f"[skip] {run_dir}: {exc}")
        for path in written:
            print(f"Saved AC ranking report: {path}")
        return

    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else Path(DEFAULT_OUTPUTS_DIR).resolve()
    output = Path(args.output).expanduser().resolve() if args.output else None
    print(f"Saved AC ranking report: {write_ranking(run_dir, input_dir, output)}")


if __name__ == "__main__":
    main()
