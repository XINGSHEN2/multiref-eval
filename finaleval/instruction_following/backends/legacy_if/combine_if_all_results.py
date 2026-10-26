#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any


DIMENSIONS = [
    "Visual Task Following",
    "Audio Task Following",
    "Temporal Order Following",
    "Speech Content Accuracy",
]
VISUAL_DETAIL_COLUMNS = ["Important Action", "Wearing Requirement", "Other Detail"]

WEIGHTS = {
    "Visual Task Following": 0.3,
    "Audio Task Following": 0.3,
    "Speech Content Accuracy": 0.2,
    "Temporal Order Following": 0.2,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def score(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    if 1 <= val <= 5:
        return val
    return None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def task_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("task_id", "")),
        str(row.get("sample_id", "")),
        str(row.get("model_name", "")),
    )


def board_from_sample(sample_id: str) -> str:
    return sample_id.split("_", 1)[0] if "_" in sample_id else ""


def parse_csv_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def weighted_score(values: dict[str, float | None]) -> float | None:
    present = [(dim, val) for dim, val in values.items() if val is not None]
    if not present:
        return None
    weight_sum = sum(WEIGHTS[dim] for dim, _ in present)
    if weight_sum <= 0:
        return None
    return sum(WEIGHTS[dim] * float(val) for dim, val in present) / weight_sum


def empty_record(key: tuple[str, str, str]) -> dict[str, Any]:
    return {
        "task_id": key[0],
        "sample_id": key[1],
        "board": board_from_sample(key[1]),
        "model_name": key[2],
        "error": "",
        **{dim: None for dim in DIMENSIONS},
        **{column: None for column in VISUAL_DETAIL_COLUMNS},
    }


def combine_per_sample(if_rows: list[dict[str, str]], speech_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in if_rows:
        key = task_key(row)
        record = records.setdefault(key, empty_record(key))
        for dim in ["Visual Task Following", "Audio Task Following", "Temporal Order Following"]:
            value = score(row.get(dim))
            if value is not None:
                record[dim] = value
        for column in VISUAL_DETAIL_COLUMNS:
            value = score(row.get(column))
            if value is not None:
                record[column] = value
        if row.get("error"):
            record["error"] = row.get("error")

    for row in speech_rows:
        key = task_key(row)
        record = records.setdefault(key, empty_record(key))
        value = score(row.get("Speech Content Accuracy"))
        if value is not None:
            record["Speech Content Accuracy"] = value
        if row.get("error"):
            record["error"] = row.get("error")

    out = []
    for key in sorted(records):
        record = records[key]
        values = {dim: record.get(dim) for dim in DIMENSIONS}
        out.append(
            {
                "task_id": record["task_id"],
                "sample_id": record["sample_id"],
                "board": record["board"],
                "model_name": record["model_name"],
                "error": record.get("error") or "",
                **{dim: fmt(score(record.get(dim))) for dim in DIMENSIONS},
                **{column: fmt(score(record.get(column))) for column in VISUAL_DETAIL_COLUMNS},
                "Instruction Following": fmt(weighted_score(values)),
            }
        )
    return out


def grouped_scores(rows: list[dict[str, Any]], group_key: str | None = None) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row[group_key]) if group_key else "overall"
        buckets.setdefault(key, []).append(row)
    out = []
    for group, group_rows in buckets.items():
        result: dict[str, Any] = {"group": group, "count": len(group_rows)}
        dim_means: dict[str, float | None] = {}
        for dim in DIMENSIONS:
            vals = [score(row.get(dim)) for row in group_rows]
            vals = [val for val in vals if val is not None]
            dim_means[dim] = mean(vals) if vals else None
            result[dim] = dim_means[dim]
            result[f"{dim}_count"] = len(vals)
        for column in VISUAL_DETAIL_COLUMNS:
            vals = [score(row.get(column)) for row in group_rows]
            vals = [val for val in vals if val is not None]
            result[column] = mean(vals) if vals else None
            result[f"{column}_count"] = len(vals)
        result["Instruction Following"] = weighted_score(dim_means)
        result["Instruction Following_count"] = len(group_rows)
        out.append(result)
    return out


def write_md(
    path: Path,
    ranking: list[dict[str, Any]],
    board_ranking: list[dict[str, Any]],
    if_run_dirs: list[Path],
    speech_run_dirs: list[Path],
) -> None:
    lines = [
        "# IF 4-Task Weighted Ranking",
        "",
        "Weighting: `Visual=0.3`, `Audio=0.3`, `Speech=0.2`, `Temporal=0.2`.",
        "",
        "Combined score = `0.3*Visual + 0.3*Audio + 0.2*Speech + 0.2*Temporal`.",
        "",
        "If a dimension is unavailable for a row, the available weights are renormalized for that per-sample row. Model and board rankings are computed from dimension means and the same weights.",
        "",
        "## Source Run Directories",
        "",
        "### IF",
        *[f"- `{path}`" for path in if_run_dirs],
        "",
        "### Speech",
        *[f"- `{path}`" for path in speech_run_dirs],
        "",
        "## 总排名",
        "",
        "| Rank | Model | IF All | Visual | Important Action | Wearing Requirement | Other Detail | Audio | Temporal | Speech | Count |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranking:
        lines.append(
            "| {rank} | {model_name} | {if_all} | {visual} | {action} | {wearing} | {detail} | {audio} | {temporal} | {speech} | {count} |".format(
                rank=row["rank"],
                model_name=row["model_name"],
                if_all=fmt(row["Instruction Following"]),
                visual=fmt(row["Visual Task Following"]),
                action=fmt(row["Important Action"]),
                wearing=fmt(row["Wearing Requirement"]),
                detail=fmt(row["Other Detail"]),
                audio=fmt(row["Audio Task Following"]),
                temporal=fmt(row["Temporal Order Following"]),
                speech=fmt(row["Speech Content Accuracy"]),
                count=row["count"],
            )
        )
    if board_ranking:
        lines.extend(["", "## 每个 Board 综合排名"])
        for board in sorted({str(row["board"]) for row in board_ranking}):
            lines.extend(
                [
                    "",
                    f"### {board}",
                    "| Rank | Model | IF All | Visual | Important Action | Wearing Requirement | Other Detail | Audio | Temporal | Speech | Count |",
                    "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in [item for item in board_ranking if item["board"] == board]:
                lines.append(
                    "| {rank} | {model_name} | {if_all} | {visual} | {action} | {wearing} | {detail} | {audio} | {temporal} | {speech} | {count} |".format(
                        rank=row["rank"],
                        model_name=row["model_name"],
                        if_all=fmt(row["Instruction Following"]),
                        visual=fmt(row["Visual Task Following"]),
                        action=fmt(row["Important Action"]),
                        wearing=fmt(row["Wearing Requirement"]),
                        detail=fmt(row["Other Detail"]),
                        audio=fmt(row["Audio Task Following"]),
                        temporal=fmt(row["Temporal Order Following"]),
                        speech=fmt(row["Speech Content Accuracy"]),
                        count=row["count"],
                    )
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine IF visual/audio/temporal outputs with speech outputs.")
    parser.add_argument("--if-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--speech-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefix", default="if_all")
    parser.add_argument("--ranking-basename", default=None)
    parser.add_argument("--boards", default=None, help="Comma-separated boards to include, e.g. board1,board2.")
    args = parser.parse_args()

    if_run_dirs = [path.expanduser().resolve() for path in args.if_run_dir]
    speech_run_dirs = [path.expanduser().resolve() for path in args.speech_run_dir]
    if not if_run_dirs and not speech_run_dirs:
        raise ValueError("Provide at least one --if-run-dir or --speech-run-dir.")
    default_output_dir = if_run_dirs[0] if if_run_dirs else speech_run_dirs[0]
    output_dir = (args.output_dir or default_output_dir).expanduser().resolve()

    if_rows: list[dict[str, str]] = []
    speech_rows: list[dict[str, str]] = []
    for if_run_dir in if_run_dirs:
        visual_final = if_run_dir / "visual_final_per_sample.csv"
        if visual_final.exists():
            if_rows.extend(read_csv(visual_final))
        else:
            if_rows.extend(read_csv(if_run_dir / "if_per_sample.csv"))
    for speech_run_dir in speech_run_dirs:
        speech_rows.extend(read_csv(speech_run_dir / "speech_per_sample.csv"))
    per_sample = combine_per_sample(if_rows, speech_rows)
    selected_boards = parse_csv_set(args.boards)
    if selected_boards:
        per_sample = [row for row in per_sample if str(row.get("board", "")) in selected_boards]
        if not per_sample:
            raise ValueError(f"No per-sample rows match --boards {args.boards!r}.")

    model_rows = grouped_scores(per_sample, "model_name")
    model_rows.sort(key=lambda row: (row["Instruction Following"] is None, -(row["Instruction Following"] or -1)))
    ranking = []
    for rank, row in enumerate(model_rows, start=1):
        ranking.append({"rank": rank, "model_name": row["group"], **row})

    board_ranking = []
    for board in sorted({str(row.get("board", "")) for row in per_sample if row.get("board")}):
        rows_for_board = [row for row in per_sample if row.get("board") == board]
        board_rows = grouped_scores(rows_for_board, "model_name")
        board_rows.sort(key=lambda row: (row["Instruction Following"] is None, -(row["Instruction Following"] or -1)))
        for rank, row in enumerate(board_rows, start=1):
            board_ranking.append({"board": board, "rank": rank, "model_name": row["group"], **row})

    ranking_fields = [
        "rank",
        "model_name",
        "Instruction Following",
        "Visual Task Following",
        "Important Action",
        "Wearing Requirement",
        "Other Detail",
        "Audio Task Following",
        "Temporal Order Following",
        "Speech Content Accuracy",
        "count",
    ]
    per_sample_fields = [
        "task_id",
        "sample_id",
        "model_name",
        "error",
        "board",
        "Visual Task Following",
        "Important Action",
        "Wearing Requirement",
        "Other Detail",
        "Audio Task Following",
        "Temporal Order Following",
        "Speech Content Accuracy",
        "Instruction Following",
    ]
    ranking_basename = args.ranking_basename or f"{args.prefix}_ranking"
    write_csv(output_dir / f"{args.prefix}_per_sample.csv", per_sample, per_sample_fields)
    write_csv(output_dir / f"{ranking_basename}.csv", ranking, ranking_fields)
    write_md(output_dir / f"{ranking_basename}.md", ranking, board_ranking, if_run_dirs, speech_run_dirs)
    print(f"Wrote {output_dir / f'{ranking_basename}.md'}")
    print(f"Wrote {output_dir / f'{ranking_basename}.csv'}")
    print(f"Wrote {output_dir / f'{args.prefix}_per_sample.csv'}")


if __name__ == "__main__":
    main()
