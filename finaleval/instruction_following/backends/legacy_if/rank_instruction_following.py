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
DIMENSIONS = [
    "Instruction Following",
    "Visual Task Following",
    "Audio Task Following",
    "Speech Content Accuracy",
    "Temporal Order Following",
]
FINAL_VISUAL_COLUMNS = ["Important Action", "Wearing Requirement", "Other Detail"]


def latest_run_dir() -> Path:
    candidates = sorted(path for path in DEFAULT_OUTPUTS_DIR.glob("if_*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No IF output run found under {DEFAULT_OUTPUTS_DIR}")
    return candidates[-1].resolve()


def score_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_type_map(input_dir: Path) -> dict[str, str]:
    mapping = {}
    for path in sorted(input_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sample_id = data.get("id") or path.stem
        text = json.dumps(data, ensure_ascii=False).lower()
        mapping[str(sample_id)] = "cartoon" if "cartoon" in text else "real"
    return mapping


def add_grouped(rows: list[dict[str, Any]], keys: list[str], metric: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    totals: dict[tuple[Any, ...], int] = defaultdict(int)
    for row in rows:
        group_key = tuple(row.get(key) for key in keys)
        totals[group_key] += 1
        value = score_float(row.get(metric))
        if value is None:
            continue
        groups[group_key].append(value)
    out = []
    for group_key, values in sorted(groups.items()):
        item = {key: value for key, value in zip(keys, group_key)}
        item["mean_1_5"] = round(sum(values) / len(values), 4)
        item["scored_count"] = len(values)
        item["total_count"] = totals[group_key]
        out.append(item)
    return sorted(
        out,
        key=lambda item: (
            -item["mean_1_5"],
            -item["scored_count"],
            str(item.get("model_name", "")),
        ),
    )


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


def append_final_visual_report(lines: list[str], run_dir: Path) -> None:
    final_per_sample = run_dir / "visual_final_per_sample.csv"
    if not final_per_sample.exists():
        raise FileNotFoundError(
            f"Final visual checklist CSV not found: {final_per_sample}. "
            "Run score_visual_checklist_final.py before rank_instruction_following.py for visual-only runs."
        )

    rows = read_csv(final_per_sample)
    lines.append("\n## Final Visual Checklist Ranking")
    lines.append(
        "\nScoring: DROP questions are excluded; final score uses "
        "`0.5 * Important Action + 0.3 * Wearing Requirement + 0.2 * Other Detail`, "
        "renormalized when a sample has missing categories."
    )

    grouped = add_grouped(rows, ["model_name"], "Visual Task Following")
    by_model_cat: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for category in FINAL_VISUAL_COLUMNS:
        for row in add_grouped(rows, ["model_name"], category):
            by_model_cat[str(row["model_name"])][category] = row

    lines.append("\n### Visual Final\n")
    lines.append(
        markdown_table(
            ["Rank", "model_name", "Mean 1-5", *FINAL_VISUAL_COLUMNS, "Scored Count", "Total Samples"],
            [
                [
                    rank,
                    row["model_name"],
                    score_text(row.get("mean_1_5")),
                    *[
                        score_text(by_model_cat.get(str(row["model_name"]), {}).get(category, {}).get("mean_1_5"))
                        for category in FINAL_VISUAL_COLUMNS
                    ],
                    row.get("scored_count"),
                    row.get("total_count"),
                ]
                for rank, row in enumerate(grouped, start=1)
            ],
        )
    )

    for category in FINAL_VISUAL_COLUMNS:
        lines.append(f"\n### {category}\n")
        lines.append(
            markdown_table(
                ["Rank", "model_name", "Mean 1-5", "Scored Count", "Total Samples"],
                [
                    [
                        rank,
                        row["model_name"],
                        score_text(row.get("mean_1_5")),
                        row.get("scored_count"),
                        row.get("total_count"),
                    ]
                    for rank, row in enumerate(add_grouped(rows, ["model_name"], category), start=1)
                ],
            )
        )


def append_ranking(lines: list[str], title: str, rows: list[dict[str, Any]], group_keys: list[str], metric: str) -> None:
    ranking = add_grouped(rows, group_keys, metric)
    lines.append(f"\n### {title}")
    if not ranking:
        lines.append("\nNo valid scores.")
        return
    table_rows = []
    for rank, row in enumerate(ranking, start=1):
        table_rows.append(
            [rank]
            + [row.get(key) for key in group_keys]
            + [
                score_text(row.get("mean_1_5")),
                row.get("scored_count"),
                row.get("total_count"),
            ]
        )
    lines.append("")
    lines.append(
        markdown_table(
            ["Rank"] + group_keys + ["Mean 1-5", "Scored Count", "Total Samples"],
            table_rows,
        )
    )


def append_group_sections(lines: list[str], rows: list[dict[str, Any]], group_key: str, section_title: str, metrics: list[str]) -> None:
    groups = sorted({row.get(group_key) for row in rows if row.get(group_key)})
    lines.append(f"\n## {section_title}")
    if not groups:
        lines.append("\nNo valid groups.")
        return
    for group in groups:
        group_rows = [row for row in rows if row.get(group_key) == group]
        lines.append(f"\n### {group}")
        for metric in metrics:
            append_ranking(lines, metric, group_rows, ["model_name"], metric)


def metric_has_scores(rows: list[dict[str, Any]], metric: str) -> bool:
    return any(score_float(row.get(metric)) is not None for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create IF ranking markdown report.")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else latest_run_dir()
    output = Path(args.output).expanduser().resolve() if args.output else run_dir / "if_ranking.md"
    item_detail = run_dir / "if_item_detail.csv"
    include_type_sections = True
    if item_detail.exists():
        item_rows = read_csv(item_detail)
        dimensions = {str(row.get("dimension") or "") for row in item_rows if row.get("dimension")}
        if item_rows and dimensions == {"Visual Task Following"}:
            lines = [f"# Instruction Following Ranking\n\nRun: `{run_dir}`"]
            append_final_visual_report(lines, run_dir)
            output.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
            print(f"Saved IF final visual checklist ranking report: {output}")
            return
        if item_rows:
            include_type_sections = False

    per_sample = run_dir / "if_per_sample.csv"
    if not per_sample.exists():
        raise FileNotFoundError(f"IF per-sample CSV not found: {per_sample}")
    rows: list[dict[str, Any]] = read_csv(per_sample)
    types = sample_type_map(Path(args.input_dir).expanduser().resolve())
    for row in rows:
        sample_id = row.get("sample_id", "")
        row["source_board"] = sample_id.split("_", 1)[0] if sample_id else "unknown"
        row["sample_type"] = types.get(sample_id, "unknown")

    lines = [f"# Instruction Following Ranking\n\nRun: `{run_dir}`"]
    metrics_to_show = [
        metric
        for metric in DIMENSIONS
        if metric == "Instruction Following" or metric_has_scores(rows, metric)
    ]
    lines.append("\n## Overall")
    for metric in metrics_to_show:
        append_ranking(lines, metric, rows, ["model_name"], metric)
    append_group_sections(lines, rows, "source_board", "By Board", metrics_to_show)
    if include_type_sections:
        append_group_sections(lines, rows, "sample_type", "By Type", metrics_to_show)

    output.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"Saved IF ranking report: {output}")


if __name__ == "__main__":
    main()
