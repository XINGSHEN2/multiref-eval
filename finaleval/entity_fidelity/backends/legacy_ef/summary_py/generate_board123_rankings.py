#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


FINAL_EF_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = FINAL_EF_ROOT / "outputs" / "xlsx_entity_fidelity"
DEFAULT_OUTPUT_DIR = FINAL_EF_ROOT / "outputs" / "board123_combined_rankings"
BOARD_RE = re.compile(r"(board[123])")


def parse_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def board_from_row(row: dict[str, Any]) -> str:
    for key in ("sample_id", "sample", "video_name", "result_json"):
        match = BOARD_RE.search(str(row.get(key, "")))
        if match:
            return match.group(1)
    return "unknown"


def row_other_score(row: dict[str, Any]) -> float | None:
    values = [
        parse_float(row.get("Object_EF_with_LTC")),
        parse_float(row.get("Scene_EF_with_LTC")),
    ]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def row_board_total(row: dict[str, Any]) -> float | None:
    human = parse_float(row.get("Human_EF_with_LTC"))
    other = row_other_score(row)
    if human is None:
        return None
    if other is None:
        return human
    return 0.7 * human + 0.3 * other


def read_summary_rows(summary_csvs: list[Path]) -> list[dict[str, Any]]:
    latest_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for source_index, path in enumerate(summary_csvs):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row_index, row in enumerate(csv.DictReader(handle)):
                sample_id = str(row.get("sample_id") or row.get("sample") or "").strip()
                model = str(row.get("model") or row.get("model_name") or "").strip()
                if not sample_id or not model:
                    continue
                row["_source_csv"] = str(path)
                row["_source_index"] = source_index
                row["_row_index"] = row_index
                row["_board"] = board_from_row(row)
                latest_by_key[(sample_id, model)] = row
    return list(latest_by_key.values())


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def rank_rows(rows: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (-(row[score_key] or float("-inf")), str(row["model"])))
    ranked = []
    prev_score = None
    prev_rank = 0
    for index, row in enumerate(ordered, start=1):
        score = row[score_key]
        rank = prev_rank if prev_score is not None and score == prev_score else index
        prev_score = score
        prev_rank = rank
        ranked.append({"rank": rank, **row})
    return ranked


def aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_board_model: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    row_counts: dict[tuple[str, str], int] = defaultdict(int)

    for row in rows:
        board = str(row.get("_board") or "unknown")
        model = str(row.get("model") or row.get("model_name") or "").strip()
        if board not in {"board1", "board2", "board3"} or not model:
            continue
        key = (board, model)
        row_counts[key] += 1
        values = {
            "human": parse_float(row.get("Human_EF_with_LTC")),
            "object": parse_float(row.get("Object_EF_with_LTC")),
            "background": parse_float(row.get("Scene_EF_with_LTC")),
            "other": row_other_score(row),
            "total": row_board_total(row),
        }
        for name, value in values.items():
            if value is not None:
                by_board_model[key][name].append(value)

    board_rankings: dict[str, list[dict[str, Any]]] = {}
    for board in ("board1", "board2", "board3"):
        board_rows = []
        for (row_board, model), metrics in by_board_model.items():
            if row_board != board:
                continue
            human = mean(metrics["human"]) if metrics["human"] else None
            other = mean(metrics["other"]) if metrics["other"] else None
            total = mean(metrics["total"]) if metrics["total"] else None
            board_rows.append(
                {
                    "board": board,
                    "model": model,
                    "score": total,
                    "human": human,
                    "other": other,
                    "object": mean(metrics["object"]) if metrics["object"] else None,
                    "background": mean(metrics["background"]) if metrics["background"] else None,
                    "rows": row_counts[(board, model)],
                    "n": len(metrics["total"]),
                }
            )
        board_rankings[board] = rank_rows([row for row in board_rows if row["score"] is not None], "score")

    by_model: dict[str, dict[str, Any]] = defaultdict(lambda: {"board_scores": {}, "human": [], "other": []})
    for board, ranking in board_rankings.items():
        for row in ranking:
            by_model[row["model"]]["board_scores"][board] = row["score"]
            if row["human"] is not None:
                by_model[row["model"]]["human"].append(row["human"])
            if row["other"] is not None:
                by_model[row["model"]]["other"].append(row["other"])

    overall_rows = []
    for model, data in by_model.items():
        board_scores = data["board_scores"]
        if not board_scores:
            continue
        overall_rows.append(
            {
                "model": model,
                "score": mean(board_scores.values()),
                "board1": board_scores.get("board1"),
                "board2": board_scores.get("board2"),
                "board3": board_scores.get("board3"),
                "human": mean(data["human"]) if data["human"] else None,
                "other": mean(data["other"]) if data["other"] else None,
                "boards": len(board_scores),
            }
        )
    return rank_rows(overall_rows, "score"), board_rankings


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return lines


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_rankings(summary_csvs: list[Path], output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    rows = read_summary_rows(summary_csvs)
    overall, board_rankings = aggregate(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "board123_overall_ranking.csv",
        overall,
        ["rank", "model", "score", "board1", "board2", "board3", "human", "other", "boards"],
    )
    board_rows = [row for board in ("board1", "board2", "board3") for row in board_rankings[board]]
    write_csv(
        output_dir / "board123_board_total_rankings.csv",
        board_rows,
        ["board", "rank", "model", "score", "human", "other", "object", "background", "rows", "n"],
    )

    md_path = output_dir / "board123_overall_ranking.md"
    lines = [
        "# Board1/2/3 Entity Fidelity Combined Ranking",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "Scoring:",
        "- Human: mean `Human_EF_with_LTC`.",
        "- Other: per-row mean of available `Object_EF_with_LTC` and `Scene_EF_with_LTC`.",
        "- Board total: per-row `0.7 * Human + 0.3 * Other`; if a row has no Other branch, board total falls back to Human.",
        "- Combined ranking: simple mean of available board totals across board1, board2, and board3.",
        "- Duplicate `(sample_id, model)` rows across input summaries are deduplicated by keeping the latest summary path in input order.",
        "",
        "Inputs:",
        "",
    ]
    input_rows = []
    for path in summary_csvs:
        if path.exists():
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                count = sum(1 for _ in csv.DictReader(handle))
            input_rows.append([count, f"`{path}`"])
    lines.extend(md_table(["Rows", "Source CSV"], input_rows))

    lines.extend(["", "## Combined Overall Ranking", ""])
    lines.extend(
        md_table(
            ["Rank", "Model", "Overall Mean", "Board1", "Board2", "Board3", "Human Mean", "Other Mean", "Boards"],
            [
                [
                    row["rank"],
                    row["model"],
                    fmt(row["score"]),
                    fmt(row["board1"]),
                    fmt(row["board2"]),
                    fmt(row["board3"]),
                    fmt(row["human"]),
                    fmt(row["other"]),
                    row["boards"],
                ]
                for row in overall
            ],
        )
    )

    for board in ("board1", "board2", "board3"):
        lines.extend(["", f"# {board}", "", "## Board Total Ranking 0.7 Human / 0.3 Other", ""])
        ranking = board_rankings[board]
        if not ranking:
            lines.append("No rows.")
            continue
        lines.extend(
            md_table(
                ["Rank", "Model", "Score", "Human", "Other", "Object", "Background", "Rows", "N"],
                [
                    [
                        row["rank"],
                        row["model"],
                        fmt(row["score"]),
                        fmt(row["human"]),
                        fmt(row["other"]),
                        fmt(row["object"]),
                        fmt(row["background"]),
                        row["rows"],
                        row["n"],
                    ]
                    for row in ranking
                ],
            )
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate finalef board123 entity-fidelity rankings from summary CSV files.")
    parser.add_argument("summary_csv", nargs="*", type=Path)
    parser.add_argument("--summary-root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary_csvs = args.summary_csv or sorted(args.summary_root.glob("summary_*.csv"))
    if not summary_csvs:
        raise SystemExit(f"No summary CSV files found under {args.summary_root}")
    md_path = write_rankings([path.expanduser().resolve() for path in summary_csvs], args.output_dir.expanduser().resolve())
    print(md_path)


if __name__ == "__main__":
    main()
