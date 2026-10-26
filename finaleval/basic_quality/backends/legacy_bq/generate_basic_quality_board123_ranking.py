#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


BOARD_RE = re.compile(r"(board[123])")


def parse_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def fmt(value: float | None, digits: int = 6) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def board_from_sample(sample: str) -> str:
    match = BOARD_RE.search(sample)
    return match.group(1) if match else "unknown"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def key_from_result(item: dict[str, Any]) -> str | None:
    for key in ("filename", "file", "video_path"):
        value = item.get(key)
        if value:
            return Path(str(value)).stem
    return None


def load_json_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: item
        for item in payload.get("results", [])
        if (key := key_from_result(item))
    }


def rows_from_summary(run_dir: Path) -> list[dict[str, Any]]:
    summary = run_dir / "summary.csv"
    if not summary.exists():
        return []
    rows = []
    for row in read_csv(summary):
        rows.append(
            {
                "sample": row.get("sample", ""),
                "model": row.get("model", ""),
                "visual": parse_float(row.get("visual_technical_quality_doverpp")),
                "temporal": parse_float(row.get("temporal_stability")),
                "audio": parse_float(row.get("audio_technical_quality")),
            }
        )
    return rows


def rows_from_available_metrics(run_dir: Path) -> list[dict[str, Any]]:
    manifest = run_dir / "batch_manifest.csv"
    metrics_dir = run_dir / "_batch_metrics"
    if not manifest.exists():
        return []
    visual = load_json_results(metrics_dir / "video_technical_dover_plus_plus.json")
    temporal = load_json_results(metrics_dir / "temporal_stability.json")
    audio = load_json_results(metrics_dir / "audio_aesthetic.json")
    rows = []
    for item in read_csv(manifest):
        stem = Path(item.get("batch_filename", "")).stem
        visual_row = visual.get(stem, {})
        temporal_row = temporal.get(stem, {})
        audio_row = audio.get(stem, {})
        audio_scores = audio_row.get("scores") if isinstance(audio_row.get("scores"), dict) else {}
        rows.append(
            {
                "sample": item.get("sample", ""),
                "model": item.get("model", ""),
                "visual": parse_float(visual_row.get("technical")),
                "temporal": parse_float(temporal_row.get("temporal_stability")),
                "audio": parse_float(audio_scores.get("PQ")),
            }
        )
    return rows


def weighted_score(visual: float | None, temporal: float | None, audio: float | None) -> float | None:
    if visual is None or temporal is None or audio is None:
        return None
    return (visual * 4.0 + temporal + audio / 10.0) / 3.0


def rank_map(scores: dict[str, float | None]) -> dict[str, int]:
    valid = [(model, score) for model, score in scores.items() if score is not None]
    valid.sort(key=lambda item: (-item[1], item[0]))
    ranks: dict[str, int] = {}
    last_score = None
    last_rank = 0
    for index, (model, score) in enumerate(valid, start=1):
        rank = last_rank if last_score is not None and score == last_score else index
        ranks[model] = rank
        last_score = score
        last_rank = rank
    return ranks


def aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, int]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    availability = {"visual": 0, "temporal": 0, "audio": 0}
    for row in rows:
        board = board_from_sample(str(row.get("sample", "")))
        model = str(row.get("model", "")).strip()
        if board in {"board1", "board2", "board3"} and model:
            grouped[(board, model)].append(row)
            for key in availability:
                if row.get(key) is not None:
                    availability[key] += 1

    board_rankings: dict[str, list[dict[str, Any]]] = {}
    for board in ("board1", "board2", "board3"):
        board_rows = []
        for (row_board, model), items in grouped.items():
            if row_board != board:
                continue
            visual = mean([x["visual"] for x in items if x.get("visual") is not None]) if any(x.get("visual") is not None for x in items) else None
            temporal = mean([x["temporal"] for x in items if x.get("temporal") is not None]) if any(x.get("temporal") is not None for x in items) else None
            audio = mean([x["audio"] for x in items if x.get("audio") is not None]) if any(x.get("audio") is not None for x in items) else None
            total = weighted_score(visual, temporal, audio)
            available_score = total if total is not None else visual
            board_rows.append(
                {
                    "board": board,
                    "model": model,
                    "score": available_score,
                    "weighted_score": total,
                    "visual": visual,
                    "temporal": temporal,
                    "audio": audio,
                    "rows": len(items),
                }
            )
        ranks = rank_map({row["model"]: row["score"] for row in board_rows})
        board_rankings[board] = sorted(
            [{**row, "rank": ranks.get(row["model"], "")} for row in board_rows],
            key=lambda row: (row["rank"] or 9999, row["model"]),
        )

    by_model: dict[str, dict[str, Any]] = defaultdict(lambda: {"boards": {}, "visual": [], "temporal": [], "audio": []})
    for board_rows in board_rankings.values():
        for row in board_rows:
            by_model[row["model"]]["boards"][row["board"]] = row["score"]
            for key in ("visual", "temporal", "audio"):
                if row[key] is not None:
                    by_model[row["model"]][key].append(row[key])

    overall = []
    for model, data in by_model.items():
        board_scores = data["boards"]
        overall.append(
            {
                "model": model,
                "score": mean(board_scores.values()) if board_scores else None,
                "board1": board_scores.get("board1"),
                "board2": board_scores.get("board2"),
                "board3": board_scores.get("board3"),
                "visual": mean(data["visual"]) if data["visual"] else None,
                "temporal": mean(data["temporal"]) if data["temporal"] else None,
                "audio": mean(data["audio"]) if data["audio"] else None,
                "boards": len(board_scores),
            }
        )
    ranks = rank_map({row["model"]: row["score"] for row in overall})
    overall = sorted([{**row, "rank": ranks.get(row["model"], "")} for row in overall], key=lambda row: (row["rank"] or 9999, row["model"]))
    return overall, board_rankings, availability


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
        *["| " + " | ".join(str(item) for item in row) + " |" for row in rows],
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_outputs(run_dir: Path, output_md: Path | None = None) -> Path:
    rows = rows_from_summary(run_dir) or rows_from_available_metrics(run_dir)
    if not rows:
        raise SystemExit(f"No summary.csv or usable metric JSON found under {run_dir}")
    overall, board_rankings, availability = aggregate(rows)
    output_md = output_md or run_dir / "basic_quality_3boards_custom_visual_weighted_ranking.md"
    output_md.parent.mkdir(parents=True, exist_ok=True)

    all_board_rows = [row for board in ("board1", "board2", "board3") for row in board_rankings[board]]
    write_csv(
        output_md.with_name("basic_quality_3boards_custom_visual_weighted_ranking.csv"),
        overall,
        ["rank", "model", "score", "board1", "board2", "board3", "visual", "temporal", "audio", "boards"],
    )
    write_csv(
        output_md.with_name("basic_quality_board_total_rankings.csv"),
        all_board_rows,
        ["board", "rank", "model", "score", "weighted_score", "visual", "temporal", "audio", "rows"],
    )

    complete = all(availability[key] > 0 for key in ("visual", "temporal", "audio"))
    lines = [
        "# Basic Quality 三个 Board 排名（Visual*4 + Temporal + Audio/10 版）",
        "",
        "## 统计口径",
        "",
        "- Visual: `visual_technical_quality_doverpp` / DOVER++ technical branch。",
        "- Temporal: `temporal_stability`。",
        "- Audio: `audio_technical_quality`，进入总分前除以 10。",
        "- 完整总分公式：`weighted_mean = (visual * 4 + temporal + audio / 10) / 3`。",
        "- 三个 Board 总排名：对可用 Board 分数做简单平均。",
    ]
    if not complete:
        missing = [key for key, count in availability.items() if count == 0]
        lines.extend(
            [
                f"- 注意：当前 run 缺少 {', '.join(missing)} 指标，无法计算完整 weighted_mean；当前排名使用已可用的 Visual 分数占位。",
                "- 后续完整跑出 `summary.csv` 后，此脚本会自动生成完整 Visual/Temporal/Audio 加权排名。",
            ]
        )

    lines.extend(["", "## 加权总分排名", ""])
    lines.extend(
        md_table(
            ["排名", "模型", "总分", "Board1", "Board2", "Board3", "Visual", "Temporal", "Audio", "Boards"],
            [
                [
                    row["rank"],
                    row["model"],
                    fmt(row["score"]),
                    fmt(row["board1"]),
                    fmt(row["board2"]),
                    fmt(row["board3"]),
                    fmt(row["visual"]),
                    fmt(row["temporal"]),
                    fmt(row["audio"]),
                    row["boards"],
                ]
                for row in overall
            ],
        )
    )

    for board in ("board1", "board2", "board3"):
        lines.extend(["", f"## {board} 排名", ""])
        ranking = board_rankings[board]
        if not ranking:
            lines.append("No rows.")
            continue
        lines.extend(
            md_table(
                ["排名", "模型", "分数", "Weighted", "Visual", "Temporal", "Audio", "Rows"],
                [
                    [
                        row["rank"],
                        row["model"],
                        fmt(row["score"]),
                        fmt(row["weighted_score"]),
                        fmt(row["visual"]),
                        fmt(row["temporal"]),
                        fmt(row["audio"]),
                        row["rows"],
                    ]
                    for row in ranking
                ],
            )
        )

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Basic Quality board123 ranking markdown from a run directory.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()
    print(write_outputs(args.run_dir.expanduser().resolve(), args.output_md.expanduser().resolve() if args.output_md else None))


if __name__ == "__main__":
    main()
