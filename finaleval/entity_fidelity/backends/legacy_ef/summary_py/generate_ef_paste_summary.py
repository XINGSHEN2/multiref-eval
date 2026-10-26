#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import OrderedDict, defaultdict
from statistics import mean
from pathlib import Path


BOARDS = ["board1", "board2", "board3"]


def parse_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) else parsed


def infer_board(sample_id: str) -> str | None:
    for board in BOARDS:
        if str(sample_id).startswith(f"{board}_"):
            return board
    return None


def row_other(row: dict[str, str]) -> float | None:
    values = [
        value
        for value in [
            parse_float(row.get("Object_EF_with_LTC")),
            parse_float(row.get("Scene_EF_with_LTC")),
        ]
        if value is not None
    ]
    return sum(values) / len(values) if values else None


def load_paste(
    paste_per_sample: Path | None,
    model_aliases: dict[str, str] | None = None,
) -> tuple[dict[tuple[str, str], float], dict[str, list[float]], dict[str, float], dict[str, float], Path | None]:
    if not paste_per_sample:
        return {}, {}, {}, {}, None
    paste_per_sample = paste_per_sample.expanduser().resolve()
    model_aliases = model_aliases or {}
    paste = {}
    quality_by_model: dict[str, list[float]] = defaultdict(list)
    penalty_by_model_values: dict[str, list[float]] = defaultdict(list)
    penalty_by_model_board_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with paste_per_sample.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row.get("sample_id") or "")
            model = str(row.get("model_name") or "").strip()
            model = model_aliases.get(model, model)
            penalty = parse_float(row.get("paste_penalty"))
            quality = parse_float(row.get("Paste Artifact Quality Score"))
            if sample_id and model and penalty is not None:
                paste[(sample_id, model)] = penalty
                penalty_by_model_values[model].append(penalty)
                board = infer_board(sample_id)
                if board:
                    penalty_by_model_board_values[model][board].append(penalty)
            if model and quality is not None:
                quality_by_model[model].append(quality)
    penalty_by_model = {model: mean(values) for model, values in penalty_by_model_values.items() if values}
    dedup_board_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (sample_id, model), penalty in paste.items():
        board = infer_board(sample_id)
        if board:
            dedup_board_values[model][board].append(penalty)
    penalty_by_model_board23 = {}
    for model, board_values in dedup_board_values.items():
        board_means = [mean(board_values[board]) for board in ("board2", "board3") if board_values.get(board)]
        if board_means:
            penalty_by_model_board23[model] = mean(board_means)
    return paste, quality_by_model, penalty_by_model, penalty_by_model_board23, paste_per_sample


def add_ef_row(
    rows: list[dict[str, object]],
    row: dict[str, str],
    paste: dict[tuple[str, str], float],
    paste_model_mean: dict[str, float] | None = None,
    fallback_to_model_paste_mean: bool = False,
    paste_model_board23_mean: dict[str, float] | None = None,
    use_board23_mean_for_board1: bool = False,
) -> None:
    sample_id = str(row.get("sample_id") or "")
    board = infer_board(sample_id)
    model = str(row.get("model") or "").strip()
    if board not in BOARDS or not model:
        return

    human = parse_float(row.get("Human_EF_with_LTC"))
    face = parse_float(row.get("Face_EF"))
    body = parse_float(row.get("Body_EF"))
    obj = parse_float(row.get("Object_EF_with_LTC"))
    bg = parse_float(row.get("Scene_EF_with_LTC"))
    other = row_other(row)

    metrics = {}
    if human is not None:
        metrics["human_ltc"] = human
        metrics["human_0p5_other_0p5"] = human if other is None else 0.5 * human + 0.5 * other
    if face is not None:
        metrics["face_only"] = face
        metrics["face_plus_other"] = face if other is None else 0.7 * face + 0.3 * other
    if body is not None:
        metrics["body_only"] = body
        metrics["body_plus_other"] = body if other is None else 0.7 * body + 0.3 * other
    if obj is not None:
        metrics["object_only"] = obj
    if bg is not None:
        metrics["background_only"] = bg

    available = [value for value in [human, obj, bg] if value is not None]
    if available:
        metrics["mean_human_object_background"] = sum(available) / len(available)

    penalty = paste.get((sample_id, model))
    if board == "board1" and use_board23_mean_for_board1 and paste_model_board23_mean:
        penalty = paste_model_board23_mean.get(model)
    if penalty is None and fallback_to_model_paste_mean and paste_model_mean:
        penalty = paste_model_mean.get(model)
    if penalty is not None:
        metrics["paste_penalty"] = penalty
        for base_metric in [
            "mean_human_object_background",
            "human_0p5_other_0p5",
            "body_plus_other",
            "face_plus_other",
        ]:
            if base_metric in metrics:
                metrics[f"{base_metric}_paste_penalty"] = metrics[base_metric] * penalty

    for metric, score in metrics.items():
        rows.append({"board": board, "model": model, "metric": metric, "score": score})


def aggregate(rows: list[dict[str, object]], metric_titles: OrderedDict[str, str]) -> list[dict[str, object]]:
    buckets: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        buckets[str(row["metric"])][str(row["model"])][str(row["board"])].append(float(row["score"]))

    summary_rows = []
    for metric, title in metric_titles.items():
        for model, board_map in buckets.get(metric, {}).items():
            board_means = []
            board_counts = []
            for board in BOARDS:
                values = board_map.get(board, [])
                if values:
                    board_means.append(sum(values) / len(values))
                    board_counts.append(len(values))
                else:
                    board_means.append(None)
                    board_counts.append(0)
            available = [value for value in board_means if value is not None]
            if not available:
                continue
            summary_rows.append(
                {
                    "metric": metric,
                    "metric_name": title,
                    "model": model,
                    "overall_mean": sum(available) / len(available),
                    "board1": board_means[0],
                    "board1_n": board_counts[0],
                    "board2": board_means[1],
                    "board2_n": board_counts[1],
                    "board3": board_means[2],
                    "board3_n": board_counts[2],
                    "boards_n": len(available),
                }
            )

    ranked = []
    for metric in metric_titles:
        selected = sorted(
            [row for row in summary_rows if row["metric"] == metric],
            key=lambda row: float(row["overall_mean"]),
            reverse=True,
        )
        previous = None
        rank = 0
        for index, row in enumerate(selected, start=1):
            score = row["overall_mean"]
            if previous is None or score != previous:
                rank = index
                previous = score
            item = dict(row)
            item["rank"] = rank
            ranked.append(item)
    return ranked


def fmt(value: object) -> str:
    return "-" if value is None or value == "" else f"{float(value):.4f}"


def write_csv_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "metric",
        "metric_name",
        "rank",
        "model",
        "overall_mean",
        "board1",
        "board1_n",
        "board2",
        "board2_n",
        "board3",
        "board3_n",
        "boards_n",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ["overall_mean", "board1", "board2", "board3"]:
                out[key] = "" if out.get(key) is None else f"{float(out[key]):.6f}"
            writer.writerow({field: out.get(field, "") for field in fields})


def write_md_summary(
    path: Path,
    rows: list[dict[str, object]],
    metric_titles: OrderedDict[str, str],
    ef_csvs: list[Path],
    paste_path: Path | None,
    quality_by_model: dict[str, list[float]],
) -> None:
    lines = [
        "# EF Score Summary",
        "",
        "Sources:",
    ]
    for csv_path in ef_csvs:
        lines.append(f"- EF summary CSV: `{csv_path}`")
    if paste_path:
        lines.append(f"- Paste artifact per-sample CSV: `{paste_path}`")
    lines.extend(
        [
            "",
            "Aggregation: compute per-row metric, average within each board by model, then use equal mean over available boards.",
            "For formulas with `Other`, `Other = mean(Object_EF_with_LTC, Scene_EF_with_LTC)`, and rows without Other fall back to the human/face/body branch.",
            "",
            "Paste adjustment: adjusted score is `base_score * paste_penalty`. Exact `(sample_id, model)` paste rows are used when available; if enabled, missing sample-level rows fall back to the model mean paste penalty. `paste_penalty` is produced by the MLLM judge: 5->1.00, 4->0.95, 3->0.90, 2->0.80, 1->0.65.",
            "",
        ]
    )

    if paste_path:
        pen_rows = [row for row in rows if row["metric"] == "paste_penalty"]
        quality = {model: sum(values) / len(values) for model, values in quality_by_model.items() if values}
        lines.extend(
            [
                "## Paste Artifact Judge Summary",
                "",
                "| Rank | Model | Paste Quality 1-5 | Mean Penalty | Board1 Penalty | N | Board2 Penalty | N | Board3 Penalty | N |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in pen_rows:
            lines.append(
                f"| {row['rank']} | {row['model']} | {fmt(quality.get(str(row['model'])))} | "
                f"{fmt(row['overall_mean'])} | {fmt(row['board1'])} | {row['board1_n']} | "
                f"{fmt(row['board2'])} | {row['board2_n']} | {fmt(row['board3'])} | {row['board3_n']} |"
            )

    for metric, title in metric_titles.items():
        if metric == "paste_penalty":
            continue
        selected = [row for row in rows if row["metric"] == metric]
        if not selected:
            continue
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Rank | Model | Overall Mean | Board1 | N | Board2 | N | Board3 | N | Boards |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in selected:
            lines.append(
                f"| {row['rank']} | {row['model']} | {fmt(row['overall_mean'])} | "
                f"{fmt(row['board1'])} | {row['board1_n']} | {fmt(row['board2'])} | {row['board2_n']} | "
                f"{fmt(row['board3'])} | {row['board3_n']} | {row['boards_n']} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_model_aliases(values: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for value in values:
        if not value:
            continue
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise SystemExit(f"Invalid --paste-model-alias {item!r}; expected old=new")
            old, new = item.split("=", 1)
            old = old.strip()
            new = new.strip()
            if old and new:
                aliases[old] = new
    return aliases


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EF summary with optional paste-artifact penalty.")
    parser.add_argument("--ef-summary-csv", action="append", required=True, type=Path)
    parser.add_argument("--paste-per-sample", type=Path, default=None)
    parser.add_argument("--paste-model-alias", action="append", default=[], help="Map paste model names before matching, e.g. happyhourse=happyhouse1.1")
    parser.add_argument("--fallback-to-model-paste-mean", action="store_true", help="Use the model mean paste penalty when a row has no exact sample-level paste match.")
    parser.add_argument("--use-board23-mean-for-board1-paste", action="store_true", help="For board1 rows, use the simple average of the model's board2 and board3 mean paste penalties.")
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    paste, quality_by_model, paste_model_mean, paste_model_board23_mean, paste_path = load_paste(args.paste_per_sample, parse_model_aliases(args.paste_model_alias))

    rows: list[dict[str, object]] = []
    ef_csvs = [path.expanduser().resolve() for path in args.ef_summary_csv]
    for csv_path in ef_csvs:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                add_ef_row(
                    rows,
                    row,
                    paste,
                    paste_model_mean,
                    args.fallback_to_model_paste_mean,
                    paste_model_board23_mean,
                    args.use_board23_mean_for_board1_paste,
                )

    metric_titles = OrderedDict(
        [
            ("mean_human_object_background_paste_penalty", "Mean(Human, Object, Background) * Paste Penalty"),
            ("human_0p5_other_0p5_paste_penalty", "(0.5 Human + 0.5 Other) * Paste Penalty"),
            ("body_plus_other_paste_penalty", "(0.7 Body + 0.3 Other) * Paste Penalty"),
            ("face_plus_other_paste_penalty", "(0.7 Face + 0.3 Other) * Paste Penalty"),
            ("paste_penalty", "Paste Penalty only"),
            ("mean_human_object_background", "Mean(Human, Object, Background)"),
            ("human_0p5_other_0p5", "0.5 Human + 0.5 Other"),
            ("face_plus_other", "0.7 Face + 0.3 Other"),
            ("body_plus_other", "0.7 Body + 0.3 Other"),
            ("face_only", "Face only"),
            ("body_only", "Body only"),
            ("object_only", "Object only"),
            ("background_only", "Background only"),
        ]
    )
    ranked = aggregate(rows, metric_titles)
    write_csv_summary(args.output_csv.expanduser().resolve(), ranked)
    write_md_summary(
        args.output_md.expanduser().resolve(),
        ranked,
        metric_titles,
        ef_csvs,
        paste_path,
        quality_by_model,
    )


if __name__ == "__main__":
    main()
