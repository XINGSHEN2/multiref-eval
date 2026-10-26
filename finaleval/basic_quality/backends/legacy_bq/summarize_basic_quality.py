#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_MODELS = ["happyhourse", "keling", "seedance", "veo", "vidu", "wan"]


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def fmt(value: float | None, digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def rank_map(scores: dict[str, float | None], higher_better: bool = True) -> dict[str, int]:
    valid = [(model, score) for model, score in scores.items() if score is not None]
    valid.sort(key=lambda item: ((-item[1]) if higher_better else item[1], item[0]))
    ranks: dict[str, int] = {}
    last_score: float | None = None
    current_rank = 0
    for index, (model, score) in enumerate(valid, start=1):
        if last_score is None or score != last_score:
            current_rank = index
            last_score = score
        ranks[model] = current_rank
    return ranks


def weighted_score(visual: float | None, temporal: float | None, audio: float | None) -> float | None:
    if visual is None or temporal is None or audio is None:
        return None
    return (visual * 4.0 + temporal + audio / 10.0) / 3.0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def add_per_sample_scores(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    scored: list[dict[str, str]] = []
    for row in rows:
        out = dict(row)
        visual = parse_float(row.get("visual_technical_quality_doverpp"))
        temporal = parse_float(row.get("temporal_stability"))
        audio = parse_float(row.get("audio_technical_quality"))
        score = weighted_score(visual, temporal, audio)
        out["audio_div10"] = fmt(audio / 10.0 if audio is not None else None)
        out["basic_quality_weighted_score"] = fmt(score)
        scored.append(out)
    return scored


def write_csv(path: Path, rows: list[dict[str, str]], preferred_fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for field in preferred_fields or []:
        if field not in fields:
            fields.append(field)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_models(rows: list[dict[str, str]], models: list[str]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        model = (row.get("model") or "").strip()
        if model:
            grouped[model].append(row)

    model_order = [model for model in models if model in grouped]
    for model in sorted(grouped):
        if model not in model_order:
            model_order.append(model)

    summaries: list[dict[str, object]] = []
    for model in model_order:
        group = grouped[model]
        visual = mean([parse_float(row.get("visual_technical_quality_doverpp")) for row in group])
        temporal = mean([parse_float(row.get("temporal_stability")) for row in group])
        audio = mean([parse_float(row.get("audio_technical_quality")) for row in group])
        total = weighted_score(visual, temporal, audio)
        summaries.append(
            {
                "model": model,
                "rows": len(group),
                "samples": len({row.get("sample", "") for row in group if row.get("sample")}),
                "visual_mean": visual,
                "temporal_mean": temporal,
                "audio_mean": audio,
                "audio_div10_mean": audio / 10.0 if audio is not None else None,
                "basic_quality_weighted_score": total,
            }
        )

    visual_ranks = rank_map({str(row["model"]): row["visual_mean"] for row in summaries})
    temporal_ranks = rank_map({str(row["model"]): row["temporal_mean"] for row in summaries})
    audio_ranks = rank_map({str(row["model"]): row["audio_mean"] for row in summaries})
    total_ranks = rank_map({str(row["model"]): row["basic_quality_weighted_score"] for row in summaries})

    output: list[dict[str, str]] = []
    for row in sorted(
        summaries,
        key=lambda item: (
            total_ranks.get(str(item["model"]), 9999),
            str(item["model"]),
        ),
    ):
        model = str(row["model"])
        output.append(
            {
                "rank": str(total_ranks.get(model, "")),
                "model": model,
                "basic_quality_weighted_score": fmt(row["basic_quality_weighted_score"]),
                "visual_mean": fmt(row["visual_mean"]),
                "visual_rank": str(visual_ranks.get(model, "")),
                "temporal_mean": fmt(row["temporal_mean"]),
                "temporal_rank": str(temporal_ranks.get(model, "")),
                "audio_mean": fmt(row["audio_mean"]),
                "audio_div10_mean": fmt(row["audio_div10_mean"]),
                "audio_rank": str(audio_ranks.get(model, "")),
                "rows": str(row["rows"]),
                "samples": str(row["samples"]),
            }
        )
    return output


def write_markdown(path: Path, model_rows: list[dict[str, str]], source_csv: Path) -> None:
    lines = [
        "# Basic Quality Ranking",
        "",
        "## What This Measures",
        "",
        "- Visual technical quality: DOVER++ technical branch; higher is better.",
        "- Temporal stability: frame-to-frame similarity and motion diagnostics; higher `temporal_stability` is better.",
        "- Audio technical quality: AudioBox PQ by default; higher is better.",
        "",
        "## Scoring Formula",
        "",
        "- Per-video score: `basic_quality_weighted_score = (visual * 4 + temporal + audio / 10) / 3`.",
        "- Model score: mean of per-video metrics by model, then the same formula is applied to model means.",
        "- Audio is divided by 10 before entering the formula.",
        "- No min-max normalization is used.",
        "",
        "## Inputs And Outputs",
        "",
        f"- Source detail CSV: `{source_csv}`",
        f"- Ranking CSV: `{path.with_name('model_ranking.csv')}`",
        "",
        "## Model Ranking",
        "",
        "| Rank | Model | Weighted Score | Visual | Visual Rank | Temporal | Temporal Rank | Audio | Audio/10 | Audio Rank | Rows |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in model_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["rank"],
                    row["model"],
                    row["basic_quality_weighted_score"],
                    row["visual_mean"],
                    row["visual_rank"],
                    row["temporal_mean"],
                    row["temporal_rank"],
                    row["audio_mean"],
                    row["audio_div10_mean"],
                    row["audio_rank"],
                    row["rows"],
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Basic Quality CSV with fixed weighted formula.")
    parser.add_argument("--input", required=True, help="Merged per-video basic quality CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for summary.csv, model_ranking.csv, ranking.md.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated preferred model order.")
    args = parser.parse_args()

    input_csv = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models = [item.strip() for item in args.models.split(",") if item.strip()]

    rows = add_per_sample_scores(read_rows(input_csv))
    summary_fields = [
        "sample",
        "model",
        "source_video",
        "visual_technical_quality_doverpp",
        "temporal_stability",
        "audio_technical_quality",
        "audio_div10",
        "basic_quality_weighted_score",
    ]
    summary_csv = output_dir / "summary.csv"
    write_csv(summary_csv, rows, summary_fields)

    model_rows = summarize_models(rows, models)
    ranking_csv = output_dir / "model_ranking.csv"
    write_csv(
        ranking_csv,
        model_rows,
        [
            "rank",
            "model",
            "basic_quality_weighted_score",
            "visual_mean",
            "visual_rank",
            "temporal_mean",
            "temporal_rank",
            "audio_mean",
            "audio_div10_mean",
            "audio_rank",
            "rows",
            "samples",
        ],
    )
    write_markdown(output_dir / "ranking.md", model_rows, summary_csv)
    print(f"Wrote {summary_csv}")
    print(f"Wrote {ranking_csv}")
    print(f"Wrote {output_dir / 'ranking.md'}")


if __name__ == "__main__":
    main()
