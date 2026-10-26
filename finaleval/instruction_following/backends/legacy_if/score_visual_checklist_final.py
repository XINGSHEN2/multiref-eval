#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CLASSIFICATION = SCRIPT_DIR / "visual_question_classification.csv"
WEIGHTS = {
    "Important Action": 0.5,
    "Wearing Requirement": 0.3,
    "Other Detail": 0.2,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def score(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def load_classification(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    out = {}
    for row in rows:
        question = str(row.get("question") or "").strip()
        if question:
            out[question] = row
    return out


def weighted_score(category_scores: dict[str, float]) -> float | None:
    present = [(cat, val) for cat, val in category_scores.items() if cat in WEIGHTS]
    if not present:
        return None
    weight_sum = sum(WEIGHTS[cat] for cat, _ in present)
    return sum(WEIGHTS[cat] * val for cat, val in present) / weight_sum


def build_rows(item_rows: list[dict[str, str]], classification: dict[str, dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    buckets: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    detail_rows = []
    for row in item_rows:
        if row.get("dimension") != "Visual Task Following":
            continue
        question = str(row.get("question") or "").strip()
        cls = classification.get(question)
        final_category = cls.get("final_category") if cls else "DROP"
        item_score = score(row.get("item_score"))
        detail_rows.append(
            {
                "task_id": row.get("task_id"),
                "sample_id": row.get("sample_id"),
                "model_name": row.get("model_name"),
                "question": question,
                "final_category": final_category,
                "item_score": fmt(item_score),
                "item_reason": row.get("item_reason", ""),
                "classification_reason": (cls or {}).get("classification_reason") or (cls or {}).get("updated_classification_reason") or "",
            }
        )
        if final_category not in WEIGHTS or item_score is None:
            continue
        buckets[
            (
                str(row.get("task_id") or ""),
                str(row.get("sample_id") or ""),
                str(row.get("model_name") or ""),
                final_category,
            )
        ].append(item_score)

    by_task: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    counts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
    for (task_id, sample_id, model_name, category), values in buckets.items():
        key = (task_id, sample_id, model_name)
        by_task[key][category] = mean(values)
        counts[key][category] = len(values)

    per_sample = []
    for (task_id, sample_id, model_name), cats in sorted(by_task.items()):
        visual_score = weighted_score(cats)
        per_sample.append(
            {
                "task_id": task_id,
                "sample_id": sample_id,
                "model_name": model_name,
                "board": sample_id.split("_", 1)[0] if "_" in sample_id else "",
                "Visual Task Following": fmt(visual_score),
                "Important Action": fmt(cats.get("Important Action")),
                "Wearing Requirement": fmt(cats.get("Wearing Requirement")),
                "Other Detail": fmt(cats.get("Other Detail")),
                "Important Action_count": counts[(task_id, sample_id, model_name)].get("Important Action", 0),
                "Wearing Requirement_count": counts[(task_id, sample_id, model_name)].get("Wearing Requirement", 0),
                "Other Detail_count": counts[(task_id, sample_id, model_name)].get("Other Detail", 0),
            }
        )
    return per_sample, detail_rows


def write_md(path: Path, per_sample: list[dict[str, Any]]) -> None:
    by_model: dict[str, list[float]] = defaultdict(list)
    by_model_cat: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in per_sample:
        model = str(row["model_name"])
        val = score(row.get("Visual Task Following"))
        if val is not None:
            by_model[model].append(val)
        for cat in WEIGHTS:
            cat_val = score(row.get(cat))
            if cat_val is not None:
                by_model_cat[(model, cat)].append(cat_val)
    ranking = sorted(by_model, key=lambda m: (-mean(by_model[m]), m))
    lines = [
        "# Visual Final Checklist Ranking",
        "",
        "Scoring matches `outputs/all sample/all-visual`: DROP questions are excluded; final score uses `0.5 * Important Action + 0.3 * Wearing Requirement + 0.2 * Other Detail`, renormalized when a sample has missing categories.",
        "",
        "| Rank | Model | Visual | Important Action | Wearing Requirement | Other Detail | Count |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, model in enumerate(ranking, start=1):
        lines.append(
            "| {rank} | {model} | {visual} | {action} | {wearing} | {detail} | {count} |".format(
                rank=rank,
                model=model,
                visual=f"{mean(by_model[model]):.4f}",
                action=f"{mean(by_model_cat[(model, 'Important Action')]):.4f}" if by_model_cat[(model, "Important Action")] else "",
                wearing=f"{mean(by_model_cat[(model, 'Wearing Requirement')]):.4f}" if by_model_cat[(model, "Wearing Requirement")] else "",
                detail=f"{mean(by_model_cat[(model, 'Other Detail')]):.4f}" if by_model_cat[(model, "Other Detail")] else "",
                count=len(by_model[model]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute visual score from checklist item details using final all-visual categories.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    item_detail = run_dir / "if_item_detail.csv"
    if not item_detail.exists():
        raise FileNotFoundError(item_detail)
    classification = load_classification(args.classification.expanduser().resolve())
    per_sample, detail_rows = build_rows(read_csv(item_detail), classification)
    write_csv(
        run_dir / "visual_final_per_sample.csv",
        per_sample,
        [
            "task_id",
            "sample_id",
            "model_name",
            "board",
            "Visual Task Following",
            "Important Action",
            "Wearing Requirement",
            "Other Detail",
            "Important Action_count",
            "Wearing Requirement_count",
            "Other Detail_count",
        ],
    )
    write_csv(
        run_dir / "visual_final_item_detail.csv",
        detail_rows,
        ["task_id", "sample_id", "model_name", "question", "final_category", "item_score", "item_reason", "classification_reason"],
    )
    write_md(run_dir / "visual_final_checklist_ranking.md", per_sample)
    print(f"Saved final visual checklist scoring to {run_dir / 'visual_final_per_sample.csv'}")


if __name__ == "__main__":
    main()
