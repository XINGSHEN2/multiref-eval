#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


DIMENSION = "Stable Speaking Frontal Face"
EXTRA_FIELDS = [
    "stable_speaking_frontal_score",
    "stable_speaking_frontal_pass",
    "stable_speaking_frontal_applicable",
    "stable_speaking_frontal_reason",
]


def read_result_rows(path: Path) -> list[dict[str, Any]]:
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


def normalize_sample_id(value: str | None) -> str:
    value = (value or "").strip()
    match = re.search(r"(board[0-9]+)[_-]?([0-9]{1,3})", value, flags=re.IGNORECASE)
    if not match:
        return value
    return f"{match.group(1).lower()}_{int(match.group(2)):03d}"


def infer_sample_id(row: dict[str, str]) -> str:
    for key in ("sample_id", "source_sample_id", "sample", "id", "index"):
        value = row.get(key)
        if value:
            normalized = normalize_sample_id(value)
            if normalized:
                return normalized
    video_path = row.get("video_path") or row.get("combined_video") or row.get("path") or row.get("file") or ""
    return normalize_sample_id(Path(video_path).stem)


def infer_model_name(row: dict[str, str]) -> str:
    for key in ("model_name", "model", "method"):
        if row.get(key):
            return row[key].strip()
    video_path = row.get("video_path") or row.get("combined_video") or row.get("path") or row.get("file") or ""
    path = Path(video_path)
    parent = path.parent.name
    if re.fullmatch(r"board[0-9]+", parent, flags=re.IGNORECASE) and len(path.parents) > 1:
        return path.parents[1].name
    return parent or "model"


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass"}
    return False


def float_value(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def result_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("model_name") or ""), normalize_sample_id(str(row.get("sample_id") or ""))


def build_result_index(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    by_task_id = {}
    by_model_sample = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if task_id and task_id not in by_task_id:
            by_task_id[task_id] = row
        key = result_key(row)
        if all(key) and key not in by_model_sample:
            by_model_sample[key] = row
    return by_task_id, by_model_sample


def csv_task_id(row: dict[str, str]) -> str:
    return str(row.get("index") or row.get("id") or "")


def stable_payload(result_row: dict[str, Any]) -> dict[str, Any]:
    scores = result_row.get("scores") or {}
    payload = scores.get(DIMENSION) or {}
    return payload if isinstance(payload, dict) else {}


def should_keep(payload: dict[str, Any], min_score: float) -> bool:
    if not bool_value(payload.get("applicable")):
        return False
    if not bool_value(payload.get("stable_speaking_frontal_face")):
        return False
    score = float_value(payload.get("score"))
    return score is not None and score >= min_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter an input CSV to stable frontal speaking-face rows.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True, help="PureAC ac_results.json from the MLLM filter run.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--rejected-csv", type=Path, default=None)
    parser.add_argument("--min-score", type=float, default=4.0)
    args = parser.parse_args()

    result_rows = read_result_rows(args.results_json.expanduser().resolve())
    by_task_id, by_model_sample = build_result_index(result_rows)

    kept = []
    rejected = []
    with args.input_csv.expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {args.input_csv}")
        fieldnames = list(reader.fieldnames)
        output_fields = fieldnames + [field for field in EXTRA_FIELDS if field not in fieldnames]
        for row in reader:
            result = by_task_id.get(csv_task_id(row))
            if result is None:
                key = (infer_model_name(row), infer_sample_id(row))
                result = by_model_sample.get(key)
            payload = stable_payload(result or {})
            out = dict(row)
            out["stable_speaking_frontal_score"] = payload.get("score")
            out["stable_speaking_frontal_pass"] = should_keep(payload, args.min_score)
            out["stable_speaking_frontal_applicable"] = payload.get("applicable")
            out["stable_speaking_frontal_reason"] = payload.get("reason", "")
            if should_keep(payload, args.min_score):
                kept.append(out)
            else:
                rejected.append(out)

    args.output_csv.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.expanduser().resolve().open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(kept)

    if args.rejected_csv:
        args.rejected_csv.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        with args.rejected_csv.expanduser().resolve().open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields)
            writer.writeheader()
            writer.writerows(rejected)

    print(f"Kept {len(kept)} rows; rejected {len(rejected)} rows.")
    print(f"Saved filtered CSV to {args.output_csv.expanduser().resolve()}")


if __name__ == "__main__":
    main()
