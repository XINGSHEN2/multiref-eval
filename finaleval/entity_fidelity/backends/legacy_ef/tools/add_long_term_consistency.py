#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np


def _values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _third_mean(values: np.ndarray, start: int, end: int) -> float | None:
    if len(values) == 0:
        return None
    lo = min(len(values), max(0, start))
    hi = min(len(values), max(lo + 1, end))
    return float(np.mean(values[lo:hi])) if hi > lo else None


def temporal_metrics(rows: list[dict[str, Any]], key: str = "EF_t") -> dict[str, Any]:
    values = np.asarray(_values(rows, key), dtype=np.float32)
    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "early_mean": None,
            "late_mean": None,
            "late_over_early": None,
            "max_drop": None,
            "collapse_ratio": None,
            "normalized_std": None,
            "LTC": None,
            "EF_with_LTC": None,
        }

    mean = float(np.mean(values))
    std = float(np.std(values))
    min_value = float(np.min(values))
    n = len(values)
    third = max(1, n // 3)
    early_mean = _third_mean(values, 0, third)
    late_mean = _third_mean(values, n - third, n)
    late_over_early = (
        float(late_mean / early_mean)
        if early_mean is not None and late_mean is not None and abs(early_mean) > 1e-6
        else None
    )
    drops = values[:-1] - values[1:]
    max_drop = float(np.max(drops)) if len(drops) else 0.0

    collapse_threshold = 0.7 * mean
    collapse_ratio = float(np.mean(values < collapse_threshold))
    normalized_std = float(std / max(abs(mean), 1e-6))
    ltc = 1.0
    ltc -= 0.5 * min(1.0, normalized_std)
    ltc -= 0.3 * collapse_ratio
    if late_over_early is not None:
        ltc -= 0.2 * max(0.0, 1.0 - min(1.0, late_over_early))
    ltc = float(max(0.0, min(1.0, ltc)))

    return {
        "count": int(n),
        "mean": mean,
        "std": std,
        "min": min_value,
        "early_mean": early_mean,
        "late_mean": late_mean,
        "late_over_early": late_over_early,
        "max_drop": max_drop,
        "collapse_ratio": collapse_ratio,
        "collapse_threshold": float(collapse_threshold),
        "normalized_std": normalized_std,
        "LTC": ltc,
        "EF_with_LTC": float(mean * (0.7 + 0.3 * ltc)),
    }


def component_std(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, name in (
        ("Face_EF_t", "Face_EF"),
        ("Face_ArcFace_t", "Face_ArcFace"),
        ("Face_CurricularFace_t", "Face_CurricularFace"),
        ("Body_EF_t", "Body_EF"),
    ):
        values = np.asarray(_values(rows, key), dtype=np.float32)
        out[f"{name}_std"] = float(np.std(values)) if len(values) else None
        out[f"{name}_min"] = float(np.min(values)) if len(values) else None
    return out


def aggregate_entity_rows(video: dict[str, Any]) -> list[dict[str, Any]]:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for entity in (video.get("entities") or {}).values():
        for row in entity.get("frame_scores") or []:
            if row.get("EF_t") is None:
                continue
            by_frame.setdefault(int(row["frame_index"]), []).append(row)

    rows = []
    for frame_index in sorted(by_frame):
        items = by_frame[frame_index]
        rows.append(
            {
                "frame_index": frame_index,
                "EF_t": float(np.mean(_values(items, "EF_t"))),
                "Face_EF_t": (
                    float(np.mean(_values(items, "Face_EF_t")))
                    if _values(items, "Face_EF_t")
                    else None
                ),
                "Body_EF_t": (
                    float(np.mean(_values(items, "Body_EF_t")))
                    if _values(items, "Body_EF_t")
                    else None
                ),
            }
        )
    return rows


def video_ltc(video: dict[str, Any]) -> dict[str, Any]:
    rows = video.get("frame_scores") or []
    if not any(row.get("EF_t") is not None for row in rows):
        rows = aggregate_entity_rows(video)

    metrics = temporal_metrics(rows, "EF_t")
    metrics.update(component_std(rows))
    entity_metrics = {}
    for entity_id, entity in (video.get("entities") or {}).items():
        entity_rows = entity.get("frame_scores") or []
        entity_ltc = temporal_metrics(entity_rows, "EF_t")
        entity_ltc.update(component_std(entity_rows))
        entity_metrics[entity_id] = entity_ltc
    if entity_metrics:
        metrics["entities"] = entity_metrics
    return metrics


def branch_rows(video: dict[str, Any], branch: str) -> list[dict[str, Any]]:
    if branch == "human":
        rows = video.get("frame_scores") or []
        if any(row.get("EF_t") is not None for row in rows):
            return rows
        return aggregate_entity_rows(video)

    branch_data = video.get(branch.capitalize()) or {}
    return branch_data.get("frame_scores") or []


def overall_rows(video: dict[str, Any]) -> list[dict[str, Any]]:
    branch_weights = video.get("branch_weights") or {"human": 1.0}
    rows_by_branch = {
        "human": branch_rows(video, "human"),
        "object": branch_rows(video, "object"),
        "scene": branch_rows(video, "scene"),
    }
    by_frame: dict[int, dict[str, float]] = {}
    for branch, rows in rows_by_branch.items():
        if branch not in branch_weights:
            continue
        for row in rows:
            if row.get("EF_t") is None:
                continue
            by_frame.setdefault(int(row["frame_index"]), {})[branch] = float(row["EF_t"])

    out = []
    for frame_index in sorted(by_frame):
        scores = by_frame[frame_index]
        weight_sum = sum(float(branch_weights[key]) for key in scores if key in branch_weights)
        if weight_sum <= 0:
            continue
        ef_t = sum(float(branch_weights[key]) * value for key, value in scores.items() if key in branch_weights) / weight_sum
        out.append({"frame_index": frame_index, "EF_t": float(ef_t), "branch_scores": scores})
    return out


def object_ltc(video: dict[str, Any]) -> dict[str, Any] | None:
    object_data = video.get("Object")
    if not object_data:
        return None
    metrics = temporal_metrics(object_data.get("frame_scores") or [], "EF_t")
    object_metrics = {}
    for key, obj in (object_data.get("objects") or {}).items():
        object_metrics[key] = temporal_metrics(obj.get("frame_scores") or [], "EF_t")
    if object_metrics:
        metrics["objects"] = object_metrics
    return metrics


def summary_from_results(results: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "sample_id": results["sample_id"],
        "mode": results.get("mode", "single_human"),
        "videos": [],
    }
    for video in results.get("videos") or []:
        human_ltc = video_ltc(video)
        scene_ltc = temporal_metrics((video.get("Scene") or {}).get("frame_scores") or [], "EF_t") if video.get("Scene") else None
        obj_ltc = object_ltc(video)
        overall_ltc = temporal_metrics(overall_rows(video), "EF_t")
        item = {
            "video_name": Path(video["video_path"]).name,
            "EF": video.get("EF"),
            "Overall_EF": video.get("Overall_EF"),
            "Human_EF": video.get("Human_EF"),
            "Object_EF": video.get("Object_EF"),
            "Scene_EF": video.get("Scene_EF"),
            "branch_weights": video.get("branch_weights"),
            "EF_min": video.get("EF_min"),
            "EF_visible_weighted": video.get("EF_visible_weighted"),
            "Face_EF": video.get("Face_EF"),
            "Face_ArcFace": video.get("Face_ArcFace"),
            "Face_CurricularFace": video.get("Face_CurricularFace"),
            "Body_EF": video.get("Body_EF"),
            "LongTermConsistency": overall_ltc,
            "Human_LongTermConsistency": human_ltc,
            "Object_LongTermConsistency": obj_ltc,
            "Scene_LongTermConsistency": scene_ltc,
            "Object": {
                "Object_EF": (video.get("Object") or {}).get("Object_EF"),
                "valid_object_ratio": (video.get("Object") or {}).get("valid_object_ratio"),
                "objects": {
                    key: {
                        "Object_EF": obj.get("Object_EF"),
                        "valid_object_ratio": obj.get("valid_object_ratio"),
                        "prompt": obj.get("prompt"),
                    }
                    for key, obj in ((video.get("Object") or {}).get("objects") or {}).items()
                },
            }
            if video.get("Object")
            else None,
            "Scene": {
                "Scene_EF": (video.get("Scene") or {}).get("Scene_EF"),
                "valid_scene_ratio": (video.get("Scene") or {}).get("valid_scene_ratio"),
            }
            if video.get("Scene")
            else None,
            "entities": {
                entity_id: {
                    "EF": entity_result.get("EF"),
                    "visible_ratio": entity_result.get("visible_ratio"),
                    "Face_EF": entity_result.get("Face_EF"),
                    "Face_ArcFace": entity_result.get("Face_ArcFace"),
                    "Face_CurricularFace": entity_result.get("Face_CurricularFace"),
                    "Body_EF": entity_result.get("Body_EF"),
                }
                for entity_id, entity_result in (video.get("entities") or {}).items()
            },
        }
        summary["videos"].append(item)
    return summary


def default_summary_path(input_path: Path, separate_ltc_summary: bool) -> Path:
    if input_path.name.endswith("_summary.json"):
        return input_path
    if input_path.name == "result.json":
        return input_path.parent / ("ltc_summary.json" if separate_ltc_summary else "summary.json")
    if separate_ltc_summary:
        return input_path.with_name(f"{input_path.stem}_ltc_summary{input_path.suffix}")
    return input_path.with_name(f"{input_path.stem}_summary{input_path.suffix}")


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return sorted(set(paths))


def process_file(input_path: Path, output_path: Path) -> None:
    results = json.loads(input_path.read_text(encoding="utf-8"))
    summary = summary_from_results(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote LTC summary: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add long-term consistency metrics to EF summary JSON.")
    parser.add_argument("--input-json", nargs="+", required=True, help="EF result JSON path(s) or glob(s).")
    parser.add_argument("--output-json", help="Output summary JSON. Only valid with one input file.")
    parser.add_argument("--separate-ltc-summary", action="store_true", help="Write *_ltc_summary.json instead of *_summary.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = expand_inputs(args.input_json)
    if args.output_json and len(inputs) != 1:
        raise ValueError("--output-json can only be used with exactly one input JSON.")
    for input_path in inputs:
        output_path = Path(args.output_json) if args.output_json else default_summary_path(input_path, args.separate_ltc_summary)
        process_file(input_path, output_path)


if __name__ == "__main__":
    main()
