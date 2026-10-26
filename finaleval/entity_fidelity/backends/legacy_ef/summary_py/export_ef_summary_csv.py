#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = [
    "sample",
    "sample_id",
    "model",
    "video_name",
    "EF",
    "EF_with_LTC",
    "LTC",
    "Human_EF",
    "Human_EF_with_LTC",
    "Human_LTC",
    "Scene_EF",
    "Scene_EF_mean",
    "Scene_best_frame_index",
    "Scene_aggregation",
    "Scene_EF_with_LTC",
    "Scene_LTC",
    "Object_EF",
    "Object_EF_with_LTC",
    "Object_LTC",
    "Face_EF",
    "Face_Recognition",
    "face_recognition_model",
    "Face_ArcFace",
    "Face_CurricularFace",
    "Cartoon_Head_EF",
    "Body_EF",
    "Body_DINOv2",
    "Body_SigLIP",
    "Scene_DINOv2",
    "Scene_SigLIP",
    "valid_face_ratio",
    "valid_cartoon_head_ratio",
    "valid_body_ratio",
    "valid_scene_ratio",
    "valid_object_ratio",
    "used_frames",
    "sampled_frames",
    "branch_weights",
    "result_json",
    "summary_json",
]

def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric(block: dict[str, Any] | None, key: str) -> float | None:
    if not block:
        return None
    return _float(block.get(key))


def _entity_mean(video: dict[str, Any], key: str) -> float | None:
    values = [
        float(entity[key])
        for entity in (video.get("entities") or {}).values()
        if entity.get(key) is not None
    ]
    return _mean(values)


def _video_summary_by_name(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(video.get("video_name")): video
        for video in summary.get("videos") or []
        if video.get("video_name")
    }


def row_from_run(sample_dir: Path, model_dir: Path) -> list[dict[str, Any]]:
    result_path = model_dir / "result.json"
    summary_path = model_dir / "summary.json"
    if not result_path.exists() or not summary_path.exists():
        return []

    result = _load(result_path)
    summary = _load(summary_path)
    summary_by_name = _video_summary_by_name(summary)
    rows = []
    for video in result.get("videos") or []:
        video_name = Path(video.get("video_path", "")).name
        summary_video = summary_by_name.get(video_name, {})
        overall_ltc = summary_video.get("LongTermConsistency") or {}
        human_ltc = summary_video.get("Human_LongTermConsistency") or {}
        scene_ltc = summary_video.get("Scene_LongTermConsistency") or {}
        object_ltc = summary_video.get("Object_LongTermConsistency") or {}

        scene = video.get("Scene") or {}
        object_data = video.get("Object") or {}
        face_ef = _float(video.get("Face_EF"))
        body_ef = _float(video.get("Body_EF"))
        if face_ef is None:
            face_ef = _entity_mean(video, "Face_EF")
        face_recognition = _float(video.get("Face_Recognition"))
        if face_recognition is None:
            face_recognition = _entity_mean(video, "Face_Recognition")
        face_arcface = _float(video.get("Face_ArcFace"))
        face_curricular = _float(video.get("Face_CurricularFace"))
        if face_arcface is None:
            face_arcface = _entity_mean(video, "Face_ArcFace")
        if face_curricular is None:
            face_curricular = _entity_mean(video, "Face_CurricularFace")
        cartoon_head_ef = _float(video.get("Cartoon_Head_EF"))
        if cartoon_head_ef is None:
            cartoon_head_ef = _entity_mean(video, "Cartoon_Head_EF")
        if body_ef is None:
            body_ef = _entity_mean(video, "Body_EF")

        rows.append(
            {
                "sample": sample_dir.name,
                "sample_id": result.get("sample_id"),
                "model": model_dir.name,
                "video_name": video_name,
                "EF": _float(summary_video.get("EF", video.get("EF"))),
                "EF_with_LTC": _metric(overall_ltc, "EF_with_LTC"),
                "LTC": _metric(overall_ltc, "LTC"),
                "Human_EF": _float(summary_video.get("Human_EF", video.get("Human_EF"))),
                "Human_EF_with_LTC": _metric(human_ltc, "EF_with_LTC"),
                "Human_LTC": _metric(human_ltc, "LTC"),
                "Scene_EF": _float(summary_video.get("Scene_EF", video.get("Scene_EF"))),
                "Scene_EF_mean": _float(scene.get("Scene_EF_mean")),
                "Scene_best_frame_index": scene.get("best_frame_index"),
                "Scene_aggregation": scene.get("aggregation"),
                "Scene_EF_with_LTC": _metric(scene_ltc, "EF_with_LTC"),
                "Scene_LTC": _metric(scene_ltc, "LTC"),
                "Object_EF": _float(summary_video.get("Object_EF", video.get("Object_EF"))),
                "Object_EF_with_LTC": _metric(object_ltc, "EF_with_LTC"),
                "Object_LTC": _metric(object_ltc, "LTC"),
                "Face_EF": face_ef,
                "Face_Recognition": face_recognition,
                "face_recognition_model": video.get("face_recognition_model") or result.get("face_recognition_model"),
                "Face_ArcFace": face_arcface,
                "Face_CurricularFace": face_curricular,
                "Cartoon_Head_EF": cartoon_head_ef,
                "Body_EF": body_ef,
                "Body_DINOv2": _float(video.get("Body_DINOv2")),
                "Body_SigLIP": _float(video.get("Body_SigLIP")),
                "Scene_DINOv2": _float(scene.get("Scene_DINOv2")),
                "Scene_SigLIP": _float(scene.get("Scene_SigLIP")),
                "valid_face_ratio": _float(video.get("valid_face_ratio")),
                "valid_cartoon_head_ratio": _float(video.get("valid_cartoon_head_ratio")),
                "valid_body_ratio": _float(video.get("valid_body_ratio")),
                "valid_scene_ratio": _float(scene.get("valid_scene_ratio")),
                "valid_object_ratio": _float(object_data.get("valid_object_ratio")),
                "used_frames": video.get("used_frames"),
                "sampled_frames": video.get("sampled_frames"),
                "branch_weights": json.dumps(video.get("branch_weights") or {}, ensure_ascii=False, sort_keys=True),
                "result_json": str(result_path),
                "summary_json": str(summary_path),
            }
        )
    return rows


def collect_rows(root: Path, run_timestamp: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if run_timestamp and not sample_dir.name.endswith(f"_{run_timestamp}"):
            continue
        for model_dir in sorted(path for path in sample_dir.iterdir() if path.is_dir()):
            rows.extend(row_from_run(sample_dir, model_dir))
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export EF run_outputs tree to a model-by-sample CSV summary.")
    parser.add_argument("--root", default="ef_eval/run_outputs/testvideo_ef", help="Root containing {sample}/{model}/result.json.")
    parser.add_argument("--output-csv", help="Output CSV path. Defaults to <root>/summary.csv.")
    parser.add_argument("--run-timestamp", help="Only include sample directories ending with _<timestamp>.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_path = Path(args.output_csv) if args.output_csv else root / "summary.csv"
    rows = collect_rows(root, run_timestamp=args.run_timestamp)
    write_csv(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
