#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np


SCOPES = ["overall", "board1", "board2", "board3"]
DEFAULT_EXCLUDED = "seedance:board2_007,seedance:board2_100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank VSLS results with lowest-LSE-D track selection and stable-frontal filtering."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--detail-csv", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--id-column", default="index")
    parser.add_argument("--model-column", default="model_name")
    parser.add_argument("--excluded", default=DEFAULT_EXCLUDED, help="Comma-separated model:sample_id pairs.")
    return parser.parse_args()


def parse_excluded(value: str) -> set[tuple[str, str]]:
    out = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid excluded item {item!r}; expected model:sample_id")
        model, sample_id = item.split(":", 1)
        out.add((model.strip(), sample_id.strip()))
    return out


def load_metadata(path: Path | None, id_column: str) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if id_column not in (reader.fieldnames or []):
            return {}
        return {row[id_column]: row for row in reader}


def infer_model(item_id: str, result: dict[str, Any], metadata: dict[str, dict[str, str]], model_column: str) -> str:
    if item_id in metadata and metadata[item_id].get(model_column):
        return metadata[item_id][model_column]
    if result.get("model_name"):
        return str(result["model_name"])
    if "::" in item_id:
        return item_id.split("::", 1)[0]
    if "/" in item_id:
        return item_id.split("/", 1)[0]
    return item_id.split("_", 1)[0]


def infer_sample_id(item_id: str, result: dict[str, Any], metadata: dict[str, dict[str, str]]) -> str:
    if item_id in metadata and metadata[item_id].get("sample_id"):
        return metadata[item_id]["sample_id"]
    if result.get("sample_id"):
        return str(result["sample_id"])
    if "::" in item_id:
        return item_id.rsplit("::", 1)[-1]
    if "/" in item_id:
        return item_id.split("/", 1)[-1]
    parts = item_id.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 else item_id


def norm_score(lse_c: float, lse_d: float) -> tuple[float, float, float, float]:
    c_norm = min(max((lse_c - 1.5) / (7.0 - 1.5), 0.0), 1.0)
    d_norm = min(max((13.0 - lse_d) / (13.0 - 7.0), 0.0), 1.0)
    score_0_1 = 0.60 * c_norm + 0.40 * d_norm
    return c_norm, d_norm, score_0_1, 1.0 + 4.0 * score_0_1


def sample_frame_indices(frame_count: int, n: int = 8) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count <= n:
        return list(range(frame_count))
    return sorted({int(round(x)) for x in np.linspace(0, frame_count - 1, n)})


def audit_crop(crop_path: str, cascade: cv2.CascadeClassifier) -> dict[str, Any]:
    path = Path(crop_path)
    if not path.is_file():
        return {
            "crop_exists": False,
            "frames_sampled": 0,
            "frontal_hits": 0,
            "frontal_hit_rate": 0.0,
            "mean_face_w_ratio": 0.0,
            "mean_face_h_ratio": 0.0,
            "center_std_ratio": math.inf,
            "size_cv": math.inf,
            "stable_frontal": False,
            "audit_reason": "missing_crop",
        }

    cap = cv2.VideoCapture(str(path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0)
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0)
    hits: list[tuple[float, float, float, float]] = []
    sampled = 0
    for idx in sample_frame_indices(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        sampled += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
        if len(faces) == 0:
            continue
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        hits.append(((x + w / 2.0) / width, (y + h / 2.0) / height, w / width, h / height))
    cap.release()

    frontal_hits = len(hits)
    hit_rate = frontal_hits / sampled if sampled else 0.0
    if hits:
        centers = np.array([[item[0], item[1]] for item in hits], dtype=float)
        sizes = np.array([item[2] * item[3] for item in hits], dtype=float)
        mean_w = float(mean(item[2] for item in hits))
        mean_h = float(mean(item[3] for item in hits))
        center_std = float(np.linalg.norm(np.std(centers, axis=0)))
        size_cv = float(np.std(sizes) / max(float(np.mean(sizes)), 1e-9))
    else:
        mean_w = 0.0
        mean_h = 0.0
        center_std = math.inf
        size_cv = math.inf

    stable = hit_rate >= 0.75 and mean_w >= 0.28 and center_std <= 0.12 and size_cv <= 0.35
    if stable:
        reason = "stable_frontal"
    elif hit_rate < 0.75:
        reason = "low_frontal_detection_rate"
    elif mean_w < 0.28:
        reason = "face_too_small"
    elif center_std > 0.12:
        reason = "face_center_unstable"
    else:
        reason = "face_size_unstable"
    return {
        "crop_exists": True,
        "frames_sampled": sampled,
        "frontal_hits": frontal_hits,
        "frontal_hit_rate": hit_rate,
        "mean_face_w_ratio": mean_w,
        "mean_face_h_ratio": mean_h,
        "center_std_ratio": center_std,
        "size_cv": size_cv,
        "stable_frontal": stable,
        "audit_reason": reason,
    }


def choose_lowest_lsed_track(segment: dict[str, Any]) -> dict[str, Any] | None:
    candidates = []
    for row in segment.get("all_track_scores") or []:
        try:
            lse_c = float(row["confidence"])
            lse_d = float(row["min_distance"])
            offset = float(row["offset_frames"])
        except (KeyError, TypeError, ValueError):
            continue
        if abs(offset) >= 10:
            continue
        c_norm, d_norm, score_0_1, score_1_5 = norm_score(lse_c, lse_d)
        enriched = dict(row)
        enriched.update(
            {
                "lse_c": lse_c,
                "lse_d": lse_d,
                "offset_frames": offset,
                "abs_offset_frames": abs(offset),
                "c_norm": c_norm,
                "d_norm": d_norm,
                "score_0_1": score_0_1,
                "score_1_5": score_1_5,
            }
        )
        candidates.append(enriched)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item["lse_d"],
            item["abs_offset_frames"],
            -item["lse_c"],
        ),
    )


def build_detail_rows(
    results_dir: Path,
    metadata: dict[str, dict[str, str]],
    model_column: str,
    excluded: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"failed to load cascade: {cascade_path}")

    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        item_id = str(result.get("id") or path.stem)
        model = infer_model(item_id, result, metadata, model_column)
        sample_id = infer_sample_id(item_id, result, metadata)
        if (model, sample_id) in excluded:
            continue
        board = sample_id.split("_", 1)[0] if "_" in sample_id else "unknown"
        if result.get("video_score_0_1") is None:
            continue
        for segment in result.get("segments") or []:
            if segment.get("status") != "scored":
                continue
            selected = choose_lowest_lsed_track(segment)
            if selected is None:
                continue
            audit = audit_crop(str(selected.get("crop_path") or ""), cascade)
            rows.append(
                {
                    "model": model,
                    "sample_id": sample_id,
                    "board": board,
                    "segment_id": segment.get("segment_id"),
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "track_id": selected.get("track_id"),
                    "original_best_track_id": segment.get("best_track_id"),
                    "lse_c": selected["lse_c"],
                    "lse_d": selected["lse_d"],
                    "offset_frames": selected["offset_frames"],
                    "abs_offset_frames": selected["abs_offset_frames"],
                    "c_norm": selected["c_norm"],
                    "d_norm": selected["d_norm"],
                    "score_0_1": selected["score_0_1"],
                    "score_1_5": selected["score_1_5"],
                    "crop_path": selected.get("crop_path"),
                    **audit,
                }
            )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = [row for row in rows if row["abs_offset_frames"] < 10 and row["stable_frontal"]]
    out: list[dict[str, Any]] = []
    for scope in SCOPES:
        scoped = usable if scope == "overall" else [row for row in usable if row["board"] == scope]
        video_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in scoped:
            video_groups[(row["model"], row["sample_id"])].append(row)

        per_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for (model, _sample_id), group in video_groups.items():
            per_model[model].append(
                {
                    "score_1_5": mean(row["score_1_5"] for row in group),
                    "score_0_1": mean(row["score_0_1"] for row in group),
                    "lse_c": mean(row["lse_c"] for row in group),
                    "lse_d": mean(row["lse_d"] for row in group),
                    "abs_offset_frames": mean(row["abs_offset_frames"] for row in group),
                    "windows": len(group),
                }
            )

        ranked = []
        for model, videos in per_model.items():
            ranked.append(
                {
                    "scope": scope,
                    "model": model,
                    "mean_score_1_5": mean(row["score_1_5"] for row in videos),
                    "mean_score_0_1": mean(row["score_0_1"] for row in videos),
                    "mean_lse_c": mean(row["lse_c"] for row in videos),
                    "mean_lse_d": mean(row["lse_d"] for row in videos),
                    "mean_abs_offset_frames": mean(row["abs_offset_frames"] for row in videos),
                    "scored_videos": len(videos),
                    "kept_windows": sum(row["windows"] for row in videos),
                }
            )
        ranked.sort(key=lambda row: row["mean_score_1_5"], reverse=True)
        for index, row in enumerate(ranked, start=1):
            row["rank"] = index
            out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fallback_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else fallback_fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    return f"{float(value):.4f}"


def render_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Rank | Model | Score 1-5 | LSE-C | LSE-D | Abs Offset | Videos | Windows |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['model']} | {fmt(row['mean_score_1_5'])} | "
            f"{fmt(row['mean_lse_c'])} | {fmt(row['mean_lse_d'])} | "
            f"{fmt(row['mean_abs_offset_frames'])} | {row['scored_videos']} | {row['kept_windows']} |"
        )
    return lines


def write_md(path: Path, ranking: list[dict[str, Any]], detail_rows: list[dict[str, Any]], excluded: set[tuple[str, str]]) -> None:
    by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranking:
        by_scope[row["scope"]].append(row)
    stable = [row for row in detail_rows if row["stable_frontal"]]
    reasons = Counter(str(row["audit_reason"]) for row in detail_rows)
    lines = [
        "# Stable Frontal Face Lowest LSE-D Rankings",
        "",
        f"- Base input/statistics excludes: `{', '.join(f'{m}/{s}' for m, s in sorted(excluded)) or 'none'}`.",
        "- Base video/window set: only VSLS scored videos and scored speech windows are considered.",
        "- Candidate filter: `abs(offset_frames) < 10`.",
        "- Multi-person track selection: choose the candidate track with the lowest `LSE-D`; tie-break by lower `abs(offset_frames)`, then higher `LSE-C`.",
        "- Score formula after track selection: `Score_0_1 = 0.60*C_norm + 0.40*D_norm`, `Score_1_5 = 1 + 4*Score_0_1`.",
        "- Stable frontal heuristic on selected crop: sample 8 frames; frontal-face hit rate >= 75%; mean face width >= 28% of crop; center std <= 0.12; size CV <= 0.35.",
        "- Aggregation: video-macro after filtering to stable frontal crop windows only.",
        "",
        "## Filter Counts",
        "",
        f"- Candidate windows after `abs(offset)<10`: {len(detail_rows)}",
        f"- Stable frontal crop windows: {len(stable)}",
        f"- Stable frontal videos: {len({(row['model'], row['sample_id']) for row in stable})}",
        f"- Audit reason counts: `{dict(reasons)}`",
        "",
    ]
    for scope in SCOPES:
        title = "Overall" if scope == "overall" else scope
        lines.extend([f"## {title}", ""])
        rows = by_scope.get(scope, [])
        if rows:
            lines.extend(render_table(rows))
        else:
            lines.append("No stable frontal scored windows.")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    excluded = parse_excluded(args.excluded)
    metadata = load_metadata(args.metadata_csv, args.id_column)
    detail_rows = build_detail_rows(args.results_dir.expanduser().resolve(), metadata, args.model_column, excluded)
    ranking = aggregate(detail_rows)
    write_csv(
        args.detail_csv.expanduser().resolve(),
        detail_rows,
        [
            "model",
            "sample_id",
            "board",
            "segment_id",
            "lse_c",
            "lse_d",
            "abs_offset_frames",
            "score_0_1",
            "score_1_5",
            "stable_frontal",
            "audit_reason",
        ],
    )
    write_csv(
        args.output_csv.expanduser().resolve(),
        ranking,
        [
            "scope",
            "rank",
            "model",
            "mean_score_1_5",
            "mean_score_0_1",
            "mean_lse_c",
            "mean_lse_d",
            "mean_abs_offset_frames",
            "scored_videos",
            "kept_windows",
        ],
    )
    write_md(args.output_md.expanduser().resolve(), ranking, detail_rows, excluded)
    print(f"Saved stable frontal ranking to {args.output_md}")


if __name__ == "__main__":
    main()
