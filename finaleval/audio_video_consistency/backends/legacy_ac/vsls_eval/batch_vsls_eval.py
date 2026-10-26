#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from multi_person_syncnet_eval import evaluate_video, load_models, resolve_ffmpeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VSLS evaluation for a CSV of videos.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--latentsync-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ffmpeg-bin", default=None)
    parser.add_argument("--vad-backend", choices=["silero", "webrtcvad", "energy"], default="silero")
    parser.add_argument("--video-column", default="combined_video")
    parser.add_argument("--id-column", default="index")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delete-crops", action="store_true")
    parser.add_argument("--delete-temp", action="store_true", help="Delete standard/syncnet_temp work files.")
    parser.add_argument(
        "--subprocess",
        action="store_true",
        help="Run each video in a fresh subprocess. Default reuses one loaded model pair.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script = Path(__file__).with_name("multi_person_syncnet_eval.py")
    output_dir = args.output_dir.expanduser().resolve()
    results_dir = output_dir / "results"
    work_root = output_dir / "work"
    results_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    with args.input_csv.expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]

    detector = None
    syncnet = None
    ffmpeg_bin = None
    latentsync_dir = args.latentsync_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    if rows and not args.subprocess:
        print(f"[model] Loading VSLS models once on device={args.device}", flush=True)
        detector, syncnet = load_models(latentsync_dir, model_path, args.device)
        ffmpeg_bin = resolve_ffmpeg(args.ffmpeg_bin)
        print("[model] Models loaded; reusing for all videos", flush=True)

    summary = []
    for row_number, row in enumerate(rows):
        video_path = Path(row[args.video_column]).expanduser().resolve()
        item_id = row.get(args.id_column) or f"video_{row_number:05d}"
        safe_id = "".join(char if char.isalnum() or char in "._-" else "_" for char in item_id)
        output_json = results_dir / f"{safe_id}.json"
        work_dir = work_root / safe_id
        print(f"[{row_number + 1}/{len(rows)}] {item_id}", flush=True)
        if args.subprocess:
            command = [
                args.python_bin,
                str(script),
                "--video-path",
                str(video_path),
                "--latentsync-dir",
                str(latentsync_dir),
                "--model-path",
                str(model_path),
                "--work-dir",
                str(work_dir),
                "--output-json",
                str(output_json),
                "--device",
                args.device,
            ]
            if args.ffmpeg_bin:
                command.extend(["--ffmpeg-bin", args.ffmpeg_bin])
            command.extend(["--vad-backend", args.vad_backend])
            if args.delete_crops:
                command.append("--delete-crops")
            if args.delete_temp:
                command.append("--delete-temp")
            completed = subprocess.run(command)
            returncode = completed.returncode
            if output_json.exists():
                result = json.loads(output_json.read_text(encoding="utf-8"))
            else:
                result = {
                    "metric": "Visible Speech-Lip Synchronization",
                    "video_path": str(video_path),
                    "video_status": "error",
                    "video_score_0_1": None,
                    "video_mos_1_5": None,
                    "error": f"evaluation subprocess exited with {returncode} and did not write {output_json}",
                }
        else:
            returncode = 0
            try:
                result = evaluate_video(
                    video_path=video_path,
                    latentsync_dir=latentsync_dir,
                    model_path=model_path,
                    work_dir=work_dir,
                    output_json=output_json,
                    device=args.device,
                    ffmpeg_bin=str(ffmpeg_bin),
                    keep_crops=not args.delete_crops,
                    keep_temp=not args.delete_temp,
                    detector=detector,
                    syncnet=syncnet,
                    vad_backend=args.vad_backend,
                )
            except Exception as exc:  # noqa: BLE001
                returncode = 1
                result = {
                    "metric": "Visible Speech-Lip Synchronization",
                    "video_path": str(video_path),
                    "video_status": "error",
                    "video_score_0_1": None,
                    "video_mos_1_5": None,
                    "error": str(exc),
                }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        result.update(
            {
                "id": item_id,
                "model_name": row.get("model_name"),
                "sample_id": row.get("sample_id"),
                "source_sample_id": row.get("source_sample_id"),
                "video_path": str(video_path),
                "returncode": returncode,
            }
        )
        output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary.append(
            {
                "id": item_id,
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {args.id_column, args.video_column}
                },
                "video_path": str(video_path),
                "returncode": returncode,
                "video_status": result.get("video_status"),
                "video_score_0_1": result.get("video_score_0_1"),
                "video_mos_1_5": result.get("video_mos_1_5"),
                "scored_speech_ratio": result.get("scored_speech_ratio"),
                "result_json": str(output_json),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]) if summary else ["id"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved batch summary to {output_dir}")


if __name__ == "__main__":
    main()
