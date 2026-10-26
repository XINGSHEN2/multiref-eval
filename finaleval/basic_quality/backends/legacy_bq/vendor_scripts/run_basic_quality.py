#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def run(cmd: list[str], cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_bash(script: str, cwd: Path) -> None:
    print("+ bash -lc " + script, flush=True)
    subprocess.run(["bash", "-lc", script], cwd=str(cwd), check=True)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def key_from_result(item: dict) -> str | None:
    filename = item.get("filename")
    if filename:
        return Path(filename).stem
    file_path = item.get("file")
    if file_path:
        return Path(file_path).stem
    return None


def index_results(payload: dict) -> dict[str, dict]:
    indexed = {}
    for item in payload.get("results", []):
        key = key_from_result(item)
        if key:
            indexed[key] = item
    return indexed


def write_dover_plus_config(source: Path, destination: Path, weight_path: Path) -> None:
    text = source.read_text(encoding="utf-8")
    lines = []
    replaced = False
    for line in text.splitlines():
        if line.strip().startswith("test_load_path:"):
            lines.append(f"test_load_path: {weight_path}")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        lines.append(f"test_load_path: {weight_path}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_video_stems(video_dir: Path) -> list[tuple[str, Path]]:
    videos = sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise FileNotFoundError(f"No videos found in {video_dir}")
    return [(video.stem, video) for video in videos]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    code_root = script_dir.parent
    default_output = code_root / "Output" / "basic_quality"

    parser = argparse.ArgumentParser(description="Run Basic Quality objective metrics and merge them into one CSV")
    parser.add_argument("input_dir", help="Directory containing generated videos")
    parser.add_argument("output_dir", nargs="?", default=str(default_output), help="Directory for JSON outputs and merged CSV")
    parser.add_argument("--num-aesthetic-frames", type=int, default=10)
    parser.add_argument("--num-temporal-frames", type=int, default=32)
    parser.add_argument("--audio-tool", choices=["audiobox", "nisqa", "both", "none"], default="audiobox")
    parser.add_argument("--skip-aesthetic", action="store_true", help="Skip Aesthetic Predictor V2.5")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = collect_video_stems(input_dir)

    os.environ.setdefault("BQ_USE_ACTIVE_ENV", "1")
    os.environ.setdefault("PYTHON_BIN", sys.executable)

    common = script_dir / "common.sh"
    dover_dir = code_root / "Objective" / "Video" / "DOVER"
    dover_config = dover_dir / "dover.yml"
    dover_plus_weight = dover_dir / "pretrained_weights" / "DOVER_plus_plus.pth"
    dover_plus_config = output_dir / "dover_plus_plus.yml"

    video_technical_json = output_dir / "video_technical_dover_plus_plus.json"
    video_aesthetic_json = output_dir / "video_aesthetic.json"
    temporal_json = output_dir / "temporal_stability.json"
    audio_aesthetic_json = output_dir / "audio_aesthetic.json"
    speech_quality_json = output_dir / "speech_quality.json"

    if not args.skip_existing or not video_technical_json.exists():
        run_bash(
            f"source {common}; ensure_cache_layout; ensure_conda; ensure_dover_env",
            code_root,
        )
        if not dover_plus_weight.exists():
            raise FileNotFoundError(f"DOVER++ weight not found after setup: {dover_plus_weight}")
        write_dover_plus_config(dover_config, dover_plus_config, dover_plus_weight)
        run_bash(
            f"source {common}; ensure_conda; conda_run_in basic-quality-dover python {script_dir / 'batch_dover.py'} "
            f"--input {input_dir} --output {video_technical_json} --config {dover_plus_config} --device cuda",
            code_root,
        )

    if not args.skip_aesthetic and (not args.skip_existing or not video_aesthetic_json.exists()):
        run(["bash", str(script_dir / "eval_video_aesthetic.sh"), str(input_dir), str(output_dir), str(args.num_aesthetic_frames)], code_root)

    if not args.skip_existing or not temporal_json.exists():
        run_bash(
            f"source {common}; ensure_cache_layout; ensure_conda; ensure_core_env; "
            f"conda_run_in $BQ_CORE_ENV python {script_dir / 'batch_temporal_stability.py'} "
            f"--video_dir {input_dir} --output {temporal_json} --num_frames {args.num_temporal_frames}",
            code_root,
        )

    if args.audio_tool in {"audiobox", "both"} and (not args.skip_existing or not audio_aesthetic_json.exists()):
        run(["bash", str(script_dir / "eval_audio_aesthetic.sh"), str(input_dir), str(output_dir)], code_root)

    if args.audio_tool in {"nisqa", "both"} and (not args.skip_existing or not speech_quality_json.exists()):
        run(["bash", str(script_dir / "eval_speech_quality.sh"), str(input_dir), str(output_dir)], code_root)

    technical = index_results(load_json(video_technical_json))
    aesthetic = {} if args.skip_aesthetic else index_results(load_json(video_aesthetic_json))
    temporal = index_results(load_json(temporal_json))
    audiobox = index_results(load_json(audio_aesthetic_json))
    nisqa = index_results(load_json(speech_quality_json))

    rows = []
    for stem, video_path in videos:
        tech = technical.get(stem, {})
        aes = aesthetic.get(stem, {})
        temp = temporal.get(stem, {})
        abox = audiobox.get(stem, {})
        nisqa_row = nisqa.get(stem, {})
        abox_scores = abox.get("scores", {})

        audio_quality = ""
        audio_source = ""
        if isinstance(abox_scores, dict) and abox_scores.get("PQ") not in (None, ""):
            audio_quality = abox_scores.get("PQ")
            audio_source = "audiobox_pq"
        elif nisqa_row.get("mos_pred") not in (None, ""):
            audio_quality = nisqa_row.get("mos_pred")
            audio_source = "nisqa_mos"

        row = {
            "visual_technical_quality_doverpp": tech.get("technical", ""),
            "temporal_stability": temp.get("temporal_stability", ""),
            "audio_technical_quality": audio_quality,
            "filename": video_path.name,
            "video_path": str(video_path),
            "frame_similarity_mean": temp.get("frame_similarity_mean", ""),
            "luma_flicker_mean": temp.get("luma_flicker_mean", ""),
            "histogram_diff_mean": temp.get("histogram_diff_mean", ""),
            "flow_magnitude_mean": temp.get("flow_magnitude_mean", ""),
            "flow_jitter": temp.get("flow_jitter", ""),
            "audio_technical_source": audio_source,
            "audiobox_PQ": abox_scores.get("PQ", "") if isinstance(abox_scores, dict) else "",
            "audiobox_CU": abox_scores.get("CU", "") if isinstance(abox_scores, dict) else "",
            "audiobox_CE": abox_scores.get("CE", "") if isinstance(abox_scores, dict) else "",
            "audiobox_PC": abox_scores.get("PC", "") if isinstance(abox_scores, dict) else "",
            "nisqa_mos_pred": nisqa_row.get("mos_pred", ""),
            "nisqa_noi_pred": nisqa_row.get("noi_pred", ""),
            "nisqa_dis_pred": nisqa_row.get("dis_pred", ""),
            "nisqa_col_pred": nisqa_row.get("col_pred", ""),
            "nisqa_loud_pred": nisqa_row.get("loud_pred", ""),
        }
        if not args.skip_aesthetic:
            row.update(
                {
                    "visual_aesthetic_quality_v2_5": aes.get("score", ""),
                    "doverpp_aesthetic_aux": tech.get("aesthetic", ""),
                    "doverpp_overall_aux": tech.get("overall", ""),
                    "aesthetic_num_frames": aes.get("num_frames", ""),
                    "aesthetic_frame_scores": json.dumps(aes.get("frame_scores", []), ensure_ascii=False),
                }
            )
        rows.append(row)

    csv_path = output_dir / "basic_quality.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "basic_quality_summary.json"
    metrics = {
        "visual_technical_quality": "DOVER++ technical",
        "temporal_stability": "OpenCV frame cosine similarity with optical-flow/flicker diagnostics",
        "audio_technical_quality": args.audio_tool,
    }
    json_outputs = {
        "video_technical": str(video_technical_json),
        "temporal_stability": str(temporal_json),
        "audio_aesthetic": str(audio_aesthetic_json) if audio_aesthetic_json.exists() else None,
        "speech_quality": str(speech_quality_json) if speech_quality_json.exists() else None,
    }
    if not args.skip_aesthetic:
        metrics["visual_aesthetic_quality"] = "Aesthetic Predictor V2.5 sampled-frame mean"
        json_outputs["video_aesthetic"] = str(video_aesthetic_json)

    summary = {
        "input_dir": str(input_dir),
        "output_csv": str(csv_path),
        "metrics": metrics,
        "json_outputs": json_outputs,
        "total_samples": len(rows),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved merged CSV to {csv_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
