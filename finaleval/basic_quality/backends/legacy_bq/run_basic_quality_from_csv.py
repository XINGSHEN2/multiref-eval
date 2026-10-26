#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(str(item) for item in cmd), flush=True)
    subprocess.run([str(item) for item in cmd], cwd=str(cwd) if cwd else None, check=True)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Input CSV is empty: {path}")
    return rows


def first_present(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and value.strip():
            return value.strip()
    return ""


def normalize_rows(rows: list[dict[str, str]], video_col: str, sample_col: str, model_col: str) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        video = first_present(row, [video_col, "video_path", "source_video", "path", "file"])
        if not video:
            raise ValueError(f"Row {index} has no video path column: {row}")
        video_path = Path(video).expanduser().resolve()
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Row {index} video has unsupported extension: {video_path}")
        if not video_path.exists():
            raise FileNotFoundError(f"Row {index} video does not exist: {video_path}")

        sample = first_present(row, [sample_col, "sample", "sample_id", "id"])
        model = first_present(row, [model_col, "model", "model_name"])
        if not sample or not model:
            stem = video_path.stem
            if "__" in stem:
                inferred_sample, inferred_model = stem.split("__", 1)
                sample = sample or inferred_sample
                model = model or inferred_model
        if not sample or not model:
            raise ValueError(f"Row {index} must provide sample and model columns: {row}")

        normalized.append(
            {
                "sample": sample,
                "model": model,
                "source_video": str(video_path),
                "input_row": str(index),
            }
        )
    return normalized


def infer_models(rows: list[dict[str, str]]) -> list[str]:
    seen: dict[str, None] = {}
    for row in rows:
        seen.setdefault(row["model"], None)
    return sorted(seen)


def sample_counts_by_model(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["model"]] = counts.get(row["model"], 0) + 1
    return counts


def validate_expected(rows: list[dict[str, str]], models: list[str], expected_samples_per_model: int | None) -> None:
    counts = {model: 0 for model in models}
    unexpected: dict[str, int] = {}
    for row in rows:
        model = row["model"]
        if model in counts:
            counts[model] += 1
        else:
            unexpected[model] = unexpected.get(model, 0) + 1
    missing = [model for model, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"Missing expected models: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected models in CSV: {unexpected}")
    bad_counts = {
        model: count
        for model, count in counts.items()
        if expected_samples_per_model is not None and count != expected_samples_per_model
    }
    if bad_counts:
        raise ValueError(
            f"Expected {expected_samples_per_model} rows per model; got {bad_counts}. "
            "Use --expected-samples-per-model auto or adjust --models to match the input CSV."
        )


def prepare_batch_input(rows: list[dict[str, str]], output_dir: Path) -> tuple[Path, Path]:
    batch_input = output_dir / "_batch_input"
    manifest_csv = output_dir / "batch_manifest.csv"
    if batch_input.exists():
        shutil.rmtree(batch_input)
    batch_input.mkdir(parents=True, exist_ok=True)

    prepared: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for row in rows:
        source = Path(row["source_video"])
        link_name = f"{row['sample']}__{row['model']}{source.suffix.lower()}"
        if link_name in seen_names:
            raise ValueError(f"Duplicate sample/model pair creates duplicate batch filename: {link_name}")
        seen_names.add(link_name)
        link_path = batch_input / link_name
        os.symlink(source, link_path)
        prepared.append(
            {
                **row,
                "batch_filename": link_name,
                "batch_video": str(link_path),
            }
        )

    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample", "model", "source_video", "batch_filename", "batch_video", "input_row"],
        )
        writer.writeheader()
        writer.writerows(prepared)
    return batch_input, manifest_csv


def merge_metric_csv(metric_csv: Path, manifest_csv: Path, output_csv: Path) -> None:
    manifest: dict[str, dict[str, str]] = {}
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            manifest[Path(row["batch_filename"]).stem] = row

    with metric_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        metric_fields = list(reader.fieldnames or [])
        output_fields = ["sample", "model", "source_video", *metric_fields]
        rows: list[dict[str, str]] = []
        for row in reader:
            stem = Path(row.get("filename", "")).stem
            info = manifest.get(stem, {})
            source_video = info.get("source_video", row.get("video_path", ""))
            row["filename"] = Path(source_video).name if source_video else row.get("filename", "")
            row["video_path"] = source_video or row.get("video_path", "")
            rows.append(
                {
                    "sample": info.get("sample", stem.split("__", 1)[0] if "__" in stem else ""),
                    "model": info.get("model", stem.split("__", 1)[1] if "__" in stem else ""),
                    "source_video": source_video,
                    **row,
                }
            )

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    here = Path(__file__).resolve().parent
    basic_quality_root = here

    parser = argparse.ArgumentParser(description="Run Basic Quality metrics from a CSV manifest.")
    parser.add_argument("--input-csv", required=True, help="CSV with video_path/source_video, sample, model columns.")
    parser.add_argument("--output-dir", required=True, help="Output run directory.")
    parser.add_argument("--video-col", default="video_path", help="Preferred video path column.")
    parser.add_argument("--sample-col", default="sample", help="Preferred sample id column.")
    parser.add_argument("--model-col", default="model", help="Preferred model name column.")
    parser.add_argument(
        "--models",
        default="auto",
        help="Comma-separated expected models, or 'auto' to infer from input CSV. Default: auto.",
    )
    parser.add_argument(
        "--expected-samples-per-model",
        default="auto",
        help="Expected rows per model, or 'auto' to use the input CSV as-is. Default: auto.",
    )
    parser.add_argument(
        "--no-default-validation",
        action="store_true",
        help="Deprecated compatibility flag; validation is automatic by default.",
    )
    parser.add_argument("--num-temporal-frames", type=int, default=32)
    parser.add_argument("--temporal-sample-fps", type=float, default=0.0, help="Sample temporal metric at this real-time FPS instead of fixed frame count.")
    parser.add_argument("--audio-tool", choices=["audiobox", "nisqa", "both", "none"], default="audiobox")
    parser.add_argument("--python", default=sys.executable, help="Python executable for running BQ scripts.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing metric JSON/CSV if present.")
    parser.add_argument("--skip-run", action="store_true", help="Only summarize an existing _batch_metrics/basic_quality.csv.")
    args = parser.parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_metrics = output_dir / "_batch_metrics"
    batch_metrics.mkdir(parents=True, exist_ok=True)
    models: list[str] = []

    if not args.skip_run:
        rows = normalize_rows(read_manifest(input_csv), args.video_col, args.sample_col, args.model_col)
        models = infer_models(rows) if args.models.strip().lower() == "auto" else [
            item.strip() for item in args.models.split(",") if item.strip()
        ]
        expected_samples_per_model = None
        if str(args.expected_samples_per_model).strip().lower() != "auto":
            expected_samples_per_model = int(args.expected_samples_per_model)
        if args.models.strip().lower() != "auto" or expected_samples_per_model is not None:
            validate_expected(rows, models, expected_samples_per_model)
        counts = sample_counts_by_model(rows)
        print(
            "[input] "
            + ", ".join(f"{model}:{counts.get(model, 0)}" for model in models),
            flush=True,
        )
        batch_input, manifest_csv = prepare_batch_input(rows, output_dir)
        metric_csv = batch_metrics / "basic_quality.csv"
        if not args.skip_existing or not metric_csv.exists():
            env = os.environ.copy()
            env.setdefault("BQ_USE_ACTIVE_ENV", "1")
            env.setdefault("PYTHON_BIN", args.python)
            cmd = [
                args.python,
                basic_quality_root / "scripts" / "run_basic_quality.py",
                batch_input,
                batch_metrics,
                "--skip-aesthetic",
                "--num-temporal-frames",
                str(args.num_temporal_frames),
                "--temporal-sample-fps",
                str(args.temporal_sample_fps),
                "--audio-tool",
                args.audio_tool,
            ]
            if args.skip_existing:
                cmd.append("--skip-existing")
            print("+ " + " ".join(str(item) for item in cmd), flush=True)
            subprocess.run([str(item) for item in cmd], env=env, check=True)
    else:
        manifest_csv = output_dir / "batch_manifest.csv"
        metric_csv = batch_metrics / "basic_quality.csv"
        if not manifest_csv.exists() or not metric_csv.exists():
            raise FileNotFoundError("--skip-run requires batch_manifest.csv and _batch_metrics/basic_quality.csv")
        with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
            models = infer_models(list(csv.DictReader(handle)))

    merged_csv = output_dir / "metric_details.csv"
    merge_metric_csv(batch_metrics / "basic_quality.csv", manifest_csv, merged_csv)
    run(
        [
            args.python,
            here / "summarize_basic_quality.py",
            "--input",
            merged_csv,
            "--output-dir",
            output_dir,
            "--models",
            ",".join(models),
        ]
    )
    print(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
