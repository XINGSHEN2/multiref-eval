from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from finaleval.common.output_writer import write_metric_outputs
from finaleval.common.schema import InputSample


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "finaleval"
BASIC_QUALITY_BACKEND = PACKAGE_ROOT / "basic_quality" / "backends" / "legacy_bq"
ENTITY_FIDELITY_BACKEND = PACKAGE_ROOT / "entity_fidelity" / "backends" / "legacy_ef"
AUDIO_VIDEO_BACKEND = PACKAGE_ROOT / "audio_video_consistency" / "backends" / "legacy_ac"
INSTRUCTION_FOLLOWING_BACKEND = PACKAGE_ROOT / "instruction_following" / "backends" / "legacy_if"


def write_manifest(samples: list[InputSample], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "board",
        "prompt",
        "model_name",
        "video_path",
        "chinese_prompt",
        "has_speech",
        "has_sound",
        "has_music",
        "reference_audio_path",
        "reference_image_paths",
        "reference_video_path",
        "reference_json_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({field: getattr(sample, field) for field in fields})
    return path


def sample_error_rows(samples: list[InputSample], metric_name: str, reason: str) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample.sample_id,
            "model_name": sample.model_name,
            "video_path": sample.video_path,
            "metric_name": metric_name,
            "score": "",
            "status": "error",
            "reason": reason,
            "raw_score": "",
        }
        for sample in samples
    ]


def run_command(cmd: list[str], cwd: Path, env: dict[str, str] | None = None, timeout: int | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    merged_env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    subprocess.run(
        [str(item) for item in cmd],
        cwd=str(cwd),
        env=merged_env,
        check=True,
        timeout=timeout,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_csv(root: Path, preferred_names: list[str], contains: list[str] | None = None) -> Path | None:
    if not root.exists():
        return None
    for name in preferred_names:
        hits = sorted(root.rglob(name), key=lambda item: item.stat().st_mtime, reverse=True)
        if hits:
            return hits[0]
    candidates = sorted(root.rglob("*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)
    if contains:
        lowered = [item.lower() for item in contains]
        for path in candidates:
            text = path.name.lower()
            if all(item in text for item in lowered):
                return path
    return candidates[0] if candidates else None


def first(row: dict[str, Any], names: list[str]) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def is_false_value(value: object) -> bool:
    return str(value).strip().lower() in {"false", "0", "no", "n"}


def is_not_applicable(source: dict[str, Any], metric_name: str) -> bool:
    metric_title = metric_name.replace("_", " ").title()
    keys = [
        "applicable",
        f"{metric_title}_applicable",
        f"{metric_name}_applicable",
    ]
    return any(key in source and is_false_value(source.get(key)) for key in keys)


def normalize_csv_rows(
    samples: list[InputSample],
    metric_name: str,
    source_csv: Path,
    score_fields: list[str],
    extra_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    fallback_by_sample: dict[str, dict[str, str]] = {}
    fallback_by_video_model: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(source_csv):
        sample_id = first(row, ["sample_id", "sample", "id", "task_id"])
        model_name = first(row, ["model_name", "model"])
        video_path = first(row, ["video_path", "generated_video_path", "path"])
        if sample_id and model_name:
            by_key[(sample_id, model_name)] = row
        if sample_id:
            fallback_by_sample[sample_id] = row
        if video_path and model_name:
            fallback_by_video_model[(str(Path(video_path).expanduser().resolve()), model_name)] = row

    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_video = str(Path(sample.video_path).expanduser().resolve())
        source = (
            by_key.get((sample.sample_id, sample.model_name))
            or fallback_by_sample.get(sample.sample_id)
            or fallback_by_video_model.get((sample_video, sample.model_name))
        )
        if not source:
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "model_name": sample.model_name,
                    "video_path": sample.video_path,
                    "metric_name": metric_name,
                    "score": "",
                    "status": "error",
                    "reason": f"No legacy result row found in {source_csv.name}.",
                    "raw_score": "",
                }
            )
            continue
        score = first(source, score_fields)
        not_applicable = score == "" and is_not_applicable(source, metric_name)
        out: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "model_name": sample.model_name,
            "video_path": sample.video_path,
            "metric_name": metric_name,
            "score": score,
            "status": "scored" if score != "" else ("skipped" if not_applicable else "error"),
            "reason": first(source, ["reason", "error", "failure_reason"]),
            "raw_score": score,
            "legacy_source_csv": str(source_csv),
        }
        for field in extra_fields or []:
            out[field] = source.get(field, "")
        rows.append(out)
    return rows


def write_backend_error(
    samples: list[InputSample],
    output_root: Path,
    metric_name: str,
    group: str,
    error: BaseException,
    backend: str,
) -> Path:
    reason = f"{backend} backend failed: {error}"
    return write_metric_outputs(
        output_root,
        metric_name,
        sample_error_rows(samples, metric_name, reason),
        details={"metric": metric_name, "group": group, "implemented": True, "backend": backend, "error": reason},
    )


def python_bin() -> str:
    return os.getenv("PYTHON_BIN") or sys.executable


def run_shell_metric(
    samples: list[InputSample],
    output_root: Path,
    metric_name: str,
    group: str,
    backend_name: str,
    cwd: Path,
    command: list[str],
    score_fields: list[str],
    preferred_csv_names: list[str],
    env: dict[str, str] | None = None,
    extra_fields: list[str] | None = None,
) -> Path:
    legacy_run = output_root / "_legacy" / metric_name
    manifest = write_manifest(samples, legacy_run / "input_manifest.csv")
    merged_env = {
        "PYTHON_BIN": python_bin(),
        "PYTHONPATH": str(PROJECT_ROOT),
        "CSV": str(manifest),
        "INPUT_CSV": str(manifest),
        "OUTPUT_DIR": str(legacy_run),
        "OUT_DIR": str(legacy_run),
        "RUN_NAME": metric_name,
        "BOARD": "all",
        "MODELS": ",".join(sorted({sample.model_name for sample in samples if sample.model_name})) or "all",
    }
    if env:
        merged_env.update(env)
    try:
        run_command(command, cwd=cwd, env=merged_env)
        source_csv = find_csv(legacy_run, preferred_csv_names, contains=[metric_name])
        if source_csv is None:
            source_csv = find_csv(cwd, preferred_csv_names, contains=[metric_name])
        if source_csv is None:
            raise FileNotFoundError(f"No legacy CSV found for {metric_name}.")
        rows = normalize_csv_rows(samples, metric_name, source_csv, score_fields, extra_fields=extra_fields)
        return write_metric_outputs(
            output_root,
            metric_name,
            rows,
            details={
                "metric": metric_name,
                "group": group,
                "implemented": True,
                "backend": backend_name,
                "source_csv": str(source_csv),
            },
            extra_summary_fields=extra_fields,
        )
    except Exception as exc:
        return write_backend_error(samples, output_root, metric_name, group, exc, backend_name)
