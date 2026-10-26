#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from google.oauth2 import service_account

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TASKS = SCRIPT_DIR / "inputs/if_eval_tasks.json"
DEFAULT_PROMPT = SCRIPT_DIR / "if_gemini_judge_prompt_v2.md"
DEFAULT_OUTPUTS_DIR = SCRIPT_DIR / "outputs"
DEFAULT_CREDENTIALS = Path(os.getenv("FINALEVAL_GEMINI_CREDENTIALS", ""))
DEFAULT_MODEL = os.getenv("IF_EVAL_MODEL") or os.getenv("RC_EVAL_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-3.1-pro-preview"
DEFAULT_VIDEO_FPS = float(os.getenv("IF_GEMINI_VIDEO_FPS", "4"))

MLLM_DIMENSIONS = [
    "Visual Task Following",
    "Audio Task Following",
    "Temporal Order Following",
]
ALL_DIMENSIONS = [
    "Visual Task Following",
    "Audio Task Following",
    "Speech Content Accuracy",
    "Temporal Order Following",
]
CHECKLIST_KEY_BY_DIMENSION = {
    "Visual Task Following": "visual_task_following",
    "Audio Task Following": "audio_task_following",
    "Speech Content Accuracy": "speech_prompt",
    "Temporal Order Following": "temporal_order_following",
}
SKIPPED_DIMENSIONS_BY_MODEL = {
    "foley": {"Audio Task Following", "Speech Content Accuracy"},
    "omni": {"Audio Task Following", "Speech Content Accuracy"},
    "reference2v": {"Audio Task Following", "Speech Content Accuracy"},
    "skyreels-v3": {"Audio Task Following", "Speech Content Accuracy"},
    "skyreels-official-r2v": {"Audio Task Following", "Speech Content Accuracy"},
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_row(path: Path, row: dict[str, Any]) -> None:
    rows = read_rows(path)
    rows.append(row)
    write_rows(path, rows)


def parse_csv_set(value: str | None) -> set[str] | None:
    if not value or value.lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def load_project_id(credentials_path: Path) -> str:
    data = json.loads(credentials_path.read_text(encoding="utf-8"))
    project_id = data.get("project_id")
    if not project_id:
        raise RuntimeError(f"Missing project_id in credentials file: {credentials_path}")
    return project_id


def build_client(credentials_path: Path, location: str):
    credentials = service_account.Credentials.from_service_account_file(
        str(credentials_path),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return genai.Client(
        vertexai=True,
        credentials=credentials,
        project=load_project_id(credentials_path),
        location=location,
    )


def file_part(path: str):
    file_path = Path(path)
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    data = file_path.read_bytes()
    if mime_type.startswith("video/"):
        return types.Part(
            inline_data=types.Blob(data=data, mime_type=mime_type),
            video_metadata=types.VideoMetadata(fps=DEFAULT_VIDEO_FPS),
        )
    return types.Part.from_bytes(data=data, mime_type=mime_type)


def build_prompt(prompt_template: str, spec: dict[str, Any]) -> str:
    spec_json = json.dumps(spec, ensure_ascii=False, indent=2)
    return prompt_template.replace("{if_eval_spec_json}", spec_json)


def evaluation_spec(spec: dict[str, Any], dimensions: set[str] | None) -> dict[str, Any]:
    if dimensions is None:
        selected = set(ALL_DIMENSIONS)
    else:
        selected = set(dimensions)
    applicable = spec.get("applicable_dimensions", {})
    checklist = spec.get("checklist", {})
    return {
        "sample_id": spec.get("sample_id"),
        "applicable_dimensions": {
            dimension: bool(applicable.get(dimension))
            for dimension in ALL_DIMENSIONS
            if dimension in selected
        },
        "checklist": {
            CHECKLIST_KEY_BY_DIMENSION[dimension]: checklist.get(
                CHECKLIST_KEY_BY_DIMENSION[dimension], []
            )
            for dimension in ALL_DIMENSIONS
            if dimension in selected
        },
    }


def skipped_model_dimensions(task: dict[str, Any]) -> set[str]:
    return SKIPPED_DIMENSIONS_BY_MODEL.get(str(task.get("model_name", "")).lower(), set())


def skipped_score(dim: str, model_name: str) -> dict[str, Any]:
    return {
        "applicable": False,
        "score": None,
        "reason": f"{dim} is skipped for model {model_name}.",
        "item_results": [],
    }


def task_for_model_dimension_skips(task: dict[str, Any]) -> dict[str, Any]:
    skipped = skipped_model_dimensions(task)
    if not skipped:
        return task
    task_copy = dict(task)
    spec = dict(task["if_eval_spec"])
    applicable = dict(spec.get("applicable_dimensions", {}))
    checklist = dict(spec.get("checklist", {}))
    for dim in skipped:
        applicable[dim] = False
    if "Audio Task Following" in skipped:
        checklist["audio_task_following"] = []
    if "Speech Content Accuracy" in skipped:
        checklist["speech_prompt"] = []
    spec["applicable_dimensions"] = applicable
    spec["checklist"] = checklist
    task_copy["if_eval_spec"] = spec
    return task_copy


def build_contents(task: dict[str, Any], prompt_text: str, include_references: bool = True) -> list[Any]:
    spec = task["if_eval_spec"]
    contents: list[Any] = [prompt_text]

    if include_references:
        for ref in spec.get("references", []):
            ref_path = ref.get("path")
            if not ref_path or not Path(ref_path).exists():
                continue
            contents.append(f"Reference image {ref.get('placeholder')} ({ref.get('type')}): {ref_path}")
            contents.append(file_part(ref_path))

    video_path = task.get("video_path")
    contents.append("Generated video:")
    if video_path and Path(video_path).exists():
        contents.append(file_part(video_path))
    else:
        contents.append("WARNING: Generated video file is missing or unreadable.")
    return contents


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def normalize_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score < 1 or score > 5:
        return None
    return score


def normalize_item_results(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        if "score" in row:
            if row.get("applicable") is False:
                row["score"] = None
            else:
                row["score"] = normalize_score(row.get("score"))
        if "result" in row:
            result = str(row["result"]).strip().lower()
            row["result"] = result if result in {"yes", "no", "partial"} else None
        out.append(row)
    return out


def average_item_score(item_results: list[dict[str, Any]]) -> float | None:
    values = []
    for item in item_results:
        if item.get("applicable") is False:
            continue
        score = normalize_score(item.get("score"))
        if score is not None:
            values.append(score)
    return float(math.floor(sum(values) / len(values))) if values else None


def placeholder_speech_result(applicable: bool) -> dict[str, Any]:
    return {
        "applicable": applicable,
        "score": None,
        "reason": "Run the optional ASR/text-matching automatic metric to fill this score.",
        "method": "asr_text_match_placeholder",
        "item_results": [],
    }


def normalize_result(payload: dict[str, Any], task: dict[str, Any], dimensions: set[str] | None) -> dict[str, Any]:
    spec = task["if_eval_spec"]
    applicable = spec.get("applicable_dimensions", {})
    skipped = skipped_model_dimensions(task)
    model_name = str(task.get("model_name", ""))
    scores: dict[str, Any] = {}

    for dim in MLLM_DIMENSIONS:
        if dim in skipped:
            scores[dim] = skipped_score(dim, model_name)
            continue
        if dimensions is not None and dim not in dimensions:
            scores[dim] = {"applicable": False, "score": None, "reason": "Not evaluated in this run."}
            continue
        raw = payload.get(dim, {})
        if not isinstance(raw, dict):
            raw = {}
        dim_applicable = bool(raw.get("applicable", applicable.get(dim, True)))
        item_results = normalize_item_results(raw.get("item_results", []))
        item_score = average_item_score(item_results)
        raw_score = normalize_score(raw.get("score"))
        score = item_score if dim_applicable and item_score is not None else raw_score if dim_applicable else None
        scores[dim] = {
            "applicable": dim_applicable,
            "score": score,
            "reason": str(raw.get("reason", "")),
            "item_results": item_results,
        }

    speech_applicable = bool(applicable.get("Speech Content Accuracy"))
    if "Speech Content Accuracy" in skipped:
        scores["Speech Content Accuracy"] = skipped_score("Speech Content Accuracy", model_name)
    elif dimensions is None or "Speech Content Accuracy" in dimensions:
        scores["Speech Content Accuracy"] = placeholder_speech_result(speech_applicable)
    else:
        scores["Speech Content Accuracy"] = {"applicable": False, "score": None, "reason": "Not evaluated in this run."}

    if_values = []
    for dim in ALL_DIMENSIONS:
        row = scores.get(dim, {})
        if row.get("applicable") and row.get("score") is not None:
            if_values.append(float(row["score"]))
    scores["Instruction Following"] = {
        "applicable": bool(if_values),
        "score": sum(if_values) / len(if_values) if if_values else None,
        "reason": "Average over applicable dimensions with available scores.",
    }
    return scores


def dry_run_result(task: dict[str, Any], dimensions: set[str] | None) -> dict[str, Any]:
    effective_task = task_for_model_dimension_skips(task)
    payload = {}
    applicable = effective_task["if_eval_spec"].get("applicable_dimensions", {})
    for dim in MLLM_DIMENSIONS:
        payload[dim] = {
            "applicable": bool(applicable.get(dim)),
            "score": None,
            "reason": "Dry run: Gemini was not called.",
            "item_results": [],
        }
    return result_row(task, normalize_result(payload, effective_task, dimensions), payload, None)


def result_row(task: dict[str, Any], scores: dict[str, Any], raw_response: Any, error: str | None) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "sample_id": task["sample_id"],
        "model_name": task["model_name"],
        "video_path": task["video_path"],
        "source_json": task["if_eval_spec"].get("source_json"),
        "if_eval_spec": task["if_eval_spec"],
        "scores": scores,
        "raw_response": raw_response,
        "error": error,
    }


def evaluate_task(
    client,
    task: dict[str, Any],
    prompt_template: str,
    model: str,
    timeout: int,
    temperature: float,
    dimensions: set[str] | None,
) -> dict[str, Any]:
    effective_task = task_for_model_dimension_skips(task)
    request_spec = evaluation_spec(effective_task["if_eval_spec"], dimensions)
    prompt_text = build_prompt(prompt_template, request_spec)
    include_references = dimensions is None or "Visual Task Following" in dimensions
    contents = build_contents(effective_task, prompt_text, include_references=include_references)
    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                    http_options=types.HttpOptions(timeout=int(timeout * 1000)),
                ),
            )
            payload = parse_json_response(response.text or "{}")
            return result_row(task, normalize_result(payload, effective_task, dimensions), payload, None)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(2 * (attempt + 1))
    return result_row(task, {}, None, last_error or "unknown error")


def filter_tasks(tasks: list[dict[str, Any]], models: set[str] | None, samples: set[str] | None, limit: int | None) -> list[dict[str, Any]]:
    filtered = []
    for task in tasks:
        if models and task.get("model_name") not in models:
            continue
        if samples and task.get("sample_id") not in samples:
            continue
        filtered.append(task)
        if limit and len(filtered) >= limit:
            break
    return filtered


def build_output_path(output: str | None, output_dir: str, run_name: str | None) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{run_name}_{timestamp}" if run_name else f"if_{timestamp}"
    return Path(output_dir).expanduser().resolve() / folder_name / "if_results.json"


def write_latest_run(output_path: Path) -> None:
    latest_path = DEFAULT_OUTPUTS_DIR / "latest_run.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(output_path.parent) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Instruction Following with Gemini MLLM checklist judge.")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUTS_DIR))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--location", default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--models", default="all")
    parser.add_argument("--samples", default="all")
    parser.add_argument("--dimensions", default="all")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write placeholder rows without calling Gemini.")
    args = parser.parse_args()

    dimensions = parse_csv_set(args.dimensions)
    unknown = (dimensions or set()) - set(ALL_DIMENSIONS)
    if unknown:
        raise ValueError(f"Unknown dimensions: {', '.join(sorted(unknown))}")

    tasks = filter_tasks(
        read_rows(Path(args.tasks).expanduser().resolve()),
        parse_csv_set(args.models),
        parse_csv_set(args.samples),
        args.limit,
    )
    if not tasks:
        raise ValueError("No tasks to evaluate.")

    output = build_output_path(args.output, args.output_dir, args.run_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = [] if args.overwrite else read_rows(output)
    done = {row.get("task_id") for row in existing if not row.get("error")}
    pending = [task for task in tasks if task["task_id"] not in done]
    if args.overwrite:
        write_rows(output, [])

    print(f"Tasks total: {len(tasks)}; already done: {len(done)}; pending: {len(pending)}")
    print(f"Output: {output}")
    print(f"Dry run: {args.dry_run}")
    write_latest_run(output)

    if not pending:
        return

    if args.dry_run:
        for task in pending:
            append_row(output, dry_run_result(task, dimensions))
        return

    prompt_template = Path(args.prompt).expanduser().resolve().read_text(encoding="utf-8")
    credentials_path = args.credentials.expanduser().resolve()
    if not credentials_path.exists():
        raise FileNotFoundError(f"Gemini credentials file not found: {credentials_path}")
    client = build_client(credentials_path, args.location)
    print(f"Judge model: {args.model}")
    print("Credentials: configured")
    print(f"Location: {args.location}")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(evaluate_task, client, task, prompt_template, args.model, args.timeout, args.temperature, dimensions)
            for task in pending
        ]
        completed = as_completed(futures)
        iterator = tqdm(completed, total=len(futures), desc="Instruction Following") if tqdm else completed
        for future in iterator:
            append_row(output, future.result())


if __name__ == "__main__":
    main()
