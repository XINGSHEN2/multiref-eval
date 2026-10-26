#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = (
    SCRIPT_DIR.parent
    / "IF"
    / "outputs"
    / "if_30samples_7models_20260608_035733"
    / "if_results.json"
)
DEFAULT_PROMPT = SCRIPT_DIR / "MLLM_Prompt" / "anatomical knowledge.md"
DEFAULT_OUTPUTS_DIR = SCRIPT_DIR / "outputs" / "anatomical_consistency"
DEFAULT_CREDENTIALS = Path(os.getenv("FINALEVAL_GEMINI_CREDENTIALS", ""))
DEFAULT_MODEL = os.getenv("OIS_EVAL_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-3.1-pro-preview"
DEFAULT_VIDEO_FPS = float(os.getenv("MLLM_GEMINI_VIDEO_FPS", "4"))

DIMENSIONS = [
    "Biological Consistency",
    "Rigid Body Stability",
    "Physical Attachment & Plausibility",
    "3D Scene Integration & Photorealism",
]
OVERALL_DIMENSION = "Object Integrity Score"

DIMENSION_ALIASES = {
    "Biological Consistency": [
        "Biological Anatomical Constraints",
        "Biological / Structural Consistency",
    ],
    "Rigid Body Stability": [
        "Rigid Body Rigidity",
    ],
    "Physical Attachment & Plausibility": [
        "Spatial Attachment, Scale & Physical Plausibility",
        "Physical Attachment & Spatial Plausibility",
    ],
    "3D Scene Integration & Photorealism": [
        "Scene Integration & 3D Depth Coherence",
        "3D Scene Integration & Depth Coherence",
    ],
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def atomic_write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def parse_csv_set(value: str | None) -> set[str] | None:
    if not value or value.lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def filter_tasks(
    rows: list[dict[str, Any]],
    models: set[str] | None,
    samples: set[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    tasks = []
    for row in rows:
        if models and row.get("model_name") not in models:
            continue
        if samples and row.get("sample_id") not in samples:
            continue
        video_path = row.get("video_path")
        tasks.append(
            {
                "task_id": row["task_id"],
                "sample_id": row["sample_id"],
                "model_name": row["model_name"],
                "video_path": video_path,
                "source_json": row.get("source_json"),
            }
        )
        if limit and len(tasks) >= limit:
            break
    return tasks


def load_project_id(credentials_path: Path) -> str:
    data = json.loads(credentials_path.read_text(encoding="utf-8"))
    project_id = data.get("project_id")
    if not project_id:
        raise RuntimeError(f"Missing project_id in credentials file: {credentials_path}")
    return str(project_id)


def build_client(credentials_path: Path, location: str):
    from google import genai
    from google.oauth2 import service_account

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


def video_part(path: str):
    from google.genai import types

    video_path = Path(path)
    mime_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
    return types.Part(
        inline_data=types.Blob(data=video_path.read_bytes(), mime_type=mime_type),
        video_metadata=types.VideoMetadata(fps=DEFAULT_VIDEO_FPS),
    )


def choose_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        if isinstance(value.get("scores"), dict):
            return choose_payload(value["scores"])
        for key in ["result", "results", "response", "evaluation", "scores_by_dimension"]:
            nested = value.get(key)
            if isinstance(nested, (dict, list)):
                candidate = choose_payload(nested)
                if candidate:
                    return candidate
        return value
    if isinstance(value, list):
        merged: dict[str, Any] = {}
        for item in value:
            if isinstance(item, dict):
                candidate = choose_payload(item)
                matched = False
                for dimension in DIMENSIONS:
                    if dimension in candidate:
                        merged[dimension] = candidate[dimension]
                        matched = True
                name = candidate.get("dimension") or candidate.get("name") or candidate.get("metric")
                if isinstance(name, str):
                    for dimension in DIMENSIONS:
                        if name.strip().lower() == dimension.lower():
                            merged[dimension] = {
                                "score": candidate.get("score"),
                                "reason": candidate.get("reason", ""),
                            }
                            matched = True
                            break
                if matched and all(dim in merged for dim in DIMENSIONS):
                    return merged
                if any(dim in candidate for dim in DIMENSIONS):
                    return candidate
        if merged:
            return merged
        for item in value:
            if isinstance(item, dict):
                return choose_payload(item)
    return {}


def parse_json_response(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    try:
        return choose_payload(json.loads(text))
    except json.JSONDecodeError:
        stripped = text.strip()
        for start, char in enumerate(stripped):
            if char not in "[{":
                continue
            try:
                payload, _ = decoder.raw_decode(stripped[start:])
                chosen = choose_payload(payload)
                if chosen:
                    return chosen
            except json.JSONDecodeError:
                continue
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            return choose_payload(json.loads(match.group()))
        raise


def normalize_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 5 else None


def normalize_result(payload: dict[str, Any]) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    values = []
    for dimension in DIMENSIONS:
        raw = payload.get(dimension, {})
        if not isinstance(raw, dict):
            raw = {}
        if not raw:
            for alias in DIMENSION_ALIASES.get(dimension, []):
                candidate = payload.get(alias, {})
                if isinstance(candidate, dict) and candidate:
                    raw = candidate
                    break
        if not isinstance(raw, dict):
            raw = {}
        score = normalize_score(raw.get("score"))
        if score is not None:
            values.append(score)
        scores[dimension] = {
            "applicable": True,
            "score": score,
            "reason": str(raw.get("reason", "")),
        }

    overall_raw = payload.get(OVERALL_DIMENSION, {})
    if not isinstance(overall_raw, dict):
        overall_raw = {}
    overall_score = sum(values) / len(DIMENSIONS) if len(values) == len(DIMENSIONS) else None
    scores[OVERALL_DIMENSION] = {
        "applicable": overall_score is not None,
        "score": overall_score,
        "reason": str(
            overall_raw.get(
                "reason",
                "Arithmetic mean of the four object integrity dimensions.",
            )
        ),
    }
    return scores


def result_row(
    task: dict[str, Any],
    scores: dict[str, Any],
    raw_response: Any,
    error: str | None,
) -> dict[str, Any]:
    return {
        **task,
        "scores": scores,
        "raw_response": raw_response,
        "error": error,
    }


def dry_run_result(task: dict[str, Any]) -> dict[str, Any]:
    payload = {
        dimension: {
            "score": None,
            "reason": "Dry run: Gemini was not called.",
        }
        for dimension in DIMENSIONS
    }
    return result_row(task, normalize_result(payload), payload, None)


def evaluate_task(
    client,
    task: dict[str, Any],
    prompt: str,
    model: str,
    timeout: int,
    temperature: float,
) -> dict[str, Any]:
    from google.genai import types

    video_path = Path(task["video_path"])
    if not video_path.exists():
        return result_row(task, {}, None, f"Video not found: {video_path}")

    contents: list[Any] = [
        prompt,
        (
            f"Evaluate the complete generated video. Sample: {task['sample_id']}; "
            f"generated-video model: {task['model_name']}."
        ),
        video_part(str(video_path)),
    ]
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
            scores = normalize_result(payload)
            missing = [
                dimension
                for dimension in DIMENSIONS
                if scores[dimension]["score"] is None
            ]
            if missing:
                raise ValueError(f"Missing or invalid scores: {', '.join(missing)}")
            return result_row(task, scores, payload, None)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(2 * (attempt + 1))
    return result_row(task, {}, None, last_error or "Unknown Gemini error")


def build_output_path(output: str | None, output_dir: str, run_name: str | None) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"{run_name}_{timestamp}" if run_name else f"ois_8models_30samples_{timestamp}"
    return Path(output_dir).expanduser().resolve() / folder / "anatomical_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate four-dimensional object integrity with Gemini."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUTS_DIR))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--location", default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--models", default="all")
    parser.add_argument("--samples", default="all")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = filter_tasks(
        read_rows(Path(args.manifest).expanduser().resolve()),
        parse_csv_set(args.models),
        parse_csv_set(args.samples),
        args.limit,
    )
    if not tasks:
        raise ValueError("No tasks to evaluate.")

    output = build_output_path(args.output, args.output_dir, args.run_name)
    existing = [] if args.overwrite else (read_rows(output) if output.exists() else [])
    done = {row.get("task_id") for row in existing if not row.get("error")}
    pending = [task for task in tasks if task["task_id"] not in done]
    if args.overwrite:
        existing = []
        atomic_write_rows(output, existing)

    output.parent.mkdir(parents=True, exist_ok=True)
    latest = Path(args.output_dir).expanduser().resolve() / "latest_run.txt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(output.parent) + "\n", encoding="utf-8")
    print(f"Tasks total: {len(tasks)}; already done: {len(done)}; pending: {len(pending)}")
    print(f"Output: {output}")
    print(f"Dry run: {args.dry_run}")
    if not pending:
        return

    if args.dry_run:
        for task in pending:
            existing.append(dry_run_result(task))
        atomic_write_rows(output, existing)
        return

    credentials_path = args.credentials.expanduser().resolve()
    if not credentials_path.exists():
        raise FileNotFoundError(f"Gemini credentials file not found: {credentials_path}")
    prompt = Path(args.prompt).expanduser().resolve().read_text(encoding="utf-8")
    client = build_client(credentials_path, args.location)
    print(f"Judge model: {args.model}; location: {args.location}")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                evaluate_task,
                client,
                task,
                prompt,
                args.model,
                args.timeout,
                args.temperature,
            )
            for task in pending
        ]
        completed = as_completed(futures)
        iterator = tqdm(completed, total=len(futures), desc="Anatomical consistency") if tqdm else completed
        for future in iterator:
            existing.append(future.result())
            atomic_write_rows(output, existing)


if __name__ == "__main__":
    main()
