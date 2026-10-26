#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
import os
import re
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TASKS = SCRIPT_DIR / "inputs/speech_eval_tasks_30samples_6models.json"
DEFAULT_PROMPT = SCRIPT_DIR / "if_judge_prompts_v3/speech_accuracy.md"
DEFAULT_OUTPUTS_DIR = SCRIPT_DIR / "outputs"
DEFAULT_CREDENTIALS = Path(os.getenv("FINALEVAL_GEMINI_CREDENTIALS", ""))
DEFAULT_MODEL = os.getenv("SPEECH_EVAL_MODEL") or os.getenv("IF_EVAL_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-3.1-pro-preview"
DEFAULT_VIDEO_FPS = float(os.getenv("IF_GEMINI_VIDEO_FPS", "4"))
DEFAULT_RUN_NAME = "speech_content_30samples_6models"
LATEST_RUN_FILE = DEFAULT_OUTPUTS_DIR / "latest_speech_run.txt"
METRIC_NAME = "Speech Content Accuracy"
MISSING_VIDEO_ERROR_PREFIX = "Generated video file is missing:"
DEFAULT_SAMPLES = (
    "board1_001,board1_005,board1_007,board1_015,board1_027,board1_034,board1_045,board1_065,board1_095,board1_104,"
    "board2_001,board2_003,board2_004,board2_018,board2_024,board2_034,board2_038,board2_052,board2_063,board2_100,"
    "board3_001,board3_002,board3_005,board3_008,board3_009,board3_011,board3_013,board3_027,board3_045,board3_049"
)
DEFAULT_MODELS = "happyhourse,keling,seedance,veo,vidu,wan"

JSON_OUTPUT_INSTRUCTIONS = """
Return only valid JSON with this schema:
{
  "applicable": true,
  "score": 1-5 number or null,
  "transcript_or_observed_speech": "brief transcript or observed spoken content",
  "item_results": [
    {
      "speaker": "expected speaker if specified",
      "expected_text": "required speech text",
      "observed_speech": "matching observed speech or empty string",
      "score": 1-5 number or null,
      "reason": "short reason"
    }
  ],
  "major_errors": ["short error descriptions"],
  "reason": "brief overall explanation"
}

Only return {"applicable": false, "score": null, ...} when the prompt has no required speech.
If expected speech is specified but the video has no speech or does not contain the required speech, return applicable=true and score=1.
Ignore speaker assignment. Do not reduce the score if the required speech is spoken by the wrong character, narrator, off-screen voice, or background voice.
"""


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
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def atomic_write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def parse_csv_set(value: str | None) -> set[str] | None:
    if not value or value.lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


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


def file_part(path: str):
    from google.genai import types

    file_path = Path(path)
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    data = file_path.read_bytes()
    if mime_type.startswith("video/"):
        return types.Part(
            inline_data=types.Blob(data=data, mime_type=mime_type),
            video_metadata=types.VideoMetadata(fps=DEFAULT_VIDEO_FPS),
        )
    return types.Part.from_bytes(data=data, mime_type=mime_type)


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group())


def normalize_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 5 else None


def board_from_sample(sample_id: str | None) -> str:
    return str(sample_id or "unknown").split("_", 1)[0]


def speech_items(task: dict[str, Any]) -> list[dict[str, str]]:
    items = task.get("if_eval_spec", {}).get("checklist", {}).get("speech_prompt") or []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        out.append(
            {
                "speaker": str(item.get("speaker") or "").strip(),
                "text": text,
                "language": str(item.get("language") or "").strip(),
            }
        )
    return out


def build_prompt(prompt_template: str, task: dict[str, Any]) -> str:
    spec = task.get("if_eval_spec", {})
    payload = {
        "sample_id": task.get("sample_id"),
        "model_name": task.get("model_name"),
        "video_generation_prompt": spec.get("prompt", ""),
        "chinese_prompt": spec.get("chinese_prompt", ""),
        "expected_speech_items": speech_items(task),
        "applicable_dimensions": spec.get("applicable_dimensions", {}),
    }
    return "\n\n".join(
        [
            prompt_template.strip(),
            "## Evaluation Input",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "## Required Output",
            JSON_OUTPUT_INSTRUCTIONS.strip(),
        ]
    )


def build_contents(task: dict[str, Any], prompt_text: str) -> list[Any]:
    contents: list[Any] = [prompt_text]
    spec = task.get("if_eval_spec", {})
    for ref in spec.get("references", []):
        ref_path = ref.get("path")
        if not ref_path or not Path(ref_path).exists():
            continue
        contents.append(f"Reference image {ref.get('placeholder')} ({ref.get('type')}): {ref_path}")
        contents.append(file_part(ref_path))

    video_path = task.get("video_path")
    contents.append(f"Generated video: {video_path}")
    if video_path and Path(video_path).exists():
        contents.append(file_part(video_path))
    else:
        contents.append("WARNING: Generated video file is missing or unreadable.")
    return contents


def normalize_item_results(value: Any, expected_items: list[dict[str, str]]) -> list[dict[str, Any]]:
    raw_items = value if isinstance(value, list) else []
    out = []
    for index, expected in enumerate(expected_items):
        raw = raw_items[index] if index < len(raw_items) and isinstance(raw_items[index], dict) else {}
        out.append(
            {
                "speaker": str(raw.get("speaker") or expected.get("speaker") or ""),
                "language": str(raw.get("language") or expected.get("language") or ""),
                "expected_text": str(raw.get("expected_text") or raw.get("question") or expected.get("text") or ""),
                "observed_speech": str(raw.get("observed_speech") or raw.get("matched_text") or raw.get("transcript") or ""),
                "score": normalize_score(raw.get("score")),
                "reason": str(raw.get("reason") or ""),
            }
        )
    for raw in raw_items[len(expected_items) :]:
        if not isinstance(raw, dict):
            continue
        out.append(
            {
                "speaker": str(raw.get("speaker") or ""),
                "language": str(raw.get("language") or ""),
                "expected_text": str(raw.get("expected_text") or raw.get("question") or ""),
                "observed_speech": str(raw.get("observed_speech") or raw.get("matched_text") or raw.get("transcript") or ""),
                "score": normalize_score(raw.get("score")),
                "reason": str(raw.get("reason") or ""),
            }
        )
    return out


def normalize_payload(payload: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    expected_items = speech_items(task)
    if not expected_items:
        return {
            "applicable": False,
            "score": None,
            "transcript_or_observed_speech": "",
            "item_results": [],
            "major_errors": [],
            "reason": "No speech prompt is specified for this sample.",
        }

    payload_applicable = bool(payload.get("applicable", True))
    item_results = normalize_item_results(payload.get("item_results", []), expected_items)
    item_scores = [item["score"] for item in item_results if item.get("score") is not None]
    if len(item_scores) > 1:
        score = float(math.floor(mean(item_scores)))
    elif item_scores:
        score = item_scores[0]
    else:
        score = normalize_score(payload.get("score"))
    major_errors = payload.get("major_errors") or payload.get("major_failures") or []
    if not isinstance(major_errors, list):
        major_errors = [str(major_errors)]

    absence_text = " ".join(
        [
            str(payload.get("transcript_or_observed_speech") or payload.get("transcript") or ""),
            str(payload.get("reason") or payload.get("comments") or ""),
            " ".join(str(item) for item in major_errors),
        ]
    ).lower()
    absence_markers = [
        "no speech",
        "no audio",
        "absent",
        "missing",
        "not spoken",
        "not present",
        "silent",
        "没有语音",
        "没有说",
        "未生成",
        "无语音",
    ]
    if score is None and (not payload_applicable or any(marker in absence_text for marker in absence_markers)):
        score = 1.0
        for item in item_results:
            if item.get("score") is None:
                item["score"] = 1.0
                if not item.get("reason"):
                    item["reason"] = "Required speech is absent from the video."

    if score is None:
        raise ValueError("Missing or invalid speech score")
    return {
        "applicable": True,
        "score": score,
        "transcript_or_observed_speech": str(payload.get("transcript_or_observed_speech") or payload.get("transcript") or ""),
        "item_results": item_results,
        "major_errors": [str(item) for item in major_errors],
        "reason": str(payload.get("reason") or payload.get("comments") or ""),
    }


def result_row(task: dict[str, Any], payload: dict[str, Any] | None, raw_response: Any, error: str | None) -> dict[str, Any]:
    expected_items = speech_items(task)
    payload = payload or {}
    return {
        "task_id": task.get("task_id"),
        "sample_id": task.get("sample_id"),
        "model_name": task.get("model_name"),
        "video_path": task.get("video_path"),
        "source_json": task.get("if_eval_spec", {}).get("source_json"),
        "if_eval_spec": task.get("if_eval_spec"),
        "expected_speech_items": expected_items,
        "applicable": payload.get("applicable", bool(expected_items)),
        "score": payload.get("score"),
        "transcript_or_observed_speech": payload.get("transcript_or_observed_speech", ""),
        "item_results": payload.get("item_results", []),
        "major_errors": payload.get("major_errors", []),
        "reason": payload.get("reason", ""),
        "raw_response": raw_response,
        "error": error,
    }


def is_missing_video_error(error: Any) -> bool:
    return str(error or "").startswith(MISSING_VIDEO_ERROR_PREFIX)


def missing_video_payload() -> dict[str, Any]:
    return {
        "applicable": False,
        "score": None,
        "transcript_or_observed_speech": "",
        "item_results": [],
        "major_errors": [],
        "reason": "Generated video file is missing; excluded from speech scoring.",
    }


def normalize_cached_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if out.get("error"):
        if is_missing_video_error(out.get("error")):
            out.update(missing_video_payload())
        return out
    expected_items = out.get("expected_speech_items")
    if expected_items is None:
        expected_items = speech_items(out)
        out["expected_speech_items"] = expected_items
    fake_task = {"if_eval_spec": {"checklist": {"speech_prompt": expected_items}}}
    payload = {
        "applicable": out.get("applicable", bool(expected_items)),
        "score": out.get("score"),
        "transcript_or_observed_speech": out.get("transcript_or_observed_speech", ""),
        "item_results": out.get("item_results", []),
        "major_errors": out.get("major_errors", []),
        "reason": out.get("reason", ""),
    }
    try:
        normalized = normalize_payload(payload, fake_task)
    except Exception:  # noqa: BLE001
        return out
    out.update(normalized)
    return out


def dry_run_result(task: dict[str, Any]) -> dict[str, Any]:
    expected_items = speech_items(task)
    payload = {
        "applicable": bool(expected_items),
        "score": None,
        "transcript_or_observed_speech": "",
        "item_results": [
            {
                "speaker": item.get("speaker", ""),
                "language": item.get("language", ""),
                "expected_text": item.get("text", ""),
                "observed_speech": "",
                "score": None,
                "reason": "Dry run: Gemini was not called.",
            }
            for item in expected_items
        ],
        "major_errors": [],
        "reason": "Dry run: Gemini was not called.",
    }
    return result_row(task, payload, None, None)


def evaluate_task(
    client,
    task: dict[str, Any],
    prompt_template: str,
    model: str,
    timeout: int,
    temperature: float,
) -> dict[str, Any]:
    from google.genai import types

    if not speech_items(task):
        return result_row(task, normalize_payload({}, task), None, None)
    if not task.get("video_path") or not Path(str(task["video_path"])).exists():
        return result_row(task, missing_video_payload(), None, f"{MISSING_VIDEO_ERROR_PREFIX} {task.get('video_path')}")

    prompt_text = build_prompt(prompt_template, task)
    contents = build_contents(task, prompt_text)
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
            raw = parse_json_response(response.text or "{}")
            payload = normalize_payload(raw, task)
            return result_row(task, payload, raw, None)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(2 * (attempt + 1))
    return result_row(task, None, None, last_error or "unknown error")


def filter_tasks(tasks: list[dict[str, Any]], models: set[str] | None, samples: set[str] | None, limit: int | None) -> list[dict[str, Any]]:
    out = []
    for task in tasks:
        if models and task.get("model_name") not in models:
            continue
        if samples and task.get("sample_id") not in samples:
            continue
        out.append(task)
        if limit and len(out) >= limit:
            break
    return out


def build_output_path(output: str | None, output_dir: Path, run_name: str | None) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"{run_name or DEFAULT_RUN_NAME}_{timestamp}"
    return output_dir.expanduser().resolve() / folder / "speech_results.json"


def score_from(row: dict[str, Any]) -> float | None:
    if row.get("error") or not row.get("applicable"):
        return None
    return normalize_score(row.get("score"))


def summary_score_from(row: dict[str, Any]) -> float | None:
    score = score_from(row)
    if score is not None:
        return score
    if row.get("applicable"):
        return 1.0
    return None


def collapse(values: list[float]) -> dict[str, Any]:
    return {"mean_1_5": mean(values) if values else None, "count": len(values)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall: list[float] = []
    by_model: dict[str, list[float]] = defaultdict(list)
    by_board: dict[str, list[float]] = defaultdict(list)
    by_model_and_board: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    model_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    board_counts: dict[str, int] = defaultdict(int)
    failures = []
    models = sorted({str(row.get("model_name", "unknown")) for row in rows})

    for row in rows:
        model = str(row.get("model_name", "unknown"))
        board = board_from_sample(row.get("sample_id"))
        model_counts[model]["total"] += 1
        board_counts[board] += 1
        if row.get("applicable"):
            model_counts[model]["applicable"] += 1
        if row.get("error"):
            model_counts[model]["failed"] += 1
            failures.append({"task_id": row.get("task_id"), "model_name": model, "error": row.get("error")})

        score = score_from(row)
        summary_score = summary_score_from(row)
        if not row.get("applicable"):
            model_counts[model]["skipped"] += 1
            continue
        if score is not None:
            model_counts[model]["raw_scored"] += 1
        else:
            model_counts[model]["missing_score_as_one"] += 1
        if summary_score is None:
            continue
        model_counts[model]["scored"] += 1
        overall.append(summary_score)
        by_model[model].append(summary_score)
        by_board[board].append(summary_score)
        by_model_and_board[board][model].append(summary_score)

    return {
        "overall": {METRIC_NAME: collapse(overall)},
        "by_model": {
            model: {METRIC_NAME: collapse(by_model.get(model, [])), "counts": dict(model_counts[model])}
            for model in models
        },
        "by_board": {
            board: {METRIC_NAME: collapse(by_board.get(board, [])), "total_rows": board_counts[board]}
            for board in sorted(board_counts)
        },
        "by_model_and_board": {
            board: {model: {METRIC_NAME: collapse(values)} for model, values in sorted(models.items())}
            for board, models in sorted(by_model_and_board.items())
        },
        "total_rows": len(rows),
        "successful_rows": len(rows) - len(failures),
        "failed_rows": len(failures),
        "aggregation": "applicable_rows_missing_scores_as_one",
        "missing_score_as_one_rows": sum(counts.get("missing_score_as_one", 0) for counts in model_counts.values()),
        "failures": failures,
    }


def fmt_score(value: Any) -> str:
    return "" if value is None else f"{float(value):.3f}"


def ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (-float(row["score"]), row["model_name"]))
    previous = None
    rank = 0
    out = []
    for index, row in enumerate(ordered, start=1):
        if previous is None or row["score"] != previous:
            rank = index
            previous = row["score"]
        item = dict(row)
        item["rank"] = rank
        out.append(item)
    return out


def ranking_data(summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overall_rows = []
    for model, metrics in summary["by_model"].items():
        stats = metrics[METRIC_NAME]
        if stats["mean_1_5"] is not None:
            overall_rows.append({"model_name": model, "score": stats["mean_1_5"], "count": stats["count"]})

    board_rows = []
    for board, models in summary["by_model_and_board"].items():
        rows = []
        for model, metrics in models.items():
            stats = metrics[METRIC_NAME]
            if stats["mean_1_5"] is not None:
                rows.append({"board": board, "model_name": model, "score": stats["mean_1_5"], "count": stats["count"]})
        board_rows.extend(ranked(rows))
    return ranked(overall_rows), board_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field) for field in fields}
            for field, value in list(out.items()):
                if isinstance(value, float):
                    out[field] = fmt_score(value)
            if "score" in out:
                out["score"] = fmt_score(out["score"])
            writer.writerow(out)


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return out


def write_ranking_md(
    path: Path,
    run_dir: Path,
    summary: dict[str, Any],
    overall_rank: list[dict[str, Any]],
    board_rank: list[dict[str, Any]],
    results_path: Path,
    prompt_path: Path,
) -> None:
    models = sorted(summary["by_model"])
    boards = sorted(summary["by_board"])
    samples_per_model = 0
    if models:
        first_model = summary["by_model"][models[0]]
        samples_per_model = first_model.get("counts", {}).get("total", first_model[METRIC_NAME]["count"])
    lines = [
        "# Speech Content Accuracy Ranking",
        "",
        f"- Source: `{run_dir}`",
        f"- Results: `{results_path}`",
        f"- Prompt: `{prompt_path}`",
        f"- Models: {len(models)}",
        f"- Samples per model: {samples_per_model}",
        f"- Boards: {', '.join(boards)}",
        f"- Total metric: `{METRIC_NAME}`",
        "- Rank order: higher score is better; equal scores share the same rank.",
        "- Aggregation: applicable rows only. Missing generated videos are excluded from scoring.",
        "- If a speech-required row has no valid score for reasons other than a missing generated video, it is counted as 1.",
        f"- Non-video-missing rows counted as 1: {summary.get('missing_score_as_one_rows', 0)}",
        "",
        "## Overall Ranking",
        "",
    ]
    lines.extend(
        md_table(
            ["Rank", "Model", METRIC_NAME, "Count"],
            [[row["rank"], row["model_name"], fmt_score(row["score"]), row["count"]] for row in overall_rank],
        )
    )
    lines.extend(["", "## Board Rankings"])
    for board in boards:
        selected = [row for row in board_rank if row["board"] == board]
        lines.extend(["", f"### {board}", ""])
        if selected:
            lines.extend(
                md_table(
                    ["Rank", "Model", METRIC_NAME, "Count"],
                    [[row["rank"], row["model_name"], fmt_score(row["score"]), row["count"]] for row in selected],
                )
            )
        else:
            lines.append("No valid scores.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_outputs(run_dir: Path, rows: list[dict[str, Any]], results_path: Path, prompt_path: Path) -> None:
    rows = sorted(rows, key=lambda row: (str(row.get("sample_id")), str(row.get("model_name"))))
    summary = summarize(rows)
    (run_dir / "speech_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    model_rows = []
    for model, metrics in summary["by_model"].items():
        stats = metrics[METRIC_NAME]
        counts = metrics.get("counts", {})
        model_rows.append(
            {
                "model_name": model,
                f"{METRIC_NAME}_mean_1_5": stats["mean_1_5"],
                f"{METRIC_NAME}_count": stats["count"],
                "total_rows": counts.get("total", 0),
                "applicable_rows": counts.get("applicable", 0),
                "scored_rows": counts.get("scored", 0),
                "raw_scored_rows": counts.get("raw_scored", 0),
                "missing_score_as_one_rows": counts.get("missing_score_as_one", 0),
                "failed_rows": counts.get("failed", 0),
                "skipped_rows": counts.get("skipped", 0),
            }
        )
    write_csv(
        run_dir / "speech_model_summary.csv",
        model_rows,
        [
            "model_name",
            f"{METRIC_NAME}_mean_1_5",
            f"{METRIC_NAME}_count",
            "total_rows",
            "applicable_rows",
            "scored_rows",
            "raw_scored_rows",
            "missing_score_as_one_rows",
            "failed_rows",
            "skipped_rows",
        ],
    )

    summary_rows = [{"group_type": "overall", "group": "overall", "mean_1_5": summary["overall"][METRIC_NAME]["mean_1_5"], "count": summary["overall"][METRIC_NAME]["count"]}]
    for model, metrics in summary["by_model"].items():
        stats = metrics[METRIC_NAME]
        summary_rows.append({"group_type": "model", "group": model, "mean_1_5": stats["mean_1_5"], "count": stats["count"]})
    for board, metrics in summary["by_board"].items():
        stats = metrics[METRIC_NAME]
        summary_rows.append({"group_type": "board", "group": board, "mean_1_5": stats["mean_1_5"], "count": stats["count"]})
    write_csv(run_dir / "speech_summary.csv", summary_rows, ["group_type", "group", "mean_1_5", "count"])

    per_sample_rows = []
    for row in rows:
        per_sample_rows.append(
            {
                "task_id": row.get("task_id"),
                "sample_id": row.get("sample_id"),
                "model_name": row.get("model_name"),
                "video_path": row.get("video_path"),
                "error": row.get("error"),
                "applicable": row.get("applicable"),
                METRIC_NAME: summary_score_from(row),
                "major_errors": "; ".join(row.get("major_errors") or []),
                "reason": row.get("reason", ""),
                "transcript_or_observed_speech": row.get("transcript_or_observed_speech", ""),
            }
        )
    write_csv(
        run_dir / "speech_per_sample.csv",
        per_sample_rows,
        [
            "task_id",
            "sample_id",
            "model_name",
            "video_path",
            "error",
            "applicable",
            METRIC_NAME,
            "major_errors",
            "reason",
            "transcript_or_observed_speech",
        ],
    )

    item_rows = []
    for row in rows:
        for index, item in enumerate(row.get("item_results") or [], start=1):
            item_rows.append(
                {
                    "task_id": row.get("task_id"),
                    "sample_id": row.get("sample_id"),
                    "model_name": row.get("model_name"),
                    "item_index": index,
                    "speaker": item.get("speaker"),
                    "language": item.get("language"),
                    "expected_text": item.get("expected_text"),
                    "observed_speech": item.get("observed_speech"),
                    "item_score": item.get("score"),
                    "item_reason": item.get("reason"),
                }
            )
    write_csv(
        run_dir / "speech_item_detail.csv",
        item_rows,
        ["task_id", "sample_id", "model_name", "item_index", "speaker", "language", "expected_text", "observed_speech", "item_score", "item_reason"],
    )

    overall_rank, board_rank = ranking_data(summary)
    write_csv(run_dir / "speech_ranking_overall.csv", overall_rank, ["rank", "model_name", "score", "count"])
    write_csv(run_dir / "speech_ranking_by_board.csv", board_rank, ["board", "rank", "model_name", "score", "count"])
    write_ranking_md(run_dir / "speech_ranking.md", run_dir, summary, overall_rank, board_rank, results_path, prompt_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IF Speech Content Accuracy with a Gemini MLLM judge.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--location", default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--samples", default=DEFAULT_SAMPLES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write placeholder rows without calling Gemini.")
    parser.add_argument("--rewrite-only", action="store_true", help="Rewrite cached normalized results and summaries without new Gemini calls.")
    args = parser.parse_args()
    if args.rewrite_only and args.overwrite:
        parser.error("--rewrite-only cannot be used with --overwrite")

    output = build_output_path(args.output, args.output_dir, args.run_name)
    existing = [] if args.overwrite else [normalize_cached_row(row) for row in read_rows(output)]
    if args.rewrite_only:
        if not output.exists():
            raise FileNotFoundError(f"Cannot rewrite missing output: {output}")
        atomic_write_rows(output, existing)
        write_summary_outputs(output.parent, existing, output, args.prompt.expanduser().resolve())
        print(f"Rewrote cached results and summary outputs to {output.parent}", flush=True)
        return

    tasks = filter_tasks(read_rows(args.tasks.expanduser().resolve()), parse_csv_set(args.models), parse_csv_set(args.samples), args.limit)
    if not tasks:
        raise ValueError("No tasks match the requested --models/--samples filters.")

    output.parent.mkdir(parents=True, exist_ok=True)
    LATEST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    LATEST_RUN_FILE.write_text(str(output.parent) + "\n", encoding="utf-8")
    done = {row.get("task_id") for row in existing if not row.get("error")}
    pending = [task for task in tasks if task["task_id"] not in done]
    if args.overwrite:
        existing = []
        atomic_write_rows(output, existing)

    print(f"Tasks total: {len(tasks)}; already done: {len(done)}; pending: {len(pending)}", flush=True)
    print(f"Output: {output}", flush=True)
    print(f"Dry run: {args.dry_run}", flush=True)

    if args.dry_run:
        for task in pending:
            existing.append(dry_run_result(task))
            atomic_write_rows(output, existing)
        write_summary_outputs(output.parent, existing, output, args.prompt.expanduser().resolve())
        print(f"Saved dry-run summary outputs to {output.parent}", flush=True)
        return

    if pending:
        prompt_template = args.prompt.expanduser().resolve().read_text(encoding="utf-8")
        credentials_path = args.credentials.expanduser().resolve()
        if not credentials_path.exists():
            raise FileNotFoundError(f"Gemini credentials file not found: {credentials_path}")
        client = build_client(credentials_path, args.location)
        print(f"Judge model: {args.model}; location: {args.location}", flush=True)
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(evaluate_task, client, task, prompt_template, args.model, args.timeout, args.temperature)
                for task in pending
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                existing.append(result)
                atomic_write_rows(output, existing)
                status = "ok" if not result.get("error") else "error"
                print(f"[{index}/{len(pending)}] {status} {result.get('task_id')}", flush=True)

    write_summary_outputs(output.parent, existing, output, args.prompt.expanduser().resolve())
    print(f"Saved summary outputs to {output.parent}", flush=True)


if __name__ == "__main__":
    main()
