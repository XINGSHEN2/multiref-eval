#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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

from google import genai
from google.genai import types
from google.oauth2 import service_account


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "inputs" / "manifest.json"
DEFAULT_PROMPT = SCRIPT_DIR / "MLLM_Prompt" / "binding.md"
DEFAULT_OUTPUTS_DIR = SCRIPT_DIR / "outputs" / "binding"
DEFAULT_CREDENTIALS = Path(os.getenv("FINALEVAL_GEMINI_CREDENTIALS", ""))
DEFAULT_MODEL = os.getenv("BINDING_EVAL_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-3.5-flash"
DEFAULT_VIDEO_FPS = float(os.getenv("MLLM_GEMINI_VIDEO_FPS", "4"))
TOTAL_METRIC = "Binding Score"
SUB_DIMENSIONS = ["Identity Binding", "Attribute Binding", "Action Binding"]
ALL_METRICS = [*SUB_DIMENSIONS, TOTAL_METRIC]
DIMENSION_ITEMS = {
    "Identity Binding": ["I2", "I3"],
    "Attribute Binding": ["B1", "B3"],
    "Action Binding": ["AC1", "AC2"],
}
CHECKLIST_ITEMS = ["I2", "I3", "B1", "B3", "AC1", "AC2"]
ITEM_VALUE_SCORES = {"YES": 5.0, "PARTIAL": 3.0, "NO": 1.0}


def item_dimension(item: str) -> str | None:
    for dimension, items in DIMENSION_ITEMS.items():
        if item in items:
            return dimension
    return None


def configure_proxy() -> None:
    if os.getenv("BINDING_NO_PROXY") == "1":
        for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
            os.environ.pop(key, None)
        return
    proxy = os.getenv("BINDING_PROXY_URL", "http://oversea-squid2.ko.txyun:11080")
    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        os.environ[key] = proxy


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
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
    if mime_type.startswith("video/"):
        return types.Part(
            inline_data=types.Blob(data=file_path.read_bytes(), mime_type=mime_type),
            video_metadata=types.VideoMetadata(fps=DEFAULT_VIDEO_FPS),
        )
    return types.Part.from_bytes(data=file_path.read_bytes(), mime_type=mime_type)


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


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    checklist = payload.get("checklist", {})
    if not isinstance(checklist, dict):
        checklist = {}
    checklist = {item: str(checklist.get(item, "")).strip() for item in CHECKLIST_ITEMS}
    subscores = subdimension_scores(checklist)
    score = total_score_from_subscores(subscores)
    if score is None:
        score = normalize_score(payload.get("score"))
    if score is None:
        raise ValueError("Missing or invalid score")

    failures_critical = payload.get("critical_failures", [])
    failures_minor = payload.get("minor_failures", [])
    if not isinstance(failures_critical, list):
        failures_critical = [str(failures_critical)]
    if not isinstance(failures_minor, list):
        failures_minor = [str(failures_minor)]

    failure_reasons = payload.get("failure_reasons", {})
    if not isinstance(failure_reasons, dict):
        failure_reasons = {}
    failure_reasons = {item: str(failure_reasons.get(item, "")).strip() for item in CHECKLIST_ITEMS}

    score_reasons = payload.get("score_reasons", {})
    if not isinstance(score_reasons, dict):
        score_reasons = {}
    score_reasons = {metric: str(score_reasons.get(metric, "")).strip() for metric in ALL_METRICS}

    for dimension, items in DIMENSION_ITEMS.items():
        if score_reasons.get(dimension):
            continue
        reasons = [f"{item}: {failure_reasons[item]}" for item in items if failure_reasons.get(item)]
        score_reasons[dimension] = "; ".join(reasons)
    brief_reason = str(payload.get("brief_reason") or payload.get("comments") or "").strip()
    if not score_reasons.get(TOTAL_METRIC):
        score_reasons[TOTAL_METRIC] = brief_reason

    return {
        "num_subjects": payload.get("num_subjects"),
        "has_audio": bool(payload.get("has_audio", False)),
        "checklist": checklist,
        "score": score,
        "subscores": subscores,
        "score_reasons": score_reasons,
        "failure_reasons": failure_reasons,
        "critical_failures": [str(item) for item in failures_critical],
        "minor_failures": [str(item) for item in failures_minor],
        "comments": brief_reason,
    }


def checklist_value_score(value: Any) -> float | None:
    text = str(value or "").strip().upper()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    return ITEM_VALUE_SCORES.get(text)


def subdimension_scores(checklist: dict[str, Any]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for dimension, items in DIMENSION_ITEMS.items():
        values = [checklist_value_score(checklist.get(item)) for item in items]
        values = [value for value in values if value is not None]
        scores[dimension] = mean(values) if values else None
    return scores


def total_score_from_subscores(subscores: dict[str, Any]) -> float | None:
    values = []
    for dimension in SUB_DIMENSIONS:
        try:
            score = float(subscores.get(dimension))
        except (TypeError, ValueError):
            continue
        if 1 <= score <= 5:
            values.append(score)
    return mean(values) if values else None


def build_task(row: dict[str, Any]) -> dict[str, Any]:
    spec = row.get("if_eval_spec") or {}
    return {
        "task_id": row["task_id"],
        "sample_id": row["sample_id"],
        "model_name": row["model_name"],
        "video_path": row.get("video_path"),
        "source_json": row.get("source_json") or spec.get("source_json"),
        "prompt": spec.get("prompt", ""),
        "chinese_prompt": spec.get("chinese_prompt", ""),
        "references": spec.get("references") or spec.get("image_items") or [],
    }


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
        tasks.append(build_task(row))
        if limit and len(tasks) >= limit:
            break
    if not tasks:
        raise ValueError("No tasks to evaluate")
    print(
        f"Models: {len({task['model_name'] for task in tasks})}; "
        f"samples: {len({task['sample_id'] for task in tasks})}; tasks: {len(tasks)}",
        flush=True,
    )
    return tasks


def build_contents(task: dict[str, Any], prompt_template: str) -> list[Any]:
    contents: list[Any] = [
        prompt_template,
        (
            "Evaluation context:\n"
            f"- Sample ID: {task['sample_id']}\n"
            f"- Generated-video model: {task['model_name']}\n"
            f"- Text prompt, for context only. Do not score prompt following: {task.get('prompt', '')}\n"
            f"- Chinese text prompt, for context only. Do not score prompt following: {task.get('chinese_prompt', '')}\n"
        ),
    ]
    for ref in task.get("references", []):
        ref_path = ref.get("path")
        if not ref_path or not Path(ref_path).exists():
            continue
        contents.append(
            f"Reference image {ref.get('placeholder', '')}: "
            f"type={ref.get('type', '')}; description={ref.get('description', '')}; path={ref_path}"
        )
        contents.append(file_part(ref_path))

    video_path = task.get("video_path")
    contents.append(f"Generated video: {video_path}")
    if not video_path or not Path(video_path).exists():
        contents.append("WARNING: generated video is missing.")
    else:
        contents.append(file_part(video_path))
    return contents


def result_row(
    task: dict[str, Any],
    payload: dict[str, Any] | None,
    raw_response: Any,
    error: str | None,
) -> dict[str, Any]:
    return {
        **task,
        "num_subjects": payload.get("num_subjects") if payload else None,
        "has_audio": payload.get("has_audio") if payload else None,
        "score": payload.get("score") if payload else None,
        "subscores": payload.get("subscores") if payload else {},
        "checklist": payload.get("checklist") if payload else {},
        "score_reasons": payload.get("score_reasons") if payload else {},
        "failure_reasons": payload.get("failure_reasons") if payload else {},
        "critical_failures": payload.get("critical_failures") if payload else [],
        "minor_failures": payload.get("minor_failures") if payload else [],
        "comments": payload.get("comments") if payload else "",
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
) -> dict[str, Any]:
    if not task.get("video_path") or not Path(task["video_path"]).exists():
        return result_row(task, None, None, f"Video not found: {task.get('video_path')}")

    contents = build_contents(task, prompt_template)
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
            payload = normalize_payload(raw)
            return result_row(task, payload, raw, None)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(2 * (attempt + 1))
    return result_row(task, None, None, last_error or "Unknown Gemini error")


def board_from_sample(sample_id: str | None) -> str:
    return str(sample_id or "unknown").split("_", 1)[0]


def collapse(values: list[float]) -> dict[str, Any]:
    return {"mean_1_5": mean(values) if values else None, "count": len(values)}


def score_from(row: dict[str, Any], metric: str) -> float | None:
    if metric == TOTAL_METRIC:
        return total_score_from_subscores(row.get("subscores") or {})
    value = (row.get("subscores") or {}).get(metric)
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 5 else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_board: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_model_and_board: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    overall: dict[str, list[float]] = defaultdict(list)
    failures = []
    for row in rows:
        if row.get("error"):
            failures.append({"task_id": row.get("task_id"), "model_name": row.get("model_name"), "error": row.get("error")})
            continue
        model = str(row.get("model_name", "unknown"))
        board = board_from_sample(row.get("sample_id"))
        for metric in ALL_METRICS:
            score = score_from(row, metric)
            if score is None:
                continue
            overall[metric].append(score)
            by_model[model][metric].append(score)
            by_board[board][metric].append(score)
            by_model_and_board[board][model][metric].append(score)

    def collapse_bucket(bucket: dict[str, list[float]]) -> dict[str, Any]:
        return {metric: collapse(bucket.get(metric, [])) for metric in ALL_METRICS}

    return {
        "overall": collapse_bucket(overall),
        "by_model": {model: collapse_bucket(bucket) for model, bucket in sorted(by_model.items())},
        "by_board": {board: collapse_bucket(bucket) for board, bucket in sorted(by_board.items())},
        "by_model_and_board": {
            board: {model: collapse_bucket(bucket) for model, bucket in sorted(models.items())}
            for board, models in sorted(by_model_and_board.items())
        },
        "total_rows": len(rows),
        "successful_rows": len(rows) - len(failures),
        "failed_rows": len(failures),
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
        score = row["score"]
        if previous is None or score != previous:
            rank = index
            previous = score
        item = dict(row)
        item["rank"] = rank
        out.append(item)
    return out


def ranking_data(summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    overall_rows = []
    for model, metrics in summary["by_model"].items():
        stats = metrics[TOTAL_METRIC]
        if stats["mean_1_5"] is not None:
            overall_rows.append({"model_name": model, "score": stats["mean_1_5"], "count": stats["count"]})

    dimension_rows = []
    for dimension in SUB_DIMENSIONS:
        rows = []
        for model, metrics in summary["by_model"].items():
            stats = metrics[dimension]
            if stats["mean_1_5"] is not None:
                rows.append({"dimension": dimension, "model_name": model, "score": stats["mean_1_5"], "count": stats["count"]})
        dimension_rows.extend(ranked(rows))

    board_rows = []
    board_dimension_rows = []
    for board, models in summary["by_model_and_board"].items():
        board_rank_rows = []
        for model, metrics in models.items():
            stats = metrics[TOTAL_METRIC]
            if stats["mean_1_5"] is not None:
                board_rank_rows.append({"board": board, "model_name": model, "score": stats["mean_1_5"], "count": stats["count"]})
        board_rows.extend(ranked(board_rank_rows))

        for dimension in SUB_DIMENSIONS:
            rows = []
            for model, metrics in models.items():
                stats = metrics[dimension]
                if stats["mean_1_5"] is not None:
                    rows.append({"board": board, "dimension": dimension, "model_name": model, "score": stats["mean_1_5"], "count": stats["count"]})
            board_dimension_rows.extend(ranked(rows))

    return ranked(overall_rows), dimension_rows, board_rows, board_dimension_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field) for field in fields}
            if "score" in out:
                out["score"] = fmt_score(out["score"])
            writer.writerow(out)



def write_table(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    write_csv(path.with_suffix(".csv"), rows, fields)


def write_summary_outputs(run_dir: Path, rows: list[dict[str, Any]], results_path: Path) -> None:
    rows = sorted(rows, key=lambda row: (str(row.get("sample_id")), str(row.get("model_name"))))
    summary = summarize(rows)
    (run_dir / "binding_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    model_rows = []
    for model, metrics in summary["by_model"].items():
        row = {"model_name": model}
        for metric in ALL_METRICS:
            row[f"{metric}_mean_1_5"] = metrics[metric]["mean_1_5"]
        for metric in ALL_METRICS:
            row[f"{metric}_count"] = metrics[metric]["count"]
        model_rows.append(row)
    write_table(
        run_dir / "binding_model_summary",
        model_rows,
        ["model_name"] + [f"{metric}_mean_1_5" for metric in ALL_METRICS] + [f"{metric}_count" for metric in ALL_METRICS],
    )

    per_sample_rows = []
    for row in rows:
        score_reasons = row.get("score_reasons") or {}
        per_sample_rows.append(
            {
                "task_id": row.get("task_id"),
                "sample_id": row.get("sample_id"),
                "model_name": row.get("model_name"),
                "video_path": row.get("video_path"),
                "error": row.get("error"),
                TOTAL_METRIC: score_from(row, TOTAL_METRIC),
                "Identity Binding": score_from(row, "Identity Binding"),
                "Attribute Binding": score_from(row, "Attribute Binding"),
                "Action Binding": score_from(row, "Action Binding"),
                "Identity Binding reason": score_reasons.get("Identity Binding", ""),
                "Attribute Binding reason": score_reasons.get("Attribute Binding", ""),
                "Action Binding reason": score_reasons.get("Action Binding", ""),
                "Binding Score reason": score_reasons.get(TOTAL_METRIC, ""),
                "num_subjects": row.get("num_subjects"),
                "has_audio": row.get("has_audio"),
                "critical_failures": "; ".join(row.get("critical_failures") or []),
                "minor_failures": "; ".join(row.get("minor_failures") or []),
                "comments": row.get("comments", ""),
            }
        )
    write_table(
        run_dir / "binding_per_sample",
        per_sample_rows,
        [
            "task_id",
            "sample_id",
            "model_name",
            "video_path",
            "error",
            TOTAL_METRIC,
            "Identity Binding",
            "Attribute Binding",
            "Action Binding",
            "Identity Binding reason",
            "Attribute Binding reason",
            "Action Binding reason",
            "Binding Score reason",
            "num_subjects",
            "has_audio",
            "critical_failures",
            "minor_failures",
            "comments",
        ],
    )

    item_rows = []
    for row in rows:
        checklist = row.get("checklist") or {}
        score_reasons = row.get("score_reasons") or {}
        failure_reasons = row.get("failure_reasons") or {}
        for item in CHECKLIST_ITEMS:
            value = str(checklist.get(item, "")).strip().upper()
            if value.startswith("[") and value.endswith("]"):
                value = value[1:-1].strip()
            item_rows.append(
                {
                    "task_id": row.get("task_id"),
                    "sample_id": row.get("sample_id"),
                    "model_name": row.get("model_name"),
                    "video_path": row.get("video_path"),
                    "item": item,
                    "value": checklist.get(item, ""),
                    "score_reason": failure_reasons.get(item, "") if value in {"PARTIAL", "NO"} else "",
                }
            )
    write_table(
        run_dir / "binding_item_detail",
        item_rows,
        ["task_id", "sample_id", "model_name", "video_path", "item", "value", "score_reason"],
    )

    overall_rank, dimension_rank, board_rank, board_dimension_rank = ranking_data(summary)
    write_table(run_dir / "model_ranking_overall_total", overall_rank, ["rank", "model_name", "score", "count"])
    write_table(run_dir / "model_ranking_overall_dimensions", dimension_rank, ["dimension", "rank", "model_name", "score", "count"])
    write_table(run_dir / "model_ranking_by_board_total", board_rank, ["board", "rank", "model_name", "score", "count"])
    write_table(run_dir / "model_ranking_by_board_dimensions", board_dimension_rank, ["board", "dimension", "rank", "model_name", "score", "count"])
    write_ranking_md(run_dir / "model_rankings.md", run_dir, summary, overall_rank, dimension_rank, board_rank, board_dimension_rank, results_path)


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
    dimension_rank: list[dict[str, Any]],
    board_rank: list[dict[str, Any]],
    board_dimension_rank: list[dict[str, Any]],
    results_path: Path,
) -> None:
    models = sorted(summary["by_model"])
    boards = sorted(summary["by_board"])
    model_counts = ", ".join(
        f"{model}={summary['by_model'][model][TOTAL_METRIC]['count']}" for model in models
    )
    lines = [
        "# Model Ranking Report",
        "",
        f"- Source: `{run_dir}`",
        f"- Results: `{results_path}`",
        f"- Models: {len(models)}",
        f"- Rows per model: {model_counts}",
        f"- Boards: {', '.join(boards)}",
        f"- Total metric: `{TOTAL_METRIC}`",
        "- Total score formula: equal-weighted mean of applicable sub-dimensions; sub-dimensions with all N/A checklist items are omitted.",
        "- Rank order: higher score is better; equal scores share the same rank.",
        "",
        "## Overall Total Ranking",
        "",
    ]
    lines.extend(
        md_table(
            ["Rank", "Model", TOTAL_METRIC, "Count"],
            [[row["rank"], row["model_name"], fmt_score(row["score"]), row["count"]] for row in overall_rank],
        )
    )
    lines.extend(["", "## Overall Sub-Dimension Rankings"])
    for dimension in SUB_DIMENSIONS:
        lines.extend(["", f"### {dimension}", ""])
        selected = [row for row in dimension_rank if row["dimension"] == dimension]
        lines.extend(
            md_table(
                ["Rank", "Model", "Score", "Count"],
                [[row["rank"], row["model_name"], fmt_score(row["score"]), row["count"]] for row in selected],
            )
        )
    lines.extend(["", "## Board Total Rankings"])
    for board in boards:
        lines.extend(["", f"### {board}", ""])
        selected = [row for row in board_rank if row["board"] == board]
        lines.extend(
            md_table(
                ["Rank", "Model", TOTAL_METRIC, "Count"],
                [[row["rank"], row["model_name"], fmt_score(row["score"]), row["count"]] for row in selected],
            )
        )
    lines.extend(["", "## Board Sub-Dimension Rankings"])
    for board in boards:
        lines.extend(["", f"### {board}"])
        for dimension in SUB_DIMENSIONS:
            lines.extend(["", f"#### {dimension}", ""])
            selected = [row for row in board_dimension_rank if row["board"] == board and row["dimension"] == dimension]
            lines.extend(
                md_table(
                    ["Rank", "Model", "Score", "Count"],
                    [[row["rank"], row["model_name"], fmt_score(row["score"]), row["count"]] for row in selected],
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_output_path(output: str | None, output_dir: Path, run_name: str | None) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"{run_name}_{timestamp}" if run_name else f"binding_8models_30samples_{timestamp}"
    return output_dir.expanduser().resolve() / folder / "binding_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate binding correctness with a Gemini MLLM judge.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--run-name", default="binding_8models_30samples")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--location", default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--models", default="all")
    parser.add_argument("--samples", default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    configure_proxy()
    tasks = filter_tasks(read_rows(args.manifest.expanduser().resolve()), parse_csv_set(args.models), parse_csv_set(args.samples), args.limit)
    output = build_output_path(args.output, args.output_dir, args.run_name)
    existing = [] if args.overwrite else read_rows(output)
    done = {row.get("task_id") for row in existing if not row.get("error")}
    pending = [task for task in tasks if task["task_id"] not in done]
    if args.overwrite:
        existing = []
        atomic_write_rows(output, existing)

    output.parent.mkdir(parents=True, exist_ok=True)
    latest = output.parent.parent / "latest_run.txt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(output.parent) + "\n", encoding="utf-8")
    print(f"Output: {output}", flush=True)
    print(f"Judge model: {args.model}; location: {args.location}; pending: {len(pending)}", flush=True)

    if pending:
        client = build_client(args.credentials.expanduser().resolve(), args.location)
        prompt_template = args.prompt.expanduser().resolve().read_text(encoding="utf-8")
        completed = 0
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(evaluate_task, client, task, prompt_template, args.model, args.timeout, args.temperature)
                for task in pending
            ]
            for future in as_completed(futures):
                result = future.result()
                existing.append(result)
                completed += 1
                atomic_write_rows(output, existing)
                status = "ok" if not result.get("error") else "error"
                print(f"[{completed}/{len(pending)}] {status} {result.get('task_id')}", flush=True)

    write_summary_outputs(output.parent, existing, output)
    print(f"Saved summary outputs to {output.parent}", flush=True)


if __name__ == "__main__":
    main()
