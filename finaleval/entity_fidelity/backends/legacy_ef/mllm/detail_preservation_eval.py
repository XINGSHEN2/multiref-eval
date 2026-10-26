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

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "inputs" / "manifest.json"
DEFAULT_PROMPT = SCRIPT_DIR / "MLLM_Prompt" / "detail preservetion.md"
DEFAULT_OUTPUTS_DIR = SCRIPT_DIR / "outputs" / "detail_preservation"
DEFAULT_CREDENTIALS = Path(os.getenv("FINALEVAL_GEMINI_CREDENTIALS", ""))
DEFAULT_MODEL = os.getenv("DP_EVAL_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-3.5-flash"
DEFAULT_VIDEO_FPS = float(os.getenv("MLLM_GEMINI_VIDEO_FPS", "4"))
TOTAL_METRIC = "Detail Preservation Score"
SUB_DIMENSIONS = ["Human Detail", "Object Detail", "Background Detail"]
ALL_METRICS = [*SUB_DIMENSIONS, TOTAL_METRIC]
TOTAL_WEIGHTS = {"Human Detail": 1.0, "Object Detail": 1.0, "Background Detail": 1.0}
SECTION_KEY_TO_DIMENSION = {
    "human_detail": "Human Detail",
    "object_detail": "Object Detail",
    "background_detail": "Background Detail",
}
DIMENSION_TO_SECTION_KEY = {value: key for key, value in SECTION_KEY_TO_DIMENSION.items()}
CHECKLIST_ITEMS_BY_SECTION = {
    "human_detail": ["H1", "H2", "H3"],
    "object_detail": ["O1", "O2", "O3"],
    "background_detail": ["BG1", "BG2", "BG3"],
}
CHECKLIST_ITEMS = [item for items in CHECKLIST_ITEMS_BY_SECTION.values() for item in items]
ITEM_VALUE_SCORES = {"YES": 5.0, "PARTIAL": 3.0, "NO": 1.0}
MISSING_OR_REPLACED_RE = re.compile(
    r"main .*missing|missing|absent|absence|replaced|completely lost|"
    r"entirely missing|does not appear|not present|omitted|category is missing",
    re.IGNORECASE,
)



def configure_proxy() -> None:
    if os.getenv("DP_NO_PROXY") == "1":
        for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
            os.environ.pop(key, None)
        return
    proxy = os.getenv("DP_PROXY_URL", "http://oversea-squid2.ko.txyun:11080")
    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        os.environ[key] = proxy


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


def file_part(path: str):
    from google.genai import types

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


def weighted_total_score(subscores: dict[str, Any]) -> float | None:
    values = []
    for dimension in SUB_DIMENSIONS:
        score = normalize_score(subscores.get(dimension))
        if score is not None:
            values.append(score)
    return mean(values) if values else None


def checklist_label(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if text in {"", "NAN", "NONE", "NULL"}:
        return "N/A"
    return text


def section_reason_from_row(row: dict[str, Any], section: str) -> str:
    section_reasons = row.get("section_reasons")
    if isinstance(section_reasons, dict):
        return str(section_reasons.get(section, "") or "")
    raw = row.get("raw_response")
    if isinstance(raw, dict) and isinstance(raw.get("section_reasons"), dict):
        return str(raw["section_reasons"].get(section, "") or "")
    return ""


def strict_section_score(
    checklist: dict[str, dict[str, str]],
    section: str,
    section_reason: str = "",
) -> float | None:
    labels = [
        checklist_label(checklist.get(section, {}).get(item))
        for item in CHECKLIST_ITEMS_BY_SECTION[section]
    ]
    values = [ITEM_VALUE_SCORES[label] for label in labels if label in ITEM_VALUE_SCORES]
    if not values:
        return None
    score = mean(values)
    cap = 5.0
    if "NO" in labels:
        cap = min(cap, 3.0)
    if labels.count("PARTIAL") >= 2:
        cap = min(cap, 4.0)
    if MISSING_OR_REPLACED_RE.search(section_reason or ""):
        cap = min(cap, 2.0)
    return min(score, cap)


def strict_scores_from_checklist(
    checklist: dict[str, dict[str, str]],
    section_reasons: dict[str, str] | None = None,
) -> tuple[dict[str, float | None], float | None]:
    section_reasons = section_reasons or {}
    scores: dict[str, float | None] = {}
    labels_by_section: dict[str, list[str]] = {}
    for section, dimension in SECTION_KEY_TO_DIMENSION.items():
        reason = str(section_reasons.get(section, "") or "")
        scores[dimension] = strict_section_score(checklist, section, reason)
        labels_by_section[section] = [
            checklist_label(checklist.get(section, {}).get(item))
            for item in CHECKLIST_ITEMS_BY_SECTION[section]
        ]

    values = [score for score in scores.values() if score is not None]
    if not values:
        return scores, None
    total = mean(values)
    cap = 5.0
    if "NO" in labels_by_section.get("object_detail", []):
        cap = min(cap, 4.0)
    if sum(1 for score in values if score <= 3.0) >= 2:
        cap = min(cap, 3.5)
    return scores, min(total, cap)


def sanitize_comment(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    filtered = []
    for sentence in sentences:
        lower = sentence.lower()
        if "score is compressed" in lower or "compressed to 4" in lower or "avoid extreme values" in lower:
            continue
        filtered.append(sentence.strip())
    return " ".join(sentence for sentence in filtered if sentence).strip()


def bool_map(value: Any) -> dict[str, bool]:
    keys = [
        "outfit_change",
        "hairstyle_change",
        "hair_accessory_change",
        "accessory_change",
        "object_change",
        "background_change",
    ]
    if not isinstance(value, dict):
        value = {}
    return {key: bool(value.get(key, False)) for key in keys}


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    checklist = normalize_checklist(payload.get("checklist", {}))
    section_reasons = normalize_section_reasons(payload.get("section_reasons", {}))
    subscores = normalize_section_scores(payload.get("section_scores", {}), checklist, section_reasons)
    _, score = strict_scores_from_checklist(checklist, section_reasons)
    if score is None:
        score = weighted_total_score(subscores)
    if score is None:
        score = normalize_score(payload.get("score"))
    if score is None:
        raise ValueError("Missing or invalid score")
    failures = payload.get("major_failures", [])
    if not isinstance(failures, list):
        failures = [str(failures)]
    return {
        "applicable_sections": normalize_applicable_sections(payload.get("applicable_sections", {})),
        "section_reasons": section_reasons,
        "prompt_instructed_changes": bool_map(payload.get("prompt_instructed_changes")),
        "checklist": checklist,
        "subscores": subscores,
        "score": score,
        "major_failures": [str(item) for item in failures],
        "comments": sanitize_comment(payload.get("comments", "")),
    }


def normalize_section_reasons(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        value = {}
    return {section: str(value.get(section, "")).strip() for section in CHECKLIST_ITEMS_BY_SECTION}


def normalize_applicable_sections(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        value = {}
    return {
        section: bool(value.get(section, False))
        for section in CHECKLIST_ITEMS_BY_SECTION
    }


def normalize_checklist(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        value = {}
    checklist = {
        section: {item: "" for item in items}
        for section, items in CHECKLIST_ITEMS_BY_SECTION.items()
    }
    # Backward-compatible flat checklist: old prompt only had human items.
    for item in CHECKLIST_ITEMS_BY_SECTION["human_detail"]:
        if item in value:
            checklist["human_detail"][item] = str(value.get(item, "")).strip()
    for section, items in CHECKLIST_ITEMS_BY_SECTION.items():
        section_value = value.get(section, {})
        if isinstance(section_value, dict):
            for item in items:
                checklist[section][item] = str(section_value.get(item, checklist[section][item])).strip()
    return checklist


def checklist_value_score(value: Any) -> float | None:
    return ITEM_VALUE_SCORES.get(checklist_label(value))


def score_section_from_checklist(
    checklist: dict[str, dict[str, str]],
    section: str,
    section_reason: str = "",
) -> float | None:
    return strict_section_score(checklist, section, section_reason)


def normalize_section_scores(
    value: Any,
    checklist: dict[str, dict[str, str]],
    section_reasons: dict[str, str] | None = None,
) -> dict[str, float | None]:
    if not isinstance(value, dict):
        value = {}
    scores, _ = strict_scores_from_checklist(checklist, section_reasons or {})
    for section, dimension in SECTION_KEY_TO_DIMENSION.items():
        if scores[dimension] is None:
            scores[dimension] = normalize_score(value.get(section))
    return scores


def reference_flags(sample_id: str, references: list[dict[str, Any]]) -> dict[str, Any]:
    board = board_from_sample(sample_id)
    human_refs = []
    object_refs = []
    background_refs = []
    for ref in references:
        ref_type = str(ref.get("type", "")).lower()
        path = str(ref.get("path", "")).lower()
        description = str(ref.get("description", "")).lower()
        target = " ".join([ref_type, path, description])
        if ref_type == "human" or "/human/" in path or "person" in target:
            human_refs.append(ref)
        if ref_type == "object" or "/object/" in path:
            object_refs.append(ref)
        if ref_type in {"scene", "background"} or "/scene/" in path or "/background/" in path or "background" in target:
            background_refs.append(ref)
    return {
        "source_board": board,
        "has_human_reference": bool(human_refs),
        "has_object_reference": bool(object_refs),
        "has_background_reference": bool(background_refs) and board in {"board2", "board3"},
        "human_reference_count": len(human_refs),
        "object_reference_count": len(object_refs),
        "background_reference_count": len(background_refs) if board in {"board2", "board3"} else 0,
    }


def build_task(row: dict[str, Any]) -> dict[str, Any]:
    spec = row.get("if_eval_spec") or {}
    references = spec.get("references") or spec.get("image_items") or []
    return {
        "task_id": row["task_id"],
        "sample_id": row["sample_id"],
        "model_name": row["model_name"],
        "video_path": row.get("video_path"),
        "source_json": row.get("source_json") or spec.get("source_json"),
        "prompt": spec.get("prompt", ""),
        "chinese_prompt": spec.get("chinese_prompt", ""),
        "references": references,
        "reference_flags": reference_flags(str(row["sample_id"]), references),
    }


def filter_tasks(
    rows: list[dict[str, Any]],
    models: set[str] | None,
    samples: set[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    tasks = []
    seen_samples: set[str] = set()
    for row in rows:
        if models and row.get("model_name") not in models:
            continue
        if samples and row.get("sample_id") not in samples:
            continue
        tasks.append(build_task(row))
        seen_samples.add(str(row.get("sample_id")))
        if limit and len(tasks) >= limit:
            break
    if not tasks:
        raise ValueError("No tasks to evaluate")
    print(f"Models: {len({task['model_name'] for task in tasks})}; samples: {len(seen_samples)}; tasks: {len(tasks)}")
    return tasks


def build_contents(task: dict[str, Any], prompt_template: str) -> list[Any]:
    contents: list[Any] = [
        prompt_template,
        (
            "Reference applicability flags computed from the manifest. Use these flags "
            "to decide which detail sections are applicable:\n"
            + json.dumps(task.get("reference_flags", {}), ensure_ascii=False, indent=2)
        ),
        (
            "Evaluation context:\n"
            f"- Sample ID: {task['sample_id']}\n"
            f"- Generated-video model: {task['model_name']}\n"
            f"- Text prompt: {task.get('prompt', '')}\n"
            f"- Chinese text prompt: {task.get('chinese_prompt', '')}\n"
        ),
    ]
    for ref in task.get("references", []):
        ref_path = ref.get("path")
        if not ref_path or not Path(ref_path).exists():
            continue
        label = (
            f"Reference image {ref.get('placeholder', '')}: "
            f"type={ref.get('type', '')}; description={ref.get('description', '')}; path={ref_path}"
        )
        contents.append(label)
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
        "score": payload.get("score") if payload else None,
        "subscores": payload.get("subscores") if payload else {},
        "applicable_sections": payload.get("applicable_sections") if payload else {},
        "checklist": payload.get("checklist") if payload else {},
        "section_reasons": payload.get("section_reasons") if payload else {},
        "prompt_instructed_changes": payload.get("prompt_instructed_changes") if payload else {},
        "major_failures": payload.get("major_failures") if payload else [],
        "comments": payload.get("comments") if payload else "",
        "raw_response": raw_response,
        "error": error,
    }


def normalize_cached_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if out.get("error"):
        out["comments"] = sanitize_comment(out.get("comments", ""))
        return out

    subscores = out.get("subscores") if isinstance(out.get("subscores"), dict) else {}
    payload = {
        "score": out.get("score"),
        "section_scores": {
            section: subscores.get(dimension)
            for section, dimension in SECTION_KEY_TO_DIMENSION.items()
        },
        "applicable_sections": out.get("applicable_sections", {}),
        "prompt_instructed_changes": out.get("prompt_instructed_changes", {}),
        "section_reasons": out.get("section_reasons", {}),
        "checklist": out.get("checklist", {}),
        "major_failures": out.get("major_failures", []),
        "comments": out.get("comments", ""),
    }
    try:
        normalized = normalize_payload(payload)
    except Exception:  # noqa: BLE001
        out["comments"] = sanitize_comment(out.get("comments", ""))
        return out

    out["score"] = normalized["score"]
    out["subscores"] = normalized["subscores"]
    out["applicable_sections"] = normalized["applicable_sections"]
    out["checklist"] = normalized["checklist"]
    out["section_reasons"] = normalized["section_reasons"]
    out["prompt_instructed_changes"] = normalized["prompt_instructed_changes"]
    out["major_failures"] = normalized["major_failures"]
    out["comments"] = normalized["comments"]
    return out


def evaluate_task(
    client,
    task: dict[str, Any],
    prompt_template: str,
    model: str,
    timeout: int,
    temperature: float,
) -> dict[str, Any]:
    from google.genai import types

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
    checklist = row.get("checklist") if isinstance(row.get("checklist"), dict) else {}
    section_reasons = row.get("section_reasons") if isinstance(row.get("section_reasons"), dict) else {}
    if not section_reasons:
        section_reasons = {section: section_reason_from_row(row, section) for section in CHECKLIST_ITEMS_BY_SECTION}
    strict_subscores, strict_total = strict_scores_from_checklist(checklist, section_reasons)
    if metric == TOTAL_METRIC:
        return strict_total or weighted_total_score(strict_subscores) or normalize_score(row.get("score"))
    if metric in strict_subscores and strict_subscores[metric] is not None:
        return strict_subscores[metric]
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
            for field, value in list(out.items()):
                if isinstance(value, float):
                    out[field] = fmt_score(value)
            if "score" in out:
                out["score"] = fmt_score(out["score"])
            writer.writerow(out)


def write_summary_outputs(run_dir: Path, rows: list[dict[str, Any]], results_path: Path) -> None:
    rows = sorted(rows, key=lambda row: (str(row.get("sample_id")), str(row.get("model_name"))))
    summary = summarize(rows)
    (run_dir / "detail_preservation_summary.json").write_text(
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
    write_csv(
        run_dir / "detail_preservation_model_summary.csv",
        model_rows,
        ["model_name"] + [f"{metric}_mean_1_5" for metric in ALL_METRICS] + [f"{metric}_count" for metric in ALL_METRICS],
    )

    per_sample_rows = []
    for row in rows:
        section_reasons = row.get("section_reasons") or {}
        per_sample_rows.append(
            {
                "task_id": row.get("task_id"),
                "sample_id": row.get("sample_id"),
                "model_name": row.get("model_name"),
                "video_path": row.get("video_path"),
                "error": row.get("error"),
                TOTAL_METRIC: score_from(row, TOTAL_METRIC),
                "Human Detail": score_from(row, "Human Detail"),
                "Object Detail": score_from(row, "Object Detail"),
                "Background Detail": score_from(row, "Background Detail"),
                "Human Detail reason": section_reasons.get("human_detail", ""),
                "Object Detail reason": section_reasons.get("object_detail", ""),
                "Background Detail reason": section_reasons.get("background_detail", ""),
                "major_failures": "; ".join(row.get("major_failures") or []),
                "comments": row.get("comments", ""),
            }
        )
    write_csv(
        run_dir / "detail_preservation_per_sample.csv",
        per_sample_rows,
        [
            "task_id",
            "sample_id",
            "model_name",
            "video_path",
            "error",
            TOTAL_METRIC,
            "Human Detail",
            "Object Detail",
            "Background Detail",
            "Human Detail reason",
            "Object Detail reason",
            "Background Detail reason",
            "major_failures",
            "comments",
        ],
    )

    item_rows = []
    for row in rows:
        checklist = row.get("checklist") or {}
        for section, items in CHECKLIST_ITEMS_BY_SECTION.items():
            section_checklist = checklist.get(section, {}) if isinstance(checklist, dict) else {}
            for item in items:
                item_rows.append(
                    {
                        "task_id": row.get("task_id"),
                        "sample_id": row.get("sample_id"),
                        "model_name": row.get("model_name"),
                        "video_path": row.get("video_path"),
                        "section": section,
                        "item": item,
                        "value": section_checklist.get(item, ""),
                    }
                )
    write_csv(
        run_dir / "detail_preservation_item_detail.csv",
        item_rows,
        ["task_id", "sample_id", "model_name", "video_path", "section", "item", "value"],
    )

    overall_rank, dimension_rank, board_rank, board_dimension_rank = ranking_data(summary)
    write_csv(run_dir / "model_ranking_overall_total.csv", overall_rank, ["rank", "model_name", "score", "count"])
    write_csv(run_dir / "model_ranking_overall_dimensions.csv", dimension_rank, ["dimension", "rank", "model_name", "score", "count"])
    write_csv(run_dir / "model_ranking_by_board_total.csv", board_rank, ["board", "rank", "model_name", "score", "count"])
    write_csv(run_dir / "model_ranking_by_board_dimensions.csv", board_dimension_rank, ["board", "dimension", "rank", "model_name", "score", "count"])
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
    lines = [
        "# Model Ranking Report",
        "",
        f"- Source: `{run_dir}`",
        f"- Results: `{results_path}`",
        f"- Models: {len(models)}",
        "- Rows per model: " + ", ".join("{}={}".format(model, summary["by_model"][model][TOTAL_METRIC]["count"]) for model in models),
        f"- Boards: {', '.join(boards)}",
        f"- Total metric: `{TOTAL_METRIC}`",
        "- Total score formula: strict checklist item scoring with section and total caps.",
        "- Item scores: YES=5, PARTIAL=3, NO=1; N/A/SKIP omitted.",
        "- Section caps: any NO caps that section at 3; two or more PARTIAL caps at 4; missing/replaced main reference caps at 2.",
        "- Total caps: object_detail containing NO caps total at 4; two or more sections <=3 cap total at 3.5.",
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
        if selected:
            lines.extend(
                md_table(
                    ["Rank", "Model", "Score", "Count"],
                    [[row["rank"], row["model_name"], fmt_score(row["score"]), row["count"]] for row in selected],
                )
            )
        else:
            lines.append("No applicable samples.")
    lines.extend(["", "## Board Total Rankings"])
    for board in boards:
        lines.extend(["", f"### {board}", ""])
        selected = [row for row in board_rank if row["board"] == board]
        if selected:
            lines.extend(
                md_table(
                    ["Rank", "Model", TOTAL_METRIC, "Count"],
                    [[row["rank"], row["model_name"], fmt_score(row["score"]), row["count"]] for row in selected],
                )
            )
        else:
            lines.append("No applicable samples.")
    lines.extend(["", "## Board Sub-Dimension Rankings"])
    for board in boards:
        lines.extend(["", f"### {board}"])
        for dimension in SUB_DIMENSIONS:
            lines.extend(["", f"#### {dimension}", ""])
            selected = [row for row in board_dimension_rank if row["board"] == board and row["dimension"] == dimension]
            if selected:
                lines.extend(
                    md_table(
                        ["Rank", "Model", "Score", "Count"],
                        [[row["rank"], row["model_name"], fmt_score(row["score"]), row["count"]] for row in selected],
                    )
                )
            else:
                lines.append("No applicable samples.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_output_path(output: str | None, output_dir: Path, run_name: str | None) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"{run_name}_{timestamp}" if run_name else f"dp_8models_30samples_{timestamp}"
    return output_dir.expanduser().resolve() / folder / "detail_preservation_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detail preservation with a Gemini MLLM judge.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--run-name", default="dp_8models_30samples")
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
        write_summary_outputs(output.parent, existing, output)
        print(f"Rewrote cached results and summary outputs to {output.parent}", flush=True)
        return

    configure_proxy()
    tasks = filter_tasks(read_rows(args.manifest.expanduser().resolve()), parse_csv_set(args.models), parse_csv_set(args.samples), args.limit)
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
