#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
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
DEFAULT_TASKS = SCRIPT_DIR / "inputs/ac_eval_tasks.json"
DEFAULT_PROMPTS_DIR = SCRIPT_DIR / "prompts"
DEFAULT_OUTPUTS_DIR = SCRIPT_DIR / "outputs"
DEFAULT_CREDENTIALS = Path(os.getenv("FINALEVAL_GEMINI_CREDENTIALS", ""))
DEFAULT_MODEL = os.getenv("AC_EVAL_MODEL") or os.getenv("IF_EVAL_MODEL") or os.getenv("RC_EVAL_MODEL") or "gemini-3.5-flash"
DEFAULT_VIDEO_FPS = float(os.getenv("MLLM_GEMINI_VIDEO_FPS", "4"))

VISUAL_EVENT_SOUND_ALIGNMENT = "Visual Event-Sound Alignment"
SPEECH_LIP_SYNC = "Speech-Lip Synchronization"
SOURCE_CORRECTNESS = "Source Correctness"
VOICE_TIMBRE_SIMILARITY = "Voice Timbre Similarity"
STABLE_SPEAKING_FRONTAL_FACE = "Stable Speaking Frontal Face"
AUDIO_CONSISTENCY = "Audio Consistency"
DIMENSIONS = [
    VISUAL_EVENT_SOUND_ALIGNMENT,
    SPEECH_LIP_SYNC,
    SOURCE_CORRECTNESS,
    VOICE_TIMBRE_SIMILARITY,
    STABLE_SPEAKING_FRONTAL_FACE,
]
AC_SCORE_DIMENSIONS = [VISUAL_EVENT_SOUND_ALIGNMENT, SPEECH_LIP_SYNC, SOURCE_CORRECTNESS, VOICE_TIMBRE_SIMILARITY]
ALL_DIMENSIONS = DIMENSIONS + [AUDIO_CONSISTENCY]
PROMPT_FILES = {
    VISUAL_EVENT_SOUND_ALIGNMENT: "visual_event_sound_alignment.md",
    SPEECH_LIP_SYNC: "speech_lip_synchronization.md",
    SOURCE_CORRECTNESS: "source_correctness.md",
    VOICE_TIMBRE_SIMILARITY: "voice_timbre_similarity.md",
    STABLE_SPEAKING_FRONTAL_FACE: "stable_speaking_frontal_face.md",
}
LEGACY_DIMENSION_ALIASES = {
    "Audio-Visual Synchronization": VISUAL_EVENT_SOUND_ALIGNMENT,
    "Event-Sound Matching": VISUAL_EVENT_SOUND_ALIGNMENT,
}
DIMENSION_NAME_ALIASES = {
    "visual_event_sound_alignment": VISUAL_EVENT_SOUND_ALIGNMENT,
    "event_sound_alignment": VISUAL_EVENT_SOUND_ALIGNMENT,
    "av_sync": VISUAL_EVENT_SOUND_ALIGNMENT,
    "event_sound_matching": VISUAL_EVENT_SOUND_ALIGNMENT,
    "speech_lip_sync": SPEECH_LIP_SYNC,
    "speech_lip_sync_mllm": SPEECH_LIP_SYNC,
    "speech_lip_sync_auto": SPEECH_LIP_SYNC,
    "lip_sync": SPEECH_LIP_SYNC,
    "source_correctness": SOURCE_CORRECTNESS,
    "source": SOURCE_CORRECTNESS,
    "voice_timbre_similarity": VOICE_TIMBRE_SIMILARITY,
    "voice-timbre-similarity": VOICE_TIMBRE_SIMILARITY,
    "voice_timbre": VOICE_TIMBRE_SIMILARITY,
    "timbre_similarity": VOICE_TIMBRE_SIMILARITY,
    "audio_timbre_similarity": VOICE_TIMBRE_SIMILARITY,
    "stable_speaking_frontal_face": STABLE_SPEAKING_FRONTAL_FACE,
    "speaking_stable_frontal": STABLE_SPEAKING_FRONTAL_FACE,
    "stable_frontal": STABLE_SPEAKING_FRONTAL_FACE,
    "speaking_frontal": STABLE_SPEAKING_FRONTAL_FACE,
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


def normalize_requested_dimensions(value: str | None) -> set[str] | None:
    requested = parse_csv_set(value)
    if requested:
        requested = {
            LEGACY_DIMENSION_ALIASES.get(item, DIMENSION_NAME_ALIASES.get(item.lower(), item))
            for item in requested
        }
    unknown = (requested or set()) - set(DIMENSIONS)
    if unknown:
        raise ValueError(f"Unknown dimensions: {', '.join(sorted(unknown))}")
    return requested


def dimension_methods(speech_lip_method: str) -> tuple[set[str], set[str]]:
    automatic_dimensions = {SPEECH_LIP_SYNC} if speech_lip_method == "auto" else set()
    mllm_dimensions = set(DIMENSIONS) - automatic_dimensions
    return mllm_dimensions, automatic_dimensions


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
    if mime_type.startswith("video/"):
        return types.Part(
            inline_data=types.Blob(data=file_path.read_bytes(), mime_type=mime_type),
            video_metadata=types.VideoMetadata(fps=DEFAULT_VIDEO_FPS),
        )
    return types.Part.from_bytes(data=file_path.read_bytes(), mime_type=mime_type)


def prompt_reference(ref: dict[str, Any]) -> dict[str, Any]:
    return {
        "placeholder": ref.get("placeholder"),
        "type": ref.get("type"),
        "description": ref.get("description"),
    }


def source_eval_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return spec.get("source_correctness_eval_spec") or spec


def source_checklist(spec: dict[str, Any]) -> list[dict[str, Any]]:
    source_spec = source_eval_spec(spec)
    checklist = source_spec.get("source_correctness_checklist")
    if checklist is None:
        checklist = source_spec.get("expected_sources")
    return checklist or []


def expected_speaker_references(spec: dict[str, Any]) -> list[dict[str, Any]]:
    source_spec = source_eval_spec(spec)
    explicit = source_spec.get("expected_speaker_references") or []
    refs_by_placeholder = {
        ref.get("placeholder"): ref
        for ref in (source_spec.get("references") or spec.get("references") or [])
        if isinstance(ref, dict) and ref.get("placeholder")
    }
    refs = []
    seen = set()
    for ref in explicit:
        if not isinstance(ref, dict):
            continue
        placeholder = ref.get("placeholder")
        key = placeholder or ref.get("path") or ref.get("description")
        if key and key not in seen:
            refs.append(ref)
            seen.add(key)
    for item in source_checklist(spec):
        if not isinstance(item, dict):
            continue
        ref = item.get("expected_speaker_reference") or refs_by_placeholder.get(item.get("expected_speaker"))
        if not isinstance(ref, dict):
            continue
        placeholder = ref.get("placeholder")
        key = placeholder or ref.get("path") or ref.get("description")
        if key and key not in seen:
            refs.append(ref)
            seen.add(key)
    return refs


def video_grounded_spec(spec: dict[str, Any], dim: str) -> dict[str, Any]:
    """Return the dimension-specific spec supplied to the Gemini judge."""
    if dim == SOURCE_CORRECTNESS:
        source_spec = source_eval_spec(spec)
        checklist = []
        for item in source_checklist(spec):
            if not isinstance(item, dict):
                continue
            row = {
                "source_id": item.get("source_id"),
                "expected_speaker": item.get("expected_speaker"),
            }
            ref = item.get("expected_speaker_reference")
            if isinstance(ref, dict):
                row["expected_speaker_reference"] = prompt_reference(ref)
            checklist.append(row)
        return {
            "sample_id": source_spec.get("sample_id") or spec.get("sample_id"),
            "source_sample_id": source_spec.get("source_sample_id") or spec.get("source_sample_id"),
            "source_board": source_spec.get("source_board") or spec.get("source_board"),
            "expected_speaker_references": [prompt_reference(ref) for ref in expected_speaker_references(spec)],
            "source_correctness_checklist": checklist,
        }
    if dim == STABLE_SPEAKING_FRONTAL_FACE:
        return {
            "sample_id": spec.get("sample_id"),
            "source_sample_id": spec.get("source_sample_id"),
            "source_board": spec.get("source_board"),
            "prompt": spec.get("prompt"),
            "chinese_prompt": spec.get("chinese_prompt"),
            "speech_prompt": spec.get("speech_prompt") or [],
            "metadata": {
                "duration_sec": (spec.get("metadata") or {}).get("duration_sec"),
                "has_speech": (spec.get("metadata") or {}).get("has_speech"),
            },
        }
    if dim == VOICE_TIMBRE_SIMILARITY:
        audio = reference_audio(spec)
        return {
            "sample_id": spec.get("sample_id"),
            "source_sample_id": spec.get("source_sample_id"),
            "source_board": spec.get("source_board"),
            "reference_audio": {
                "path": audio.get("path") if audio else None,
                "description": audio.get("description") if audio else None,
            },
            "metadata": {
                "has_speech": (spec.get("metadata") or {}).get("has_speech"),
                "duration_sec": (spec.get("metadata") or {}).get("duration_sec"),
            },
            "speech_prompt": spec.get("speech_prompt") or [],
        }

    references = []
    for ref in spec.get("references", []) or []:
        if not isinstance(ref, dict):
            continue
        references.append(prompt_reference(ref))

    metadata = {}
    raw_metadata = spec.get("metadata") or {}
    if raw_metadata.get("duration_sec") is not None:
        metadata["duration_sec"] = raw_metadata.get("duration_sec")

    clean = {
        "sample_id": spec.get("sample_id"),
        "source_sample_id": spec.get("source_sample_id"),
        "source_board": spec.get("source_board"),
        "references": references,
    }
    if metadata:
        clean["metadata"] = metadata
    return clean


def build_prompt(prompt_template: str, spec: dict[str, Any], dim: str) -> str:
    spec_json = json.dumps(video_grounded_spec(spec, dim), ensure_ascii=False, indent=2)
    return prompt_template.replace("{ac_eval_spec_json}", spec_json)


def build_contents(task: dict[str, Any], prompt_text: str, dim: str) -> list[Any]:
    spec = task["ac_eval_spec"]
    contents: list[Any] = [prompt_text]

    if dim == VOICE_TIMBRE_SIMILARITY:
        audio = reference_audio(spec)
        ref_path = audio.get("path") if audio else None
        contents.append("Reference/input speech audio for voice timbre comparison")
        if ref_path and Path(ref_path).exists():
            contents.append(file_part(ref_path))
        else:
            contents.append("WARNING: Reference/input audio file is missing or unreadable.")
    elif dim == SOURCE_CORRECTNESS:
        for ref in expected_speaker_references(spec):
            ref_path = ref.get("path")
            if not ref_path or not Path(ref_path).exists():
                continue
            contents.append(f"Expected speaker image {ref.get('placeholder')} ({ref.get('type')})")
            contents.append(file_part(ref_path))
    elif dim != VISUAL_EVENT_SOUND_ALIGNMENT:
        for ref in spec.get("references", []):
            ref_path = ref.get("path")
            if not ref_path or not Path(ref_path).exists():
                continue
            contents.append(f"Reference image {ref.get('placeholder')} ({ref.get('type')})")
            contents.append(file_part(ref_path))

    video_path = task.get("video_path")
    contents.append(f"Generated video for model {task.get('model_name')}")
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


def normalize_voice_score(value: Any) -> int | None:
    score = normalize_score(value)
    if score is None:
        return None
    rounded = int(round(score))
    return rounded if rounded in {1, 3, 5} else None


def reference_audio(spec: dict[str, Any]) -> dict[str, Any] | None:
    item = spec.get("reference_audio")
    return item if isinstance(item, dict) and item.get("path") else None


def normalize_source_results(payload: dict[str, Any], task: dict[str, Any]) -> list[dict[str, Any]]:
    expected = source_checklist(task["ac_eval_spec"])
    if not expected:
        return []

    raw_items = payload.get("source_results") or []
    raw_by_id = {}
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("source_id") or "").strip()
            if source_id and source_id not in raw_by_id:
                raw_by_id[source_id] = item

    normalized = []
    for expected_item in expected:
        source_id = str(expected_item.get("source_id") or "").strip()
        item = dict(raw_by_id.get(source_id) or {})
        item["source_id"] = source_id
        item["expected_speaker"] = expected_item.get("expected_speaker")
        if expected_item.get("expected_speaker_reference"):
            item["expected_speaker_reference"] = prompt_reference(expected_item["expected_speaker_reference"])
        if "expected_speech" in expected_item:
            item["expected_speech"] = expected_item.get("expected_speech")
        if "language" in expected_item:
            item["language"] = expected_item.get("language")
        if "score" in item:
            item["score"] = normalize_score(item.get("score"))
        else:
            item["score"] = None
        if "source_match" not in item:
            item["source_match"] = "missing"
        if "timbre_match" not in item:
            item["timbre_match"] = "not_inferable"
        if "audible_sound_or_speech" not in item:
            item["audible_sound_or_speech"] = ""
        if "observed_source" not in item:
            item["observed_source"] = ""
        if "reason" not in item:
            item["reason"] = "Judge did not return a result for this fixed speech-source checklist item."
        normalized.append(item)
    return normalized


def normalize_voice_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = payload.get("voice_results") or []
    normalized = []
    if not isinstance(raw_items, list):
        raw_items = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        item["voice_id"] = str(item.get("voice_id") or "speech_voice")
        item["score"] = normalize_voice_score(item.get("score"))
        if "voice_similarity" not in item:
            item["voice_similarity"] = "not_inferable"
        if "reference_voice_description" not in item:
            item["reference_voice_description"] = ""
        if "generated_voice_description" not in item:
            item["generated_voice_description"] = ""
        if "reason" not in item:
            item["reason"] = ""
        normalized.append(item)
    return normalized


def speech_lip_no_visible_mouth(payload: dict[str, Any]) -> bool:
    texts: list[str] = []
    for key in ("reason", "visible_mouth_motion", "alignment"):
        value = payload.get(key)
        if value:
            texts.append(str(value))
    for item in payload.get("lip_sync_errors") or []:
        texts.append(str(item))
    for speaker in payload.get("speaker_results") or []:
        if not isinstance(speaker, dict):
            continue
        for key in ("reason", "visible_mouth_motion", "alignment"):
            value = speaker.get(key)
            if value:
                texts.append(str(value))
    text = " ".join(texts).lower()
    no_visible_terms = (
        "off-screen",
        "off screen",
        "offscreen",
        "out of frame",
        "not visible",
        "no visible person",
        "no visible speaker",
        "no visible face",
        "no visible mouth",
        "no mouth is visible",
        "face and mouth are not visible",
        "face and mouth are completely off-screen",
        "speaker's face and mouth are not visible",
        "speaker's face and mouth are completely off-screen",
        "close-up of a hand",
        "close-up shot of the hand",
        "close-up of hands",
        "close-up shot of hands",
        "close-up of the object",
        "close-up shot of the object",
        "close-up of the french fries",
        "close-up shot of the french fries",
        "hand brushing the grass",
    )
    return any(term in text for term in no_visible_terms)


def normalize_dimension_result(dim: str, payload: dict[str, Any], task: dict[str, Any], requested: set[str] | None) -> dict[str, Any]:
    applicable_from_spec = bool(task["ac_eval_spec"].get("applicable_dimensions", {}).get(dim))
    if requested is not None and dim not in requested:
        return {"applicable": False, "score": None, "reason": "Not evaluated in this run."}
    if not isinstance(payload, dict):
        payload = {}
    if dim == SOURCE_CORRECTNESS and not source_checklist(task["ac_eval_spec"]):
        return {
            "dimension": dim,
            "applicable": False,
            "score": None,
            "reason": "No prompted speech line is specified for this sample.",
            "source_results": [],
            "source_errors": [],
        }
    if dim == SPEECH_LIP_SYNC and speech_lip_no_visible_mouth(payload):
        result = dict(payload)
        result["applicable"] = False
        result["score"] = None
        result["reason"] = str(payload.get("reason", "Speaking face/mouth is not visible during speech; excluded from scoring."))
        result["dimension"] = dim
        return result
    if dim == VOICE_TIMBRE_SIMILARITY:
        result = dict(payload)
        normalized_items = normalize_voice_results(payload)
        item_scores = [item["score"] for item in normalized_items if item.get("score") in {1, 3, 5}]
        applicable = bool(payload.get("applicable", applicable_from_spec)) and bool(reference_audio(task["ac_eval_spec"]))
        score = normalize_voice_score(payload.get("score")) if applicable else None
        if score is None and item_scores:
            score = item_scores[0]
        if not applicable or score is None:
            applicable = False
            score = None
        result["voice_results"] = normalized_items
        result["applicable"] = applicable
        result["score"] = score
        result["reason"] = str(payload.get("reason", ""))
        result["dimension"] = dim
        return result
    applicable = bool(payload.get("applicable", applicable_from_spec))
    score = normalize_score(payload.get("score")) if applicable else None
    result = dict(payload)
    if dim == VISUAL_EVENT_SOUND_ALIGNMENT:
        normalized_events = []
        sound_values = []
        timing_values = []
        for raw_event in payload.get("event_results") or []:
            if not isinstance(raw_event, dict):
                continue
            event = dict(raw_event)
            event_sound_score = normalize_score(event.get("sound_consistency_score"))
            event_timing_score = normalize_score(event.get("temporal_alignment_score"))
            event_values = [value for value in (event_sound_score, event_timing_score) if value is not None]
            event["sound_consistency_score"] = event_sound_score
            event["temporal_alignment_score"] = event_timing_score
            event["score"] = int(sum(event_values) / len(event_values) + 0.5) if event_values else None
            if event_sound_score is not None:
                sound_values.append(event_sound_score)
            if event_timing_score is not None:
                timing_values.append(event_timing_score)
            normalized_events.append(event)
        sound_score = sum(sound_values) / len(sound_values) if sound_values else None
        timing_score = sum(timing_values) / len(timing_values) if timing_values else None
        subscore_values = [value for value in (sound_score, timing_score) if value is not None]
        applicable = bool(subscore_values)
        score = int(sum(subscore_values) / len(subscore_values) + 0.5) if subscore_values else None
        result["event_results"] = normalized_events
        result["sound_consistency_score"] = sound_score
        result["temporal_alignment_score"] = timing_score
    result["applicable"] = applicable
    result["score"] = score
    result["reason"] = str(payload.get("reason", ""))
    result["dimension"] = dim
    if dim == SOURCE_CORRECTNESS:
        result["source_results"] = normalize_source_results(payload, task)
    return result


def aggregate_scores(scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = []
    for dim in AC_SCORE_DIMENSIONS:
        row = scores.get(dim, {})
        if row.get("applicable") and row.get("score") is not None:
            values.append(float(row["score"]))
    return {
        "applicable": bool(values),
        "score": sum(values) / len(values) if values else None,
        "reason": "Average over applicable AC dimensions with valid scores.",
    }


def result_row(
    task: dict[str, Any],
    scores: dict[str, Any],
    raw_responses: dict[str, Any],
    errors: dict[str, str],
) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "sample_id": task["sample_id"],
        "model_name": task["model_name"],
        "video_path": task["video_path"],
        "source_json": task["ac_eval_spec"].get("source_json"),
        "ac_eval_spec": task["ac_eval_spec"],
        "scores": scores,
        "raw_responses": raw_responses,
        "errors": errors,
        "error": "; ".join(f"{dim}: {err}" for dim, err in errors.items()) if errors else None,
    }


def evaluate_dimension(
    client,
    task: dict[str, Any],
    dim: str,
    prompt_template: str,
    model: str,
    timeout: int,
    temperature: float,
) -> tuple[dict[str, Any] | None, str | None]:
    prompt_text = build_prompt(prompt_template, task["ac_eval_spec"], dim)
    contents = build_contents(task, prompt_text, dim)
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
            return parse_json_response(response.text or "{}"), None
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(2 * (attempt + 1))
    return None, last_error or "unknown error"


def dry_run_result(
    task: dict[str, Any],
    dimensions: set[str] | None,
    mllm_dimensions: set[str],
) -> dict[str, Any]:
    scores = {}
    raw_responses = {}
    for dim in DIMENSIONS:
        applicable = bool(task["ac_eval_spec"].get("applicable_dimensions", {}).get(dim))
        if dimensions is not None and dim not in dimensions:
            applicable = False
        scores[dim] = {
            "dimension": dim,
            "applicable": applicable,
            "score": None,
            "reason": "Dry run: Gemini was not called." if dim in mllm_dimensions else "Dry run: automatic metric was not called.",
        }
        raw_responses[dim] = None
    scores[AUDIO_CONSISTENCY] = aggregate_scores(scores)
    return result_row(task, scores, raw_responses, {})


def evaluate_task(
    client,
    task: dict[str, Any],
    prompt_templates: dict[str, str],
    model: str,
    timeout: int,
    temperature: float,
    dimensions: set[str] | None,
    mllm_dimensions: set[str],
    automatic_dimensions: set[str],
) -> dict[str, Any]:
    scores = {}
    raw_responses = {}
    errors = {}
    for dim in DIMENSIONS:
        if dimensions is not None and dim not in dimensions:
            scores[dim] = {"dimension": dim, "applicable": False, "score": None, "reason": "Not evaluated in this run."}
            raw_responses[dim] = None
            continue
        if dim in automatic_dimensions:
            scores[dim] = {
                "dimension": dim,
                "method": "latentsync_syncnet_best_face_placeholder",
                "applicable": bool(task["ac_eval_spec"].get("applicable_dimensions", {}).get(dim)),
                "score": None,
                "reason": "Run automatic Speech-Lip Synchronization metric to fill this score.",
            }
            raw_responses[dim] = None
            continue
        if dim not in mllm_dimensions:
            scores[dim] = {"dimension": dim, "applicable": False, "score": None, "reason": "No evaluation method selected."}
            raw_responses[dim] = None
            continue
        if dim == SOURCE_CORRECTNESS and not source_checklist(task["ac_eval_spec"]):
            scores[dim] = {
                "dimension": dim,
                "applicable": False,
                "score": None,
                "reason": "No expected speaker is specified for Source Correctness.",
                "source_results": [],
                "source_errors": [],
            }
            raw_responses[dim] = None
            continue
        if dim == VOICE_TIMBRE_SIMILARITY and not reference_audio(task["ac_eval_spec"]):
            scores[dim] = {
                "dimension": dim,
                "applicable": False,
                "score": None,
                "reason": "No reference/input audio is specified for Voice Timbre Similarity.",
                "voice_results": [],
                "voice_errors": [],
            }
            raw_responses[dim] = None
            continue
        payload, error = evaluate_dimension(client, task, dim, prompt_templates[dim], model, timeout, temperature)
        raw_responses[dim] = payload
        if error:
            errors[dim] = error
            scores[dim] = {
                "dimension": dim,
                "applicable": bool(task["ac_eval_spec"].get("applicable_dimensions", {}).get(dim)),
                "score": None,
                "reason": f"Gemini evaluation failed: {error}",
            }
        else:
            scores[dim] = normalize_dimension_result(dim, payload or {}, task, dimensions)
    scores[AUDIO_CONSISTENCY] = aggregate_scores(scores)
    return result_row(task, scores, raw_responses, errors)


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
    folder_name = f"{run_name}_{timestamp}" if run_name else f"ac_{timestamp}"
    return Path(output_dir).expanduser().resolve() / folder_name / "ac_results.json"


def write_latest_run(output_path: Path) -> None:
    latest_path = output_path.parent / "latest_run.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(output_path.parent) + "\n", encoding="utf-8")


def load_prompt_templates(prompts_dir: Path, required_dimensions: set[str]) -> dict[str, str]:
    templates = {}
    for dim, filename in PROMPT_FILES.items():
        if dim not in required_dimensions:
            continue
        path = prompts_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found for {dim}: {path}")
        templates[dim] = path.read_text(encoding="utf-8")
    return templates


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Audio Consistency with video-grounded Gemini MLLM judges.")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--prompts-dir", default=str(DEFAULT_PROMPTS_DIR))
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUTS_DIR))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--location", default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--models", default="all")
    parser.add_argument("--samples", default="all")
    parser.add_argument("--dimensions", default="all")
    parser.add_argument("--speech-lip-method", choices=["auto", "mllm"], default=os.getenv("AC_SPEECH_LIP_METHOD", "auto"))
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dimensions = normalize_requested_dimensions(args.dimensions)
    mllm_dimensions, automatic_dimensions = dimension_methods(args.speech_lip_method)

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

    required_prompt_dimensions = mllm_dimensions if dimensions is None else dimensions & mllm_dimensions
    prompt_templates = load_prompt_templates(Path(args.prompts_dir).expanduser().resolve(), required_prompt_dimensions)
    if args.dry_run:
        for task in pending:
            append_row(output, dry_run_result(task, dimensions, mllm_dimensions))
        return

    credentials_path = args.credentials.expanduser().resolve()
    if not credentials_path.exists():
        raise FileNotFoundError(f"Gemini credentials file not found: {credentials_path}")
    client = build_client(credentials_path, args.location)
    print(f"Judge model: {args.model}")
    print("Credentials: configured")
    print(f"Location: {args.location}")
    print(f"Dimensions: {','.join(dimensions) if dimensions else 'all'}")
    print(f"Speech-Lip method: {args.speech_lip_method}")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                evaluate_task,
                client,
                task,
                prompt_templates,
                args.model,
                args.timeout,
                args.temperature,
                dimensions,
                mllm_dimensions,
                automatic_dimensions,
            )
            for task in pending
        ]
        completed = as_completed(futures)
        iterator = tqdm(completed, total=len(futures), desc="Audio Consistency") if tqdm else completed
        for future in iterator:
            append_row(output, future.result())


if __name__ == "__main__":
    main()
