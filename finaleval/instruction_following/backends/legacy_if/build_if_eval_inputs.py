#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(os.getenv("FINALEVAL_WORKSPACE_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_INPUT_DIR = Path(os.getenv("FINALEVAL_METADATA_DIR", SCRIPT_DIR / "inputs" / "metadata"))
DEFAULT_TASKS = SCRIPT_DIR / "inputs/if_eval_tasks.json"
DEFAULT_CHECKLISTS = SCRIPT_DIR / "inputs/if_checklists.json"
DEFAULT_CHECKLISTS_DIR = SCRIPT_DIR / "inputs/if_checklists_by_sample"
DEFAULT_CHECKLIST_PROMPT = SCRIPT_DIR / "if_judge_prompts_v3/if_checklist_generation_prompt_v2.md"
DEFAULT_CREDENTIALS = Path(os.getenv("FINALEVAL_GEMINI_CREDENTIALS", ""))
DEFAULT_GEMINI_MODEL = os.getenv("IF_CHECKLIST_MODEL") or os.getenv("IF_EVAL_MODEL") or os.getenv("RC_EVAL_MODEL") or "gemini-3.1-pro-preview"
DEFAULT_EXTRA_MODEL_VIDEO_DIRS = {}

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
NON_SPEECH_AUDIO_PATTERNS = [
    r"\bmusic\b",
    r"\bambien(?:ce|t)\b",
    r"\benvironmental sound",
    r"\bfoley\b",
    r"\bsound effect",
    r"\bsound of\b",
    r"\bfootsteps?\b",
    r"\brustl",
    r"\bclick",
    r"\bmovement sound",
    r"\bsilence\b",
    r"\bsilent\b",
    r"\bquiet\b",
    r"\btrack\b",
    r"\bnoise\b",
    "音效",
    "音乐",
    "环境声",
    "脚步",
    "沙沙",
    "咔",
    "点击",
    "静音",
    "无声",
]
SPEECH_AUDIO_PATTERNS = [
    r"\bsays?\b",
    r"\bspeaks?\b",
    r"\bspeech\b",
    r"\bspoken\b",
    r"\bdialog(?:ue|)\b",
    r"\bvoice\b",
    r"\bspeaker\b",
    r"\blip\b",
    r"\bmouth\b",
    r"\btalking\b",
    "对白",
    "台词",
    "说话",
    "说道",
    "语音",
    "声线",
    "口型",
]

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def parse_csv_set(value: str | None) -> set[str] | None:
    if not value or value.lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_extra_model_dirs(value: str | None) -> dict[str, Path]:
    if not value:
        return {}
    result = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --extra-model-dirs item: {item!r}; expected model=/path")
        model_name, model_dir = item.split("=", 1)
        model_name = model_name.strip()
        model_dir = model_dir.strip()
        if not model_name or not model_dir:
            raise ValueError(f"Invalid --extra-model-dirs item: {item!r}; expected model=/path")
        result[model_name] = Path(model_dir).expanduser()
    return result


def placeholder_from_key(value: str) -> str:
    value = str(value or "").strip()
    if value.startswith("@"):
        return "@" + value[1:].capitalize()
    if value.lower().startswith("image"):
        return "@" + value.capitalize()
    return value


def sentence_chunks(prompt: str) -> list[str]:
    chunks = []
    for part in re.split(r"(?<=[.!?。！？])\s+", prompt):
        part = part.strip()
        if part:
            chunks.append(part)
    return chunks


def matches_any_pattern(text: str, patterns: list[str]) -> bool:
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in patterns)


def has_non_speech_audio_requirement(text: str) -> bool:
    return matches_any_pattern(text, NON_SPEECH_AUDIO_PATTERNS)


def is_speech_only_audio_item(text: str) -> bool:
    return matches_any_pattern(text, SPEECH_AUDIO_PATTERNS) and not has_non_speech_audio_requirement(text)


def scene_chunks(prompt: str) -> list[dict[str, str]]:
    matches = list(re.finditer(r"\bScene\s*(\d+)\s*:", prompt, flags=re.IGNORECASE))
    scenes = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(prompt)
        text = prompt[start:end].strip()
        text = re.sub(r"\s+", " ", text)
        if text:
            scenes.append({"scene": f"Scene {match.group(1)}", "text": text})
    return scenes


def extract_quoted_speech(prompt: str) -> list[dict[str, str]]:
    rows = []
    pattern = re.compile(r"(@Image\d+)\s+says,\s*[\"“](.*?)[\"”]", flags=re.IGNORECASE)
    for match in pattern.finditer(prompt):
        rows.append(
            {
                "speaker": placeholder_from_key(match.group(1)),
                "text": match.group(2).strip(),
                "language": infer_language(match.group(2)),
            }
        )
    if rows:
        return rows

    for text in re.findall(r"[\"“](.*?)[\"”]", prompt):
        text = text.strip()
        if text:
            rows.append({"speaker": "", "text": text, "language": infer_language(text)})
    return rows


def infer_language(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def image_items(sample: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in sample.get("image_items", []):
        rows.append(
            {
                "placeholder": placeholder_from_key(item.get("placeholder", "")),
                "type": item.get("type", "unknown"),
                "path": item.get("path"),
                "relative_path": item.get("relative_path"),
                "description": item.get("description", ""),
            }
        )
    return rows


def find_extra_model_video(model_name: str, model_dir: Path, source_sample_id: str) -> Path | None:
    hyphen_sample_id = source_sample_id.replace("_", "-")
    candidates = [
        model_dir / f"{model_name}_{source_sample_id}.mp4",
        model_dir / f"{source_sample_id}.mp4",
        model_dir / f"{source_sample_id}_{model_name}.mp4",
        model_dir / source_sample_id / f"{source_sample_id}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    recursive_matches = []
    for pattern in (
        f"**/{source_sample_id}*.mp4",
        f"**/*{source_sample_id}*.mp4",
        f"**/*{hyphen_sample_id}*.mp4",
        f"**/{source_sample_id}/*.mp4",
    ):
        recursive_matches.extend(model_dir.glob(pattern))
    recursive_matches = sorted(set(recursive_matches))
    if recursive_matches:
        return recursive_matches[-1]
    return None


def infer_extra_model_videos(sample: dict[str, Any], extra_model_dirs: dict[str, Path]) -> dict[str, str]:
    source_sample_id = sample.get("source_sample_id") or sample.get("sample")
    sample_id = sample.get("id") or sample.get("sample")
    if not source_sample_id and not sample_id:
        return {}
    videos = {}
    for model_name, model_dir in extra_model_dirs.items():
        candidate = None
        for candidate_sample_id in (sample_id, source_sample_id):
            if not candidate_sample_id:
                continue
            candidate = find_extra_model_video(model_name, model_dir, str(candidate_sample_id))
            if candidate is not None:
                break
        if candidate is not None:
            videos[model_name] = str(candidate)
    return videos


def find_video_root_model_video(video_root: Path, model_name: str, sample_id: str) -> Path | None:
    candidates = [
        video_root / model_name / f"{sample_id}__{model_name}.mp4",
        video_root / model_name / f"{sample_id}.mp4",
        video_root / f"{sample_id}__{model_name}.mp4",
        video_root / f"{model_name}_{sample_id}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    model_dir = video_root / model_name
    search_root = model_dir if model_dir.exists() else video_root
    recursive_matches = sorted(search_root.glob(f"**/{sample_id}*.mp4"))
    return recursive_matches[-1] if recursive_matches else None


def infer_video_root_model_videos(video_root: Path | None, sample_id: str, models: set[str] | None) -> dict[str, str]:
    if video_root is None:
        return {}
    if models:
        model_names = sorted(models)
    else:
        model_names = sorted(path.name for path in video_root.iterdir() if path.is_dir())
    videos = {}
    for model_name in model_names:
        candidate = find_video_root_model_video(video_root, model_name, sample_id)
        if candidate is not None:
            videos[model_name] = str(candidate)
    return videos


def load_project_id(credentials_path: Path) -> str:
    data = json.loads(credentials_path.read_text(encoding="utf-8"))
    project_id = data.get("project_id")
    if not project_id:
        raise RuntimeError(f"Missing project_id in credentials file: {credentials_path}")
    return project_id


def build_gemini_client(credentials_path: Path, location: str):
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


def build_visual_checklist(sample: dict[str, Any]) -> list[str]:
    prompt = sample.get("prompt", "")
    checklist = []
    scenes = scene_chunks(prompt)
    if scenes:
        for scene in scenes:
            scene_text = strip_speech_temporal_text(scene["text"])
            if scene_text:
                checklist.append(f"是否完成 {scene['scene']} 的视觉要求：{scene_text}")
    else:
        checklist.append("视频是否完成 prompt 中描述的主要视觉场景、主体动作、空间关系和交互？")

    for sent in sentence_chunks(prompt):
        sent = strip_speech_temporal_text(sent)
        if not sent:
            continue
        lower = sent.lower()
        if any(token in lower for token in ["wide shot", "close-up", "final", "camera", "standing", "wearing", "slides", "places", "turns", "moves"]):
            item = f"是否正确呈现该视觉要求：{sent}"
            if item not in checklist:
                checklist.append(item)
    return checklist


def build_audio_checklist(sample: dict[str, Any]) -> list[str]:
    prompt = sample.get("prompt", "")
    checklist = []
    for sent in sentence_chunks(prompt):
        if has_non_speech_audio_requirement(sent):
            # Only include positive requested audio; do not penalize extra unrequested sounds.
            checklist.append(f"是否生成 prompt 明确要求的音频内容：{sent}")
    return checklist


def strip_speech_temporal_text(text: str) -> str:
    cleaned = re.sub(r"\bAs the sequence continues,\s*(@Image\d+\s+says,\s*)?[\"“].*?[\"”]\.?", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b@Image\d+\s+says,\s*[\"“].*?[\"”]\.?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bwhile speaking\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bwhile saying\s*[\"“].*?[\"”]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" ,;.")


def strip_non_temporal_text(text: str) -> str:
    cleaned = strip_speech_temporal_text(text)
    cleaned = re.sub(
        r"[^.!?。！？]*(?:sound|music|audio|ambien(?:ce|t)|foley|track|noise|accompanies)[^.!?。！？]*[.!?。！？]?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" ,;.")


def temporal_item_is_speech(text: str) -> bool:
    return bool(re.search(r"\b(speaks?|says?|spoken|dialogue|speech|voice|lip sync)\b|说话|说道|对白|台词", text, flags=re.IGNORECASE))


def action_order_pairs(prompt: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    previous_action = ""
    for sent in sentence_chunks(prompt):
        sent = strip_non_temporal_text(sent)
        if not sent:
            continue
        sent_actions = action_phrases_from_sentence(sent)
        match = re.search(r"\b(.+?)\bthen\b(.+)", sent, flags=re.IGNORECASE)
        if match:
            before = match.group(1)
            after = match.group(2)
            if not has_action_verb(before) and previous_action:
                before = previous_action
            pairs.append((before, after))
        match = re.search(r"\b(.+?)\bbefore\b(.+)", sent, flags=re.IGNORECASE)
        if match:
            pairs.append((match.group(1), match.group(2)))
        match = re.search(r"\bafter\b(.+?),\s*(.+)", sent, flags=re.IGNORECASE)
        if match:
            pairs.append((match.group(1), match.group(2)))
        match = re.search(r"\binitially,\s*(.+?),\s*then\s+(.+)", sent, flags=re.IGNORECASE)
        if match:
            pairs.append((match.group(1), match.group(2)))
        if sent_actions:
            previous_action = sent_actions[-1]
    return [(shorten_temporal_clause(a), shorten_temporal_clause(b)) for a, b in pairs]


def has_action_verb(text: str) -> bool:
    return bool(
        re.search(
            r"\b(picks?|puts?|smooths?|adjusts?|walks?|turns?|moves?|slides?|places?|takes?|lifts?|holds?|"
            r"maintains?|transitions?|settles?|looks?|captures?|opens?|closes?|pushes?|pulls?|catches?|throws?)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def action_phrases_from_sentence(text: str) -> list[str]:
    text = strip_non_temporal_text(text)
    text = re.sub(r"^.*?\b(style|shot|scene)\b,\s*", "", text, flags=re.IGNORECASE)
    parts = re.split(r",\s+|\s+and\s+|\s+then\s+", text)
    return [shorten_temporal_clause(part) for part in parts if has_action_verb(part)]


def shorten_temporal_clause(text: str) -> str:
    text = strip_non_temporal_text(text)
    text = re.sub(r"^(and|,|\.|\s)+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,;.")
    return text[:180].strip()


def build_temporal_checklist(sample: dict[str, Any], speech_rows: list[dict[str, str]]) -> list[str]:
    prompt = sample.get("prompt", "")
    scenes = scene_chunks(prompt)
    checklist: list[str] = []
    if len(scenes) >= 2:
        checklist.append("Does the video contain clear, recognizable shot/scene segmentation?")
        checklist.append("Do the required shots/scenes appear in the prompt-specified order?")
        for scene in scenes:
            scene_text = shorten_temporal_clause(scene["text"])
            if scene_text and not temporal_item_is_speech(scene_text):
                checklist.append(f"Is {scene['scene']} present as a distinct shot/scene: {scene_text}?")

    for before, after in action_order_pairs(prompt):
        if before and after and not temporal_item_is_speech(before) and not temporal_item_is_speech(after):
            checklist.append(f"Does {before} occur before {after}?")

    return dedupe(checklist)[:5]


def dedupe(items: list[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def infer_applicable(sample: dict[str, Any], checklist: dict[str, Any], speech_rows: list[dict[str, str]]) -> dict[str, bool]:
    has_visual = bool(checklist["visual_task_following"])
    has_audio = bool(checklist["audio_task_following"] or sample.get("has_sound") or sample.get("has_music"))
    has_temporal = bool(checklist["temporal_order_following"])
    return {
        "Visual Task Following": has_visual,
        "Audio Task Following": has_audio,
        "Speech Content Accuracy": bool(speech_rows),
        "Temporal Order Following": has_temporal,
    }


def build_checklist(sample_path: Path, sample: dict[str, Any]) -> dict[str, Any]:
    speech_rows = extract_quoted_speech(sample.get("prompt", ""))
    checklist = {
        "visual_task_following": build_visual_checklist(sample),
        "audio_task_following": build_audio_checklist(sample),
        "speech_prompt": speech_rows,
        "temporal_order_following": [],
    }
    checklist["temporal_order_following"] = build_temporal_checklist(sample, speech_rows)
    applicable = infer_applicable(sample, checklist, speech_rows)
    return {
        "sample_id": sample.get("id") or sample_path.stem,
        "source_sample_id": sample.get("source_sample_id") or sample.get("sample"),
        "source_board": sample.get("source_board"),
        "prompt": sample.get("prompt", ""),
        "chinese_prompt": sample.get("chinese_prompt", ""),
        "references": image_items(sample),
        "metadata": {
            "human_entity_mode": sample.get("human_entity_mode"),
            "human_entity_count": sample.get("human_entity_count"),
            "source_human_entity_count": sample.get("source_human_entity_count"),
            "has_sound": sample.get("has_sound"),
            "has_music": sample.get("has_music"),
            "has_speech": sample.get("has_speech"),
            "is_multi_shot": sample.get("is_multi_shot"),
        },
        "checklist": checklist,
        "applicable_dimensions": applicable,
        "checklist_source": "rule",
        "checklist_generation": {
            "source": "rule",
            "model": None,
            "raw_response": None,
            "error": None,
        },
        "source_json": str(sample_path.resolve()),
    }


def sample_payload(sample_path: Path, sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample.get("id") or sample_path.stem,
        "source_sample_id": sample.get("source_sample_id") or sample.get("sample"),
        "source_board": sample.get("source_board"),
        "prompt": sample.get("prompt", ""),
        "chinese_prompt": sample.get("chinese_prompt", ""),
        "references": image_items(sample),
        "metadata": {
            "human_entity_mode": sample.get("human_entity_mode"),
            "human_entity_count": sample.get("human_entity_count"),
            "source_human_entity_count": sample.get("source_human_entity_count"),
            "has_sound": sample.get("has_sound"),
            "has_music": sample.get("has_music"),
            "has_speech": sample.get("has_speech"),
            "is_multi_shot": sample.get("is_multi_shot"),
            "appearance_requirement": sample.get("appearance_requirement"),
            "appearance_consistency": sample.get("appearance_consistency"),
            "appearance_consistency_note": sample.get("appearance_consistency_note"),
        },
        "source_json": str(sample_path.resolve()),
    }


def normalize_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return dedupe(out)


def normalize_visual_checklist(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, list[str]] = {}
        for key in [
            "dynamic_interaction",
            "character_performance",
            "static_visual_fidelity",
            "detail_compliance",
        ]:
            items = normalize_text_list(value.get(key))
            if items:
                out[key] = items
        return out
    return normalize_text_list(value)


def normalize_audio_text_list(value: Any) -> list[str]:
    return [item for item in normalize_text_list(value) if not is_speech_only_audio_item(item)]


def normalize_temporal_text_list(value: Any) -> list[str]:
    return [item for item in normalize_text_list(value) if not temporal_item_is_speech(item)]


def normalize_speech_rows(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    rows = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                rows.append({"speaker": "", "text": text, "language": infer_language(text)})
            continue
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        rows.append(
            {
                "speaker": str(item.get("speaker") or "").strip(),
                "text": text,
                "language": str(item.get("language") or infer_language(text)).strip(),
            }
        )
    return rows


def write_checklists_by_sample(root: Path, checklists: list[dict[str, Any]]) -> None:
    for row in checklists:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            continue
        sample_dir = root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        applicable = row.get("applicable_dimensions", {})
        checklist = row.get("checklist", {})
        for dimension, checklist_key in CHECKLIST_KEY_BY_DIMENSION.items():
            payload = {
                "sample_id": sample_id,
                "dimension": dimension,
                "applicable": bool(applicable.get(dimension)),
                "checklist": checklist.get(checklist_key, []),
            }
            write_json(sample_dir / f"{checklist_key}.json", payload)


def checklist_from_gemini_payload(
    sample_path: Path,
    sample: dict[str, Any],
    rule_row: dict[str, Any],
    payload: dict[str, Any],
    source: str,
    model: str,
) -> dict[str, Any]:
    checklist_payload = payload.get("checklist", payload)
    if not isinstance(checklist_payload, dict):
        checklist_payload = {}
    raw_applicable = payload.get("applicable_dimensions")
    explicit_audio_not_applicable = (
        isinstance(raw_applicable, dict)
        and raw_applicable.get("Audio Task Following") is False
    )

    speech_rows = normalize_speech_rows(
        checklist_payload.get("speech_content_accuracy")
        or checklist_payload.get("speech_prompt")
        or checklist_payload.get("speech")
    )
    if not speech_rows:
        speech_rows = rule_row.get("checklist", {}).get("speech_prompt", [])

    checklist = {
        "visual_task_following": normalize_visual_checklist(checklist_payload.get("visual_task_following")),
        "audio_task_following": normalize_audio_text_list(checklist_payload.get("audio_task_following")),
        "speech_prompt": speech_rows,
        "temporal_order_following": normalize_temporal_text_list(checklist_payload.get("temporal_order_following")),
    }

    # Keep the rule fallback for any missing dimension to avoid empty specs when the model under-generates JSON.
    for key, value in rule_row.get("checklist", {}).items():
        if key == "speech_prompt":
            continue
        if key == "audio_task_following" and explicit_audio_not_applicable:
            continue
        if not checklist.get(key):
            checklist[key] = value

    applicable = dict(rule_row.get("applicable_dimensions", {}))
    if isinstance(raw_applicable, dict):
        for dim in ALL_DIMENSIONS:
            if dim in raw_applicable:
                applicable[dim] = bool(raw_applicable[dim])
    applicable = infer_applicable(sample, checklist, speech_rows) | applicable

    return {
        **sample_payload(sample_path, sample),
        "checklist": checklist,
        "applicable_dimensions": applicable,
        "checklist_source": source,
        "checklist_generation": {
            "source": source,
            "model": model,
            "raw_response": payload,
            "error": None,
        },
    }


def build_checklist_prompt(prompt_template: str, sample_data: dict[str, Any], rule_row: dict[str, Any] | None, source: str) -> str:
    # The translated prompt can contain placeholder insertions that are absent
    # from the original generation prompt. Keep the original prompt authoritative
    # when deriving checklist requirements.
    checklist_sample = dict(sample_data)
    checklist_sample.pop("chinese_prompt", None)
    payload = {
        "checklist_source": source,
        "sample": checklist_sample,
        "rule_seed": rule_row if source == "hybrid" else None,
    }
    return prompt_template.replace("{checklist_input_json}", json.dumps(payload, ensure_ascii=False, indent=2))


def gemini_checklist_row(
    client,
    prompt_template: str,
    sample_path: Path,
    sample: dict[str, Any],
    rule_row: dict[str, Any],
    source: str,
    model: str,
    timeout: int,
) -> dict[str, Any]:
    from google.genai import types

    prompt = build_checklist_prompt(prompt_template, sample_payload(sample_path, sample), rule_row, source)
    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    http_options=types.HttpOptions(timeout=int(timeout * 1000)),
                ),
            )
            payload = parse_json_response(response.text or "{}")
            return checklist_from_gemini_payload(sample_path, sample, rule_row, payload, source, model)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(2 * (attempt + 1))

    fallback = dict(rule_row)
    fallback["checklist_source"] = f"{source}_fallback_rule"
    fallback["checklist_generation"] = {
        "source": source,
        "model": model,
        "raw_response": None,
        "error": last_error or "unknown error",
    }
    return fallback


def collect(
    input_dir: Path,
    models: set[str] | None,
    samples: set[str] | None,
    skip_missing_videos: bool,
    extra_model_dirs: dict[str, Path],
    video_root: Path | None,
    checklist_source: str,
    checklist_client: Any,
    checklist_prompt: str | None,
    checklist_model: str,
    checklist_timeout: int,
    silent_checklists: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checklist_rows = []
    task_rows = []
    sample_paths = []
    for sample_path in sorted(input_dir.glob("*.json")):
        if sample_path.name == "manifest.json":
            continue
        if samples and sample_path.stem not in samples:
            try:
                sample_id = read_json(sample_path).get("id") or sample_path.stem
            except Exception:
                sample_id = sample_path.stem
            if sample_id not in samples:
                continue
        sample_paths.append(sample_path)
    total_samples = len(sample_paths)
    for index, sample_path in enumerate(sample_paths, start=1):
        sample = read_json(sample_path)
        sample_id = sample.get("id") or sample_path.stem
        rule_checklist_row = build_checklist(sample_path, sample)
        if checklist_source == "rule":
            if not silent_checklists:
                print(f"[checklist] {index}/{total_samples} {sample_id}: rule", flush=True)
            checklist_row = rule_checklist_row
        else:
            if checklist_client is None or checklist_prompt is None:
                raise RuntimeError(f"Checklist source {checklist_source!r} requires a Gemini client and prompt.")
            if not silent_checklists:
                print(f"[checklist] {index}/{total_samples} {sample_id}: generating with {checklist_source}", flush=True)
            checklist_row = gemini_checklist_row(
                checklist_client,
                checklist_prompt,
                sample_path,
                sample,
                rule_checklist_row,
                checklist_source,
                checklist_model,
                checklist_timeout,
            )
            generation = checklist_row.get("checklist_generation") or {}
            if not silent_checklists:
                if generation.get("error"):
                    print(f"[checklist] {index}/{total_samples} {sample_id}: fallback after error: {generation.get('error')}", flush=True)
                else:
                    print(f"[checklist] {index}/{total_samples} {sample_id}: done", flush=True)
        checklist_rows.append(checklist_row)
        model_videos = dict(sample.get("model_videos") or {})
        for model_name, video_path in infer_extra_model_videos(sample, extra_model_dirs).items():
            model_videos.setdefault(model_name, video_path)
        for model_name, video_path in infer_video_root_model_videos(video_root, sample_id, models).items():
            model_videos[model_name] = video_path
        for model_name, video_path in sorted(model_videos.items()):
            if models and model_name not in models:
                continue
            video = Path(video_path)
            if skip_missing_videos and not video.exists():
                continue
            task_rows.append(
                {
                    "task_id": f"{sample_id}::{model_name}",
                    "sample_id": sample_id,
                    "model_name": model_name,
                    "video_path": str(video),
                    "video_exists": video.exists(),
                    "if_eval_spec": checklist_row,
                }
            )
    return task_rows, checklist_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IF eval tasks and checklists from testvideos metadata.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--tasks-output", default=str(DEFAULT_TASKS))
    parser.add_argument("--checklists-output", default=str(DEFAULT_CHECKLISTS))
    parser.add_argument(
        "--checklists-dir",
        default=str(DEFAULT_CHECKLISTS_DIR),
        help="Directory for per-sample, per-dimension checklist JSON files.",
    )
    parser.add_argument("--models", default="all")
    parser.add_argument("--samples", default="all")
    parser.add_argument(
        "--extra-model-dirs",
        default=None,
        help="Comma-separated model=/path entries for extra generated video roots.",
    )
    parser.add_argument(
        "--video-root",
        default=None,
        help="Optional root with videos stored as <model>/<sample_id>__<model>.mp4. Overrides matching metadata videos.",
    )
    parser.add_argument(
        "--checklist-source",
        choices=["rule", "gemini", "hybrid"],
        default="rule",
        help="rule: deterministic script checklist; gemini: Gemini-generated checklist; hybrid: rule seed revised by Gemini.",
    )
    parser.add_argument("--checklist-prompt", default=str(DEFAULT_CHECKLIST_PROMPT))
    parser.add_argument("--checklist-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--checklist-credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--checklist-location", default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--checklist-timeout", type=int, default=600)
    parser.add_argument("--skip-missing-videos", action="store_true")
    args = parser.parse_args()
    extra_model_dirs = dict(DEFAULT_EXTRA_MODEL_VIDEO_DIRS)
    extra_model_dirs.update(parse_extra_model_dirs(args.extra_model_dirs))
    video_root = Path(args.video_root).expanduser().resolve() if args.video_root else None
    if video_root is not None and not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")

    checklist_client = None
    checklist_prompt = None
    if args.checklist_source != "rule":
        checklist_prompt_path = Path(args.checklist_prompt).expanduser().resolve()
        if not checklist_prompt_path.exists():
            raise FileNotFoundError(f"Checklist prompt file not found: {checklist_prompt_path}")
        checklist_prompt = checklist_prompt_path.read_text(encoding="utf-8")
        checklist_credentials = args.checklist_credentials.expanduser().resolve()
        if not checklist_credentials.exists():
            raise FileNotFoundError(f"Checklist Gemini credentials file not found: {checklist_credentials}")
        checklist_client = build_gemini_client(checklist_credentials, args.checklist_location)

    tasks, checklists = collect(
        Path(args.input_dir).expanduser().resolve(),
        parse_csv_set(args.models),
        parse_csv_set(args.samples),
        args.skip_missing_videos,
        extra_model_dirs,
        video_root,
        args.checklist_source,
        checklist_client,
        checklist_prompt,
        args.checklist_model,
        args.checklist_timeout,
    )
    write_json(Path(args.tasks_output).expanduser().resolve(), tasks)
    write_json(Path(args.checklists_output).expanduser().resolve(), checklists)
    checklists_dir = Path(args.checklists_dir).expanduser().resolve()
    write_checklists_by_sample(checklists_dir, checklists)
    missing = sum(1 for task in tasks if not task["video_exists"])
    print(f"Saved {len(tasks)} tasks to {Path(args.tasks_output).expanduser().resolve()}")
    print(f"Saved {len(checklists)} checklists to {Path(args.checklists_output).expanduser().resolve()}")
    print(f"Saved per-sample dimension checklists to {checklists_dir}")
    print(f"Checklist source: {args.checklist_source}")
    print(f"Samples: {len(checklists)}; models: {len({task['model_name'] for task in tasks})}; missing videos: {missing}")


if __name__ == "__main__":
    main()
