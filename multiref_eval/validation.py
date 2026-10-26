import json
import re
from pathlib import Path
from finaleval.common.schema import read_input_csv

def validate_manifest(path, metrics):
    samples = read_input_csv(Path(path))
    errors = []
    seen = set()
    if not samples:
        errors.append("Input manifest is empty")
    for row in samples:
        key = (row.sample_id, row.model_name)
        label = f"{row.sample_id}/{row.model_name}"
        if key in seen:
            errors.append(f"{label}: duplicate sample/model pair")
        seen.add(key)
        if not all((row.sample_id, row.model_name, row.prompt.strip())):
            errors.append(f"{label}: empty ID, model or prompt")
        paths = [("video_path", row.video_path)]
        paths += [("reference_image_paths", p) for p in json.loads(row.reference_image_paths or "[]")]
        paths += [(k, getattr(row, k)) for k in ("reference_video_path", "reference_json_path", "reference_audio_path") if getattr(row, k)]
        for field, value in paths:
            if not value or not Path(value).is_file():
                errors.append(f"{label}: missing {field}: {value}")
        if "voice_timbre_similarity" in metrics and not row.reference_audio_path:
            errors.append(f"{label}: voice timbre requires reference audio")
    return samples, errors

def audit_dataset(root):
    root = Path(root).resolve()
    records = [json.loads(line) for line in (root / "data/samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    missing, audio, seen, duplicate = [], [], set(), []
    for row in records:
        sid = row["sample_id"]
        if sid in seen:
            duplicate.append(sid)
        seen.add(sid)
        if re.search(r"@audio", row.get("prompt", ""), re.I):
            audio.append(sid)
        for field in ("image_1", "image_2", "image_3", "reference_video_1", "reference_video_2", "reference_video_3"):
            value = row.get(field)
            if value and not (root / value).is_file():
                missing.append({"sample_id": sid, "field": field, "path": value})
    return {"samples": len(records), "duplicate_ids": duplicate, "missing_visual_files": missing,
            "audio_reference_sample_ids": audio,
            "note": "Audio mentions flag release limitations; this is not a semantic completeness check. No questions or assets are modified."}
