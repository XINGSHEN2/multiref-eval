#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

from build_if_eval_inputs import (
    build_gemini_client,
    collect,
    parse_csv_set,
    parse_extra_model_dirs,
    write_checklists_by_sample,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS = "happyhourse,keling,seedance,veo,vidu,wan"
DEFAULT_BOARDS = "board1,board2,board3,board4"
DEFAULT_CREDENTIALS = Path(os.getenv("FINALEVAL_GEMINI_CREDENTIALS", ""))


def first(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def parse_bool(value: str) -> bool | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "是"}:
        return True
    if text in {"0", "false", "no", "n", "否"}:
        return False
    return None


def parse_int(value: str) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def infer_board(sample_id: str, explicit: str) -> str:
    if explicit:
        return explicit
    if "_" in sample_id:
        return sample_id.split("_", 1)[0]
    if "-" in sample_id:
        return sample_id.split("-", 1)[0]
    return ""


def ref_columns(row: dict[str, str]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    raw = first(row, "reference_images", "references", "image_items")
    if raw:
        try:
            payload = json.loads(raw)
            if isinstance(payload, list):
                for idx, item in enumerate(payload, start=1):
                    if not isinstance(item, dict):
                        continue
                    path = str(item.get("path") or "").strip()
                    if not path:
                        continue
                    refs.append(
                        {
                            "placeholder": str(item.get("placeholder") or f"@Image{idx}").strip(),
                            "path": path,
                            "type": str(item.get("type") or "unknown").strip(),
                            "description": str(item.get("description") or "").strip(),
                        }
                    )
                return refs
        except json.JSONDecodeError:
            for idx, item in enumerate(raw.split(";"), start=1):
                item = item.strip()
                if not item:
                    continue
                if "=" in item:
                    placeholder, path = item.split("=", 1)
                else:
                    placeholder, path = f"@Image{idx}", item
                refs.append(
                    {
                        "placeholder": placeholder.strip(),
                        "path": path.strip(),
                        "type": first(row, f"image{idx}_type", f"reference_image{idx}_type") or "unknown",
                        "description": first(row, f"image{idx}_description", f"reference_image{idx}_description"),
                    }
                )
            return refs

    for idx in range(1, 11):
        path = first(
            row,
            f"image{idx}_path",
            f"ref_image{idx}_path",
            f"reference_image{idx}_path",
            f"reference{idx}_path",
        )
        if not path:
            continue
        refs.append(
            {
                "placeholder": first(row, f"image{idx}_placeholder", f"reference_image{idx}_placeholder") or f"@Image{idx}",
                "path": path,
                "type": first(row, f"image{idx}_type", f"reference_image{idx}_type") or "unknown",
                "description": first(row, f"image{idx}_description", f"reference_image{idx}_description"),
            }
        )
    return refs


def merge_sample(samples: dict[str, dict[str, Any]], row: dict[str, str]) -> None:
    sample_id = first(row, "sample_id", "id")
    model_name = first(row, "model_name", "model")
    video_path = first(row, "video_path", "generated_video_path", "path")
    prompt = first(row, "prompt", "video_prompt")
    if not sample_id or not model_name or not video_path:
        raise ValueError("CSV must include sample_id, model_name, and video_path for every row.")
    if sample_id not in samples:
        board = infer_board(sample_id, first(row, "board", "source_board"))
        samples[sample_id] = {
            "id": sample_id,
            "sample": first(row, "sample", "source_sample_id") or sample_id,
            "source_sample_id": first(row, "source_sample_id", "sample") or sample_id,
            "source_board": board,
            "prompt": prompt,
            "video_prompt": prompt,
            "chinese_prompt": first(row, "chinese_prompt", "prompt_zh"),
            "has_sound": parse_bool(first(row, "has_sound")),
            "has_music": parse_bool(first(row, "has_music")),
            "has_speech": parse_bool(first(row, "has_speech")),
            "is_multi_shot": parse_bool(first(row, "is_multi_shot")),
            "human_entity_count": parse_int(first(row, "human_entity_count")),
            "source_human_entity_count": parse_int(first(row, "source_human_entity_count")),
            "human_entity_mode": first(row, "human_entity_mode"),
            "image_items": ref_columns(row),
            "model_videos": {},
        }
    sample = samples[sample_id]
    if prompt and sample.get("prompt") and prompt != sample["prompt"]:
        raise ValueError(f"Conflicting prompt for sample_id={sample_id}")
    if prompt and not sample.get("prompt"):
        sample["prompt"] = prompt
        sample["video_prompt"] = prompt
    if not sample.get("image_items"):
        sample["image_items"] = ref_columns(row)
    sample["model_videos"][model_name] = video_path


def sample_from_row(row: dict[str, str]) -> dict[str, Any]:
    sample_id = first(row, "sample_id", "id")
    prompt = first(row, "prompt", "video_prompt")
    if not sample_id or not prompt:
        raise ValueError("samples CSV must include sample_id and prompt for every row.")
    board = infer_board(sample_id, first(row, "board", "source_board"))
    return {
        "id": sample_id,
        "sample": first(row, "sample", "source_sample_id") or sample_id,
        "source_sample_id": first(row, "source_sample_id", "sample") or sample_id,
        "source_board": board,
        "prompt": prompt,
        "video_prompt": prompt,
        "chinese_prompt": first(row, "chinese_prompt", "prompt_zh"),
        "has_sound": parse_bool(first(row, "has_sound")),
        "has_music": parse_bool(first(row, "has_music")),
        "has_speech": parse_bool(first(row, "has_speech")),
        "is_multi_shot": parse_bool(first(row, "is_multi_shot")),
        "human_entity_count": parse_int(first(row, "human_entity_count")),
        "source_human_entity_count": parse_int(first(row, "source_human_entity_count")),
        "human_entity_mode": first(row, "human_entity_mode"),
        "image_items": ref_columns(row),
        "model_videos": {},
    }


def merge_video(samples: dict[str, dict[str, Any]], row: dict[str, str]) -> None:
    sample_id = first(row, "sample_id", "id")
    model_name = first(row, "model_name", "model")
    video_path = first(row, "video_path", "generated_video_path", "path")
    if not sample_id or not model_name or not video_path:
        raise ValueError("videos CSV must include sample_id, model_name, and video_path for every row.")
    if sample_id not in samples:
        raise ValueError(f"videos CSV references sample_id={sample_id}, but it is missing from samples CSV.")
    samples[sample_id]["model_videos"][model_name] = video_path


def load_samples_from_two_csvs(samples_csv: Path, videos_csv: Path) -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    with samples_csv.expanduser().open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clean_row = {key: value for key, value in row.items() if key is not None}
            sample = sample_from_row(clean_row)
            sample_id = str(sample["id"])
            if sample_id in samples:
                raise ValueError(f"Duplicate sample_id in samples CSV: {sample_id}")
            samples[sample_id] = sample
    with videos_csv.expanduser().open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            merge_video(samples, {key: value for key, value in row.items() if key is not None})
    return samples


def load_samples_from_long_csv(csv_path: Path) -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    with csv_path.expanduser().open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            merge_sample(samples, {key: value for key, value in row.items() if key is not None})
    return samples


def write_metadata(samples: dict[str, dict[str, Any]], root: Path, boards: set[str] | None) -> dict[str, Path]:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    board_dirs: dict[str, Path] = {}
    for sample_id, sample in sorted(samples.items()):
        board = str(sample.get("source_board") or infer_board(sample_id, "")).strip()
        if boards and board not in boards:
            continue
        if not board:
            raise ValueError(f"Cannot infer board for sample_id={sample_id}; add a board column.")
        board_dir = root / board
        board_dir.mkdir(parents=True, exist_ok=True)
        board_dirs[board] = board_dir
        (board_dir / f"{sample_id}.json").write_text(
            json.dumps(sample, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return board_dirs


def copy_speech_inputs(prefix: Path) -> None:
    shutil.copy2(prefix.with_name(prefix.name + "_tasks.json"), prefix.with_name(prefix.name + "_speech_tasks.json"))
    shutil.copy2(prefix.with_name(prefix.name + "_checklists.json"), prefix.with_name(prefix.name + "_speech_checklists.json"))
    src_dir = prefix.with_name(prefix.name + "_checklists_by_sample")
    dst_dir = prefix.with_name(prefix.name + "_speech_checklists_by_sample")
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


def merge_visual_and_nonvisual_checklists(
    visual_checklists: list[dict[str, Any]],
    nonvisual_checklists: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    nonvisual_by_sample = {str(row.get("sample_id")): row for row in nonvisual_checklists}
    merged = []
    for visual_row in visual_checklists:
        sample_id = str(visual_row.get("sample_id"))
        nonvisual_row = nonvisual_by_sample.get(sample_id, {})
        row = dict(visual_row)
        visual_checklist = dict(visual_row.get("checklist") or {})
        nonvisual_checklist = dict(nonvisual_row.get("checklist") or {})
        row["checklist"] = {
            "visual_task_following": visual_checklist.get("visual_task_following", []),
            "audio_task_following": nonvisual_checklist.get("audio_task_following", []),
            "speech_prompt": nonvisual_checklist.get("speech_prompt", []),
            "temporal_order_following": nonvisual_checklist.get("temporal_order_following", []),
        }
        visual_applicable = dict(visual_row.get("applicable_dimensions") or {})
        nonvisual_applicable = dict(nonvisual_row.get("applicable_dimensions") or {})
        row["applicable_dimensions"] = {
            "Visual Task Following": bool(visual_applicable.get("Visual Task Following")),
            "Audio Task Following": bool(nonvisual_applicable.get("Audio Task Following")),
            "Speech Content Accuracy": bool(nonvisual_applicable.get("Speech Content Accuracy")),
            "Temporal Order Following": bool(nonvisual_applicable.get("Temporal Order Following")),
        }
        row["checklist_source"] = {
            "visual": visual_row.get("checklist_source"),
            "nonvisual": nonvisual_row.get("checklist_source"),
        }
        row["checklist_generation"] = {
            "visual": visual_row.get("checklist_generation"),
            "nonvisual": nonvisual_row.get("checklist_generation"),
        }
        merged.append(row)
    return merged


def apply_checklists_to_tasks(tasks: list[dict[str, Any]], checklists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sample = {str(row.get("sample_id")): row for row in checklists}
    out = []
    for task in tasks:
        row = dict(task)
        sample_id = str(row.get("sample_id"))
        if sample_id in by_sample:
            row["if_eval_spec"] = by_sample[sample_id]
        out.append(row)
    return out


def load_reused_checklists(
    root: Path,
    board: str,
    samples: set[str],
    skip_missing: bool = False,
) -> tuple[list[dict[str, Any]], set[str]]:
    path = root / f"if_{board}_checklists.json"
    if not path.exists():
        raise FileNotFoundError(f"reused checklist file not found: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected checklist JSON list: {path}")
    out = []
    missing = set(samples)
    for row in rows:
        if not isinstance(row, dict):
            continue
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in samples:
            continue
        copied = dict(row)
        copied["checklist_source"] = {
            "visual": f"reused:{path}",
            "nonvisual": f"reused:{path}",
        }
        copied["checklist_generation"] = {
            "visual": {"source": "reused_existing_checklist", "path": str(path)},
            "nonvisual": {"source": "reused_existing_checklist", "path": str(path)},
        }
        out.append(copied)
        missing.discard(sample_id)
    if missing:
        preview = ", ".join(sorted(missing)[:20])
        suffix = "" if len(missing) <= 20 else f", ... ({len(missing)} total)"
        message = f"reused checklist file is missing samples for {board}: {preview}{suffix}"
        if not skip_missing:
            raise ValueError(message)
        print(f"[skip] {message}", flush=True)
    return out, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare IF inputs from sample/video CSV files.")
    parser.add_argument("--csv", type=Path, help="Legacy one-table CSV with sample, prompt, model, and video columns.")
    parser.add_argument("--samples-csv", type=Path, help="Sample-level CSV with prompt/reference/metadata columns.")
    parser.add_argument("--videos-csv", type=Path, help="Video-level CSV with sample_id/model_name/video_path columns.")
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "inputs")
    parser.add_argument("--metadata-root", type=Path, default=None)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--boards", default=DEFAULT_BOARDS)
    parser.add_argument("--checklist-source", choices=["rule", "gemini", "hybrid"], default="gemini")
    parser.add_argument("--checklist-model", default="gemini-3.1-pro-preview")
    parser.add_argument("--checklist-prompt", type=Path, default=SCRIPT_DIR / "if_judge_prompts_v3/visual_checklist_generation.md")
    parser.add_argument("--nonvisual-checklist-source", choices=["rule", "gemini", "hybrid"], default="gemini")
    parser.add_argument(
        "--nonvisual-checklist-prompt",
        type=Path,
        default=SCRIPT_DIR / "if_judge_prompts_v3/if_checklist_generation_prompt_v2.md",
    )
    parser.add_argument("--checklist-credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--checklist-location", default="global")
    parser.add_argument("--checklist-timeout", type=int, default=600)
    parser.add_argument("--extra-model-dirs", default="")
    parser.add_argument("--skip-missing-videos", action="store_true")
    parser.add_argument(
        "--reuse-checklists-root",
        type=Path,
        default=None,
        help="Reuse existing if_board{1,2,3}_checklists.json files from this directory instead of generating checklists.",
    )
    parser.add_argument(
        "--skip-missing-checklists",
        action="store_true",
        help="When reusing checklists, skip CSV samples that are not present in the reused checklist files instead of generating or failing.",
    )
    args = parser.parse_args()

    if args.samples_csv or args.videos_csv:
        if not args.samples_csv or not args.videos_csv:
            raise ValueError("--samples-csv and --videos-csv must be provided together.")
        samples = load_samples_from_two_csvs(args.samples_csv, args.videos_csv)
        csv_stem = f"{args.samples_csv.stem}_{args.videos_csv.stem}"
    elif args.csv:
        samples = load_samples_from_long_csv(args.csv)
        csv_stem = args.csv.stem
    else:
        raise ValueError("Provide either --samples-csv/--videos-csv or legacy --csv.")

    output_root = args.output_root.expanduser().resolve()
    metadata_root = (args.metadata_root or output_root / f"{csv_stem}_metadata").expanduser().resolve()
    boards = parse_csv_set(args.boards)
    board_dirs = write_metadata(samples, metadata_root, boards)

    reuse_checklists_root = args.reuse_checklists_root.expanduser().resolve() if args.reuse_checklists_root else None
    if reuse_checklists_root is not None:
        print(f"[reuse] checklists root: {reuse_checklists_root}", flush=True)

    checklist_client = None
    checklist_prompt = None
    if reuse_checklists_root is None and args.checklist_source != "rule":
        checklist_prompt = args.checklist_prompt.expanduser().resolve().read_text(encoding="utf-8")
        checklist_client = build_gemini_client(args.checklist_credentials.expanduser().resolve(), args.checklist_location)
    nonvisual_checklist_client = None
    nonvisual_checklist_prompt = None
    if reuse_checklists_root is None and args.nonvisual_checklist_source != "rule":
        nonvisual_checklist_prompt = args.nonvisual_checklist_prompt.expanduser().resolve().read_text(encoding="utf-8")
        nonvisual_checklist_client = checklist_client or build_gemini_client(
            args.checklist_credentials.expanduser().resolve(),
            args.checklist_location,
        )

    models = parse_csv_set(args.models)
    extra_model_dirs = parse_extra_model_dirs(args.extra_model_dirs)
    for board in sorted(boards or board_dirs):
        input_dir = board_dirs.get(board)
        if input_dir is None:
            continue
        prefix = output_root / f"if_{board}"
        checklist_source = "rule" if reuse_checklists_root is not None else args.checklist_source
        tasks, generated_checklists = collect(
            input_dir,
            models,
            None,
            args.skip_missing_videos,
            extra_model_dirs,
            None,
            checklist_source,
            checklist_client,
            checklist_prompt,
            args.checklist_model,
            args.checklist_timeout,
            silent_checklists=reuse_checklists_root is not None,
        )
        if reuse_checklists_root is not None:
            samples_for_board = {str(task.get("sample_id")) for task in tasks}
            checklists, missing_checklists = load_reused_checklists(
                reuse_checklists_root,
                board,
                samples_for_board,
                skip_missing=args.skip_missing_checklists,
            )
            if missing_checklists:
                tasks = [task for task in tasks if str(task.get("sample_id")) not in missing_checklists]
        else:
            _, nonvisual_checklists = collect(
                input_dir,
                models,
                None,
                args.skip_missing_videos,
                extra_model_dirs,
                None,
                args.nonvisual_checklist_source,
                nonvisual_checklist_client,
                nonvisual_checklist_prompt,
                args.checklist_model,
                args.checklist_timeout,
            )
            checklists = merge_visual_and_nonvisual_checklists(generated_checklists, nonvisual_checklists)
        tasks = apply_checklists_to_tasks(tasks, checklists)
        write_json(prefix.with_name(prefix.name + "_tasks.json"), tasks)
        write_json(prefix.with_name(prefix.name + "_checklists.json"), checklists)
        write_checklists_by_sample(prefix.with_name(prefix.name + "_checklists_by_sample"), checklists)
        copy_speech_inputs(prefix)
        print(f"[done] {board}: {len(checklists)} samples, {len(tasks)} tasks -> {prefix}_tasks.json")

    print(f"[done] metadata root: {metadata_root}")
    print(f"[done] inputs root: {output_root}")


if __name__ == "__main__":
    main()
