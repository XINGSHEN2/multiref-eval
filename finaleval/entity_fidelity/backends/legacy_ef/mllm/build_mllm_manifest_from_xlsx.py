#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

EVALTEST_ROOT = Path(__file__).resolve().parents[2]
BOARD_JSON_DIRS = {
    "board1": EVALTEST_ROOT / "board1_106_testvideos",
    "board2": EVALTEST_ROOT / "board2_100_testvideos",
    "board3": EVALTEST_ROOT / "board3_100_testvideos",
}


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def normalize_board(value: Any) -> str:
    text = str(value).strip().lower()
    match = re.search(r"board\s*([1234])", text)
    if match:
        return f"board{match.group(1)}"
    if text in {"1", "2", "3", "4"}:
        return f"board{text}"
    raise ValueError(f"Cannot parse board from value: {value!r}")


def normalize_seq(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and not math.isnan(value):
        return int(value)
    text = str(value).strip()
    match = re.search(r"(?:board[1234][_-])?(\d{1,3})$", text, re.IGNORECASE)
    if not match:
        match = re.search(r"MultiOmni-board[123]-(\d{1,3})", text, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse seq from value: {value!r}")
    return int(match.group(1))


def infer_board_seq(row: pd.Series) -> tuple[str, int]:
    board_col = next((c for c in ["board", "source_board"] if c in row and not is_empty(row[c])), None)
    seq_col = next((c for c in ["seq", "index", "sample_id", "id", "sample"] if c in row and not is_empty(row[c])), None)
    if board_col and seq_col:
        return normalize_board(row[board_col]), normalize_seq(row[seq_col])

    joined = " ".join(str(row[c]) for c in row.index if not is_empty(row[c]))
    board_match = re.search(r"board\s*([1234])", joined, re.IGNORECASE)
    seq_match = re.search(r"MultiOmni-board[123]-(\d{1,3})", joined, re.IGNORECASE)
    if not seq_match:
        seq_match = re.search(r"board[123][_-](\d{1,3})", joined, re.IGNORECASE)
    if board_match and seq_match:
        return f"board{board_match.group(1)}", int(seq_match.group(1))
    raise ValueError("xlsx row must provide board+seq, or an index/sample_id like MultiOmni-board1-001")


def infer_sample_id(row: pd.Series, board: str, seq: int) -> str:
    value = row_first(row, "sample_id", "sample", "id")
    if not is_empty(value):
        return str(value).strip()
    return f"{board}_{seq:03d}"


def read_sample_json(board: str, seq: int) -> tuple[Path, dict[str, Any]]:
    json_path = BOARD_JSON_DIRS[board] / f"{board}_{seq:03d}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing sample json: {json_path}")
    return json_path, json.loads(json_path.read_text(encoding="utf-8"))


def build_references(sample: dict[str, Any]) -> list[dict[str, Any]]:
    refs = []
    for item in sample.get("image_items") or []:
        ref = dict(item)
        if "placeholder" in ref:
            ref["placeholder"] = str(ref["placeholder"]).replace("@image", "@Image")
        refs.append(ref)
    return refs


def build_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "has_sound",
        "has_music",
        "has_speech",
        "duration",
        "duration_sec",
        "is_multi_shot",
        "has_clothing_or_makeup_change",
        "appearance_requirement",
        "appearance_consistency",
        "appearance_consistency_note",
        "human_entity_mode",
        "human_entity_count",
        "source_human_entity_mode",
        "source_human_entity_count",
        "task",
    ]
    return {key: sample.get(key) for key in keys if key in sample}


def row_first(row: pd.Series, *names: str) -> Any:
    for name in names:
        if name in row and not is_empty(row[name]):
            return row[name]
    return None


def parse_bool(value: Any) -> bool | None:
    if is_empty(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def parse_jsonish(value: Any) -> Any:
    if is_empty(value):
        return None
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None


def path_description(path: str) -> str:
    stem = Path(path).stem
    return stem.replace("_combine", "").replace("_", " ")


def infer_ref_type(path: str, position: int, media_type: str) -> str:
    lowered = path.lower()
    if media_type == "video":
        return "human"
    if "scene" in lowered or "background" in lowered:
        return "background"
    if "object" in lowered or "things" in lowered:
        return "object"
    if "human" in lowered or "person" in lowered or "woman" in lowered or "man" in lowered:
        return "human"
    if position == 1:
        return "human"
    return "background" if position >= 3 else "object"


def build_csv_references(row: pd.Series) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for column, media_type in [("reference_videos", "video"), ("reference_images", "image")]:
        value = parse_jsonish(row_first(row, column))
        if not isinstance(value, list):
            continue
        for path in value:
            if is_empty(path):
                continue
            ref_path = str(path).strip()
            index = len(refs) + 1
            placeholder = f"@{'Video' if media_type == 'video' else 'Image'}{sum(1 for ref in refs if ref.get('media_type') == media_type) + 1}"
            refs.append(
                {
                    "placeholder": placeholder,
                    "path": ref_path,
                    "type": infer_ref_type(ref_path, index, media_type),
                    "description": path_description(ref_path),
                    "media_type": media_type,
                }
            )
    return refs


def csv_metadata(row: pd.Series) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in [
        "has_sound",
        "has_music",
        "has_speech",
        "duration",
        "duration_sec",
        "is_multi_shot",
        "source_board",
        "original_index",
        "running_params",
        "asset_metadata",
    ]:
        value = row_first(row, key)
        if is_empty(value):
            continue
        if key in {"has_sound", "has_music", "has_speech", "is_multi_shot"}:
            metadata[key] = parse_bool(value)
        elif key in {"running_params", "asset_metadata"}:
            metadata[key] = parse_jsonish(value) or str(value)
        else:
            metadata[key] = value
    return metadata


def build_csv_row(row: pd.Series, row_number: int) -> dict[str, Any]:
    model_value = row_first(row, "model", "model_name")
    video_value = row_first(row, "video_path", "generated_video_path", "path")
    if is_empty(model_value):
        raise ValueError(f"Row {row_number}: missing required column 'model' or 'model_name'")
    if is_empty(video_value):
        raise ValueError(f"Row {row_number}: missing required column 'video_path'")

    board, seq = infer_board_seq(row)
    sample_id = infer_sample_id(row, board, seq)
    model_name = str(model_value).strip()
    video_path = str(video_value).strip()
    references = build_csv_references(row)
    source_json = row_first(row, "source_json", "asset_metadata")
    source_json = "" if is_empty(source_json) else str(source_json).strip()

    spec = {
        "sample_id": sample_id,
        "source_sample_id": row_first(row, "source_sample_id", "original_index", "index") or sample_id,
        "source_board": row_first(row, "source_board", "board") or board,
        "prompt": row_first(row, "prompt", "video_prompt") or "",
        "chinese_prompt": row_first(row, "chinese_prompt", "prompt_zh") or "",
        "references": references,
        "image_items": references,
        "metadata": csv_metadata(row),
        "source_json": source_json,
        "checklist": {},
    }

    return {
        "task_id": f"{sample_id}::{model_name}",
        "sample_id": sample_id,
        "model_name": model_name,
        "video_path": video_path,
        "video_exists": Path(video_path).exists(),
        "source_json": source_json,
        "if_eval_spec": spec,
    }


def build_row(row: pd.Series, row_number: int) -> dict[str, Any]:
    model_value = row_first(row, "model", "model_name")
    video_value = row_first(row, "video_path", "generated_video_path", "path")
    if is_empty(model_value):
        raise ValueError(f"Row {row_number}: missing required column 'model' or 'model_name'")
    if is_empty(video_value):
        raise ValueError(f"Row {row_number}: missing required column 'video_path'")

    board, seq = infer_board_seq(row)
    sample_id = f"{board}_{seq:03d}"
    try:
        source_json, sample = read_sample_json(board, seq)
    except (FileNotFoundError, KeyError):
        return build_csv_row(row, row_number)
    model_name = str(model_value).strip()
    video_path = str(video_value).strip()
    references = build_references(sample)

    spec = {
        "sample_id": sample_id,
        "source_sample_id": sample.get("source_sample_id") or f"MultiOmni-{board}-{seq:03d}",
        "source_board": board,
        "prompt": sample.get("prompt") or sample.get("video_prompt") or "",
        "chinese_prompt": sample.get("chinese_prompt") or "",
        "references": references,
        "image_items": references,
        "metadata": build_metadata(sample),
        "source_json": str(source_json),
        "checklist": sample.get("checklist") or {},
    }

    return {
        "task_id": f"{sample_id}::{model_name}",
        "sample_id": sample_id,
        "model_name": model_name,
        "video_path": video_path,
        "video_exists": Path(video_path).exists(),
        "source_json": str(source_json),
        "if_eval_spec": spec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLLM metric manifest from an xlsx/csv video list.")
    parser.add_argument("--xlsx", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--board", default="all", help="all, board1, board2, board3, or comma-separated boards")
    parser.add_argument("--sheet", default=0, help="Excel sheet name or 0-based sheet index; default 0")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--allow-missing-videos", action="store_true")
    args = parser.parse_args()

    sheet: int | str = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)
    if bool(args.xlsx) == bool(args.csv):
        raise ValueError("Provide exactly one input: --xlsx or --csv")
    if args.csv:
        df = pd.read_csv(args.csv.expanduser().resolve())
    else:
        df = pd.read_excel(args.xlsx.expanduser().resolve(), sheet_name=sheet)
    df.columns = [str(col).strip() for col in df.columns]
    if args.limit_rows:
        df = df.head(args.limit_rows)

    board_filter = None
    if args.board.strip().lower() != "all":
        board_filter = {normalize_board(item) for item in args.board.split(",") if item.strip()}

    manifest = []
    skipped = 0
    for idx, row in df.iterrows():
        board, _ = infer_board_seq(row)
        if board_filter and board not in board_filter:
            skipped += 1
            continue
        item = build_row(row, idx + 2)
        if not args.allow_missing_videos and not item["video_exists"]:
            raise FileNotFoundError(f"Video does not exist for {item['task_id']}: {item['video_path']}")
        manifest.append(item)

    if not manifest:
        raise ValueError("No manifest rows were generated. Check --board and xlsx content.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {len(manifest)} rows to {args.output}; skipped {skipped} rows")


if __name__ == "__main__":
    main()
