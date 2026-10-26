#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

EVALTEST_ROOT = Path(__file__).resolve().parents[1]
FINAL_EF_ROOT = Path(__file__).resolve().parents[1]
EVAL_PY_ROOT = FINAL_EF_ROOT / "eval_py"
TOOLS_ROOT = FINAL_EF_ROOT / "tools"
SUMMARY_PY_ROOT = FINAL_EF_ROOT / "summary_py"
EF_ROOT = EVAL_PY_ROOT
sys.path.insert(0, str(EVAL_PY_ROOT))
sys.path.insert(1, str(TOOLS_ROOT))
sys.path.insert(2, str(SUMMARY_PY_ROOT))

from add_long_term_consistency import process_file  # noqa: E402
from body_branch import BodyBranch  # noqa: E402
from condition_pack import load_condition_references  # noqa: E402
from export_ef_summary_csv import collect_rows, write_csv  # noqa: E402
from generate_board123_rankings import write_rankings  # noqa: E402
from face_branch import FaceBranch  # noqa: E402
from make_frame_box_contact_sheet import make_branch_sheets  # noqa: E402
from make_renumbered_face_eval_samples import build_sample, load_summary  # noqa: E402
from build_video_reference_ef_samples import build_samples_from_csv  # noqa: E402
from multi_entity_fidelity import MultiEntityFidelityEvaluator  # noqa: E402
from object_branch import ObjectBranch  # noqa: E402
from reference_pack import load_human_reference_pack, load_reference_pack  # noqa: E402
from run_entity_fidelity import (  # noqa: E402
    build_cartoon_face_detector,
    build_detector,
    build_results_summary,
    build_text_detector,
    combine_video_branches,
    evaluate_video,
    has_cartoon_human,
    reference_body_image,
    reference_top_k,
)
from scene_branch import SceneBranch  # noqa: E402

DEFAULT_WORKSPACE = Path(__file__).resolve().parents[3]


def str_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def norm_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_board(value: object, fallback: str = "") -> str:
    text = norm_text(value).lower()
    if text in {"1", "2", "3"}:
        text = f"board{text}"
    if text in {"board1", "board2", "board3"}:
        return text
    if fallback:
        return fallback
    raise ValueError(f"Unsupported board value: {value!r}")


def normalize_seq(value: object) -> str:
    text = norm_text(value)
    if not text:
        raise ValueError("seq is empty")
    if text.lower().endswith(".mp4"):
        text = text[:-4]
    if "-" in text:
        text = text.rsplit("-", 1)[-1]
    if "_" in text:
        text = text.rsplit("_", 1)[-1]
    return str(int(text)).zfill(3) if text.isdigit() else text.zfill(3)


def infer_board_seq(row: pd.Series, board_filter: str = "") -> tuple[str, str]:
    if "source_board" in row.index and norm_text(row.get("source_board")):
        board = normalize_board(row.get("source_board"), board_filter)
    elif "board" in row.index and norm_text(row.get("board")):
        board = normalize_board(row.get("board"), board_filter)
    elif board_filter:
        board = board_filter
    else:
        index_text = norm_text(row.get("index") or row.get("sample") or row.get("sample_id"))
        for candidate in ("board1", "board2", "board3"):
            if candidate in index_text:
                board = candidate
                break
        else:
            raise ValueError("board column is missing and board cannot be inferred")

    if "seq" in row.index and norm_text(row.get("seq")):
        seq = normalize_seq(row.get("seq"))
    else:
        index_text = norm_text(row.get("index") or row.get("sample") or row.get("sample_id"))
        seq = normalize_seq(index_text)
    return board, seq


def load_jobs(table: Path, board_filter: str = "") -> list[dict[str, str]]:
    suffix = table.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(table)
    elif suffix in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(table)
    else:
        raise ValueError(f"input table must be .csv or .xlsx, got: {table}")
    lower_cols = {str(col).strip().lower(): col for col in df.columns}
    if "video_path" not in lower_cols:
        raise ValueError("input table must contain a video_path column")
    model_col = lower_cols.get("model") or lower_cols.get("model_name")
    if not model_col:
        raise ValueError("input table must contain a model or model_name column")

    jobs: list[dict[str, str]] = []
    for row_index, row in df.iterrows():
        board, seq = infer_board_seq(row, board_filter)
        if board_filter and board != board_filter:
            continue
        model = norm_text(row[model_col])
        video_path = Path(norm_text(row[lower_cols["video_path"]])).expanduser()
        if not model:
            raise ValueError(f"row {row_index + 2}: model is empty")
        if not video_path.is_file():
            raise FileNotFoundError(f"row {row_index + 2}: video_path not found: {video_path}")
        jobs.append(
            {
                "board": board,
                "seq": seq,
                "tag": f"{board}_{seq}",
                "sample_id": f"MultiOmni-{board}-{seq}",
                "model": model,
                "video_path": str(video_path),
            }
        )
    if not jobs:
        print("No jobs selected from input table for this board; skipping.", flush=True)
    return jobs


def csv_has_reference_videos(table: Path) -> bool:
    if table.suffix.lower() != ".csv":
        return False
    try:
        frame = pd.read_csv(table, usecols=lambda col: str(col).strip().lower() == "reference_videos")
    except Exception:
        return False
    if "reference_videos" not in frame.columns:
        return False
    return any(str(value).strip() not in {"", "[]", "nan", "None", "null"} for value in frame["reference_videos"])


def ensure_sample_json(job: dict[str, str], sample_root: Path, source_xlsx_root: Path) -> Path:
    board = job["board"]
    seq = job["seq"]
    output = sample_root / f"{board}_{seq}.json"
    if output.exists():
        return output

    workbook = source_xlsx_root / f"MultiOmni_{board}_final_success_renumbered.xlsx"
    if not workbook.exists():
        raise FileNotFoundError(f"Missing source workbook for sample metadata: {workbook}")
    frame = pd.read_excel(workbook)
    sample_id = f"MultiOmni-{board}-{seq}"
    matches = frame[frame["index"].astype(str).eq(sample_id)]
    if matches.empty:
        raise ValueError(f"Cannot find {sample_id} in {workbook}")
    summary_rows = load_summary(board)
    data = build_sample(board, seq, matches.iloc[0], summary_rows.get(sample_id, {}))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def split_shard(items: list[dict[str, str]], shard_index: int, shard_count: int) -> list[dict[str, str]]:
    if shard_count <= 1:
        return items
    return [item for index, item in enumerate(items) if index % shard_count == shard_index]


def collect_person_detection_texts(sample_paths: list[Path]) -> list[str]:
    texts: list[str] = []
    for path in sample_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("person_detection_prompts") or []:
            text = str(item).strip()
            if text and text not in texts:
                texts.append(text)
    return texts or ["person"]


def make_eval_args(args: argparse.Namespace, output_json: Path, frame_box_dir: Path | None, person_detection_texts: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        output_json=str(output_json),
        num_frames=args.num_frames,
        face_weight=args.face_weight,
        reference_top_k=args.reference_top_k,
        min_frame_ef=args.min_frame_ef,
        cartoon_min_body_ef=args.cartoon_min_body_ef,
        device=args.device,
        face_model_root=args.face_model_root,
        curricular_checkpoint=args.curricular_checkpoint,
        face_recognition_model=args.face_recognition_model,
        face_recognition_name=args.face_recognition_name,
        dinov2_model=args.dinov2_model,
        siglip_model=args.siglip_model,
        reference_human_root=args.reference_human_root,
        reference_body_mask_root=args.reference_body_mask_root,
        disable_reference_body_mask=args.disable_reference_body_mask,
        disable_body_ef=False,
        disable_cartoon_head_ef=args.disable_cartoon_head_ef,
        save_frame_boxes=args.save_frame_boxes,
        frame_box_dir=str(frame_box_dir) if frame_box_dir else None,
        disable_frame_box_export=not args.save_frame_boxes,
        draw_dropped_boxes=False,
        detector=args.detector,
        yoloworld_config=args.yoloworld_config,
        yoloworld_checkpoint=args.yoloworld_checkpoint,
        yoloworld_repo=args.yoloworld_repo,
        detector_device=args.detector_device,
        person_score_threshold=args.person_score_threshold,
        person_detection_texts=person_detection_texts,
        object_score_threshold=args.object_score_threshold,
        disable_object_ef=args.disable_object_ef,
        disable_scene_ef=args.disable_scene_ef,
        cartoon_face_detector=args.cartoon_face_detector,
        cartoon_face_score_threshold=args.cartoon_face_score_threshold,
        hysts_yolov3_config=args.hysts_yolov3_config,
        hysts_yolov3_checkpoint=args.hysts_yolov3_checkpoint,
        yolov8_anime_face_checkpoint=args.yolov8_anime_face_checkpoint,
        yolov8_device=args.yolov8_device,
    )


def condition_ref_payload(refs) -> list[dict[str, Any]]:
    return [
        {"key": ref.key, "path": str(ref.path), "description": ref.description, "style": ref.style}
        for ref in refs
    ]


def result_payload(sample_json: Path, video_path: Path, eval_args: SimpleNamespace, face_branch, body_branch, detector, reference_cache, cartoon_face_detector, object_refs, scene_refs, object_branch, scene_branch, object_frame_box_dir: Path | None) -> dict[str, Any]:
    human_pack = load_human_reference_pack(sample_json)
    pack = load_reference_pack(sample_json)
    if len(human_pack.entities) > 1:
        multi_evaluator = MultiEntityFidelityEvaluator(
            face_branch,
            body_branch,
            detector,
            face_weight=eval_args.face_weight,
            reference_top_k=reference_top_k(eval_args),
            reference_body_image_fn=lambda ref: reference_body_image(ref, eval_args),
            cartoon_face_detector=cartoon_face_detector,
        )
        human_video = multi_evaluator.evaluate_video(video_path, eval_args, human_pack)
        scene_result = scene_branch.evaluate_video(video_path, eval_args.num_frames) if scene_branch else None
        object_result = object_branch.evaluate_video(video_path, eval_args.num_frames, object_frame_box_dir) if object_branch else None
        videos = [combine_video_branches(human_video, scene_result, object_result)]
        return {
            "sample_id": human_pack.sample_id,
            "mode": "multi_human",
            "condition_types": ["human"] + (["object"] if object_refs else []) + (["scene"] if scene_refs else []),
            "human_entities": [
                {"entity_id": entity.entity_id, "person_id": entity.person_id, "reference_images": [str(ref.path) for ref in entity.references]}
                for entity in human_pack.entities
            ],
            "object_references": condition_ref_payload(object_refs),
            "scene_references": condition_ref_payload(scene_refs),
            "videos": videos,
        }

    human_video = evaluate_video(video_path, eval_args, pack, face_branch, body_branch, detector, reference_cache, cartoon_face_detector=cartoon_face_detector)
    scene_result = scene_branch.evaluate_video(video_path, eval_args.num_frames) if scene_branch else None
    object_result = object_branch.evaluate_video(video_path, eval_args.num_frames, object_frame_box_dir) if object_branch else None
    videos = [combine_video_branches(human_video, scene_result, object_result)]
    return {
        "sample_id": pack.sample_id,
        "mode": "single_human",
        "condition_types": ["human"] + (["object"] if object_refs else []) + (["scene"] if scene_refs else []),
        "reference_roles": {ref.key: ref.roles for ref in pack.references},
        "object_references": condition_ref_payload(object_refs),
        "scene_references": condition_ref_payload(scene_refs),
        "videos": videos,
    }


def write_outputs(results: dict[str, Any], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_json = output_json.parent / "summary.json"
    summary_json.write_text(json.dumps(build_results_summary(results), ensure_ascii=False, indent=2), encoding="utf-8")
    process_file(output_json, summary_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Entity Fidelity from a CSV/XLSX that provides model video paths.")
    parser.add_argument("--xlsx", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--board", choices=["board1", "board2", "board3"])
    parser.add_argument("--out-dir", type=Path, default=Path("ef_eval/finalef/outputs/xlsx_entity_fidelity"))
    parser.add_argument("--run-timestamp", required=True)
    parser.add_argument("--sample-json-root", type=Path, default=Path("generated_xlsx_testvideos"))
    parser.add_argument("--video-reference-work-root", type=Path, default=None)
    parser.add_argument("--video-reference-frames", type=int, default=16)
    parser.add_argument("--video-reference-count", type=int, default=3)
    parser.add_argument("--force-video-reference-build", action="store_true")
    parser.add_argument("--source-xlsx-root", type=Path, default=DEFAULT_WORKSPACE / "datasets/cdatasets/final_success_tables_20260521/renumbered_20260522")
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--skip-existing", type=str_bool, default=True)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--face-weight", type=float, default=0.6)
    parser.add_argument("--reference-top-k", type=int, default=1)
    parser.add_argument("--min-frame-ef", type=float, default=0.1)
    parser.add_argument("--cartoon-min-body-ef", type=float, default=0.6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-device", default="cuda:0")
    parser.add_argument("--face-model-root", default="ef_eval/model_assets/OpenS2V-Weight/face_extractor")
    parser.add_argument("--curricular-checkpoint", default="ef_eval/model_assets/OpenS2V-Weight/glint360k_curricular_face_r101_backbone.bin")
    parser.add_argument("--face-recognition-model", default="ef_eval/model_assets/face_recognizers/elasticface_r100.pth")
    parser.add_argument("--face-recognition-name", default="elasticface_r100")
    parser.add_argument("--dinov2-model", default="ef_eval/hf_home/hub/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415")
    parser.add_argument("--siglip-model", default="ef_eval/hf_home/hub/models--google--siglip-base-patch16-224/snapshots/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed")
    parser.add_argument("--reference-human-root", default=str(DEFAULT_WORKSPACE / "datasets/human"))
    parser.add_argument("--reference-body-mask-root", default=str(DEFAULT_WORKSPACE / "datasets/human_masks_sam3/masks"))
    parser.add_argument("--disable-reference-body-mask", action="store_true")
    parser.add_argument("--disable-cartoon-head-ef", action="store_true")
    parser.add_argument("--detector", choices=["yolo-world", "face-expanded"], default="yolo-world")
    parser.add_argument("--yoloworld-config", default=str(EVAL_PY_ROOT / "yolo_world_clip_fix.py"))
    parser.add_argument("--yoloworld-checkpoint", default="ef_eval/model_assets/OpenS2V-Weight/yolo_world_v2_l_image_prompt_adapter-719a7afb.pth")
    parser.add_argument("--yoloworld-repo", default=str(DEFAULT_WORKSPACE / "OpenS2V-Nexus/eval/utils/yoloworld"))
    parser.add_argument("--person-score-threshold", type=float, default=0.15)
    parser.add_argument("--object-score-threshold", type=float, default=0.05)
    parser.add_argument("--disable-object-ef", action="store_true")
    parser.add_argument("--disable-scene-ef", action="store_true")
    parser.add_argument("--cartoon-face-detector", choices=["hysts-yolov3", "yolov8-anime-face", "none"], default="yolov8-anime-face")
    parser.add_argument("--cartoon-face-score-threshold", type=float, default=0.15)
    parser.add_argument("--hysts-yolov3-config", default="ef_eval/model_assets/anime_face_detectors/mmdet3/mmdet_anime-face_yolov3.py")
    parser.add_argument("--hysts-yolov3-checkpoint", default="ef_eval/model_assets/anime_face_detectors/mmdet3/mmdet_anime-face_yolov3.pth")
    parser.add_argument("--yolov8-anime-face-checkpoint", default="ef_eval/model_assets/anime_face_detectors/yolov8/yolov8x6_animeface.pt")
    parser.add_argument("--yolov8-device", default="0")
    parser.add_argument("--save-frame-boxes", type=str_bool, default=True)
    parser.add_argument("--make-frame-box-sheet", type=str_bool, default=True)
    parser.add_argument("--keep-frame-box-images", type=str_bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shard_count < 1 or args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError(f"Invalid shard {args.shard_index}/{args.shard_count}")

    if bool(args.xlsx) == bool(args.csv):
        raise ValueError("Provide exactly one of --csv or --xlsx")
    input_table = args.csv if args.csv else args.xlsx
    sample_json_root = args.sample_json_root
    if args.csv and csv_has_reference_videos(input_table):
        build_root = args.video_reference_work_root or (args.sample_json_root.parent / f"{args.sample_json_root.name}_video_refs")
        print(f"Building CSV video-reference EF samples: {build_root}", flush=True)
        detector_args = SimpleNamespace(
            detector=args.detector,
            detector_device=args.detector_device,
            person_score_threshold=args.person_score_threshold,
            yoloworld_config=args.yoloworld_config,
            yoloworld_checkpoint=args.yoloworld_checkpoint,
            yoloworld_repo=args.yoloworld_repo,
            person_detection_texts=["person"],
        )
        manifest = build_samples_from_csv(
            csv_path=input_table,
            output_root=build_root,
            detector_args=detector_args,
            board_filter=args.board or "",
            num_reference_frames=args.video_reference_frames,
            selected_reference_count=args.video_reference_count,
            force=args.force_video_reference_build,
        )
        sample_json_root = Path(manifest["samples_root"])
        print(
            "CSV video-reference samples ready: "
            f"samples={manifest['sample_count']} unique_reference_videos={manifest['unique_reference_videos']} "
            f"sample_json_root={sample_json_root}",
            flush=True,
        )
    jobs = load_jobs(input_table, args.board or "")
    if args.limit_rows is not None:
        jobs = jobs[: args.limit_rows]
    jobs = split_shard(jobs, args.shard_index, args.shard_count)
    if not jobs:
        print("No jobs selected after limit/shard filtering; skipping.", flush=True)
        return

    sample_paths = []
    for job in jobs:
        sample_paths.append(ensure_sample_json(job, sample_json_root, args.source_xlsx_root))

    person_detection_texts = collect_person_detection_texts(sorted(set(sample_paths)))
    detector_args = make_eval_args(args, Path("dummy_result.json"), None, person_detection_texts)
    print(f"Loading face model once on {args.device}: {args.face_recognition_name}", flush=True)
    face_branch = FaceBranch(
        args.face_model_root,
        args.curricular_checkpoint,
        recognition_model=args.face_recognition_model,
        recognition_name=args.face_recognition_name,
        device=args.device,
    )
    print(f"Loading body embedders once on {args.device}", flush=True)
    body_branch = BodyBranch(args.dinov2_model, args.siglip_model, device=args.device)
    print(f"Loading detector once on {args.detector_device}; prompts={person_detection_texts}", flush=True)
    detector = build_detector(detector_args)
    any_cartoon = any(has_cartoon_human(load_human_reference_pack(path)) for path in set(sample_paths))
    cartoon_face_detector = None
    if any_cartoon and not args.disable_cartoon_head_ef and args.cartoon_face_detector != "none":
        print(f"Loading cartoon face detector once: {args.cartoon_face_detector}", flush=True)
        cartoon_face_detector = build_cartoon_face_detector(detector_args)

    reference_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    total_tasks = len(jobs)
    print(f"Running {total_tasks} xlsx jobs. RUN_TIMESTAMP={args.run_timestamp}", flush=True)
    for task_index, job in enumerate(jobs, start=1):
        sample_json = ensure_sample_json(job, sample_json_root, args.source_xlsx_root)
        video_path = Path(job["video_path"])
        run_dir = args.out_dir / f'{job["tag"]}_{args.run_timestamp}' / job["model"]
        output_json = run_dir / "result.json"
        summary_json = run_dir / "summary.json"
        print(f'[{task_index}/{total_tasks}] {job["tag"]} / {job["model"]} -> {video_path}', flush=True)
        if args.skip_existing and output_json.exists() and summary_json.exists():
            print(f"[{task_index}/{total_tasks}] skip existing: {output_json}", flush=True)
            continue

        object_refs = [] if args.disable_object_ef else load_condition_references(sample_json, "object")
        scene_refs = [] if args.disable_scene_ef else load_condition_references(sample_json, "scene")
        scene_branch = SceneBranch(body_branch, scene_refs) if scene_refs else None
        object_branch = (
            ObjectBranch(body_branch, object_refs, detector_factory=lambda prompt: build_text_detector(detector_args, prompt))
            if object_refs
            else None
        )
        frame_box_dir = run_dir / "frame_boxes" if args.save_frame_boxes else None
        eval_args = make_eval_args(args, output_json, frame_box_dir, person_detection_texts)
        object_frame_box_dir = frame_box_dir / f"{video_path.stem}_object" if frame_box_dir is not None and object_branch else None
        results = result_payload(
            sample_json,
            video_path,
            eval_args,
            face_branch,
            body_branch,
            detector,
            reference_cache,
            cartoon_face_detector,
            object_refs,
            scene_refs,
            object_branch,
            scene_branch,
            object_frame_box_dir,
        )
        write_outputs(results, output_json)
        if args.save_frame_boxes and args.make_frame_box_sheet and frame_box_dir is not None and frame_box_dir.exists():
            make_branch_sheets(frame_box_dir, run_dir / "frame_boxes_contact_sheet.jpg", columns=8, thumb_width=320)
            if not args.keep_frame_box_images:
                try:
                    shutil.rmtree(frame_box_dir, ignore_errors=True)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    print(f"WARN: failed to remove frame box dir {frame_box_dir}: {exc}", flush=True)

    summary_path = args.out_dir / f"summary_{args.run_timestamp}.csv"
    write_csv(collect_rows(args.out_dir, run_timestamp=args.run_timestamp), summary_path)
    ranking_path = write_rankings(
        sorted(args.out_dir.glob("summary_*.csv")),
        FINAL_EF_ROOT / "outputs" / "board123_combined_rankings",
    )
    print(f"Done. Results: {args.out_dir}")
    print(f"Summary: {summary_path}")
    print(f"Ranking: {ranking_path}")


if __name__ == "__main__":
    main()
