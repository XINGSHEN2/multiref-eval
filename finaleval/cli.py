from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from finaleval.common.registry import ALL_MAIN_METRICS, CALLABLE_METRICS, IMPLEMENTED_METRICS
from finaleval.common.schema import read_input_csv


METRIC_MODULES = {
    "visual_quality": "finaleval.basic_quality.visual_quality",
    "audio_quality": "finaleval.basic_quality.audio_quality",
    "anatomical_consistency": "finaleval.basic_quality.anatomical_consistency",
    "entity_fidelity": "finaleval.entity_fidelity.entity_fidelity",
    "detail_preservation": "finaleval.entity_fidelity.detail_preservation",
    "detail_binding": "finaleval.entity_fidelity.detail_binding",
    "speech_lip_synchronization": "finaleval.audio_video_consistency.speech_lip_synchronization",
    "visual_event_sound_alignment": "finaleval.audio_video_consistency.visual_event_sound_alignment",
    "source_correctness": "finaleval.audio_video_consistency.source_correctness",
    "voice_timbre_similarity": "finaleval.audio_video_consistency.voice_timbre_similarity",
    "visual_task_following": "finaleval.instruction_following.visual_task_following",
    "audio_task_following": "finaleval.instruction_following.audio_task_following",
    "temporal_order_following": "finaleval.instruction_following.temporal_order_following",
    "speech_content_accuracy": "finaleval.instruction_following.speech_content_accuracy",
}


def load_evaluator(metric: str):
    module = importlib.import_module(METRIC_MODULES[metric])
    return module.evaluate


def parse_metrics(value: str) -> list[str]:
    metrics = [item.strip() for item in value.split(",") if item.strip()]
    if not metrics or metrics == ["all"]:
        return sorted(ALL_MAIN_METRICS)
    return metrics


def run_evaluate(args: argparse.Namespace) -> None:
    input_csv = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    samples = read_input_csv(input_csv)
    metrics = parse_metrics(args.metrics)

    unsupported = [metric for metric in metrics if metric not in CALLABLE_METRICS]
    if unsupported:
        raise SystemExit(
            "Unknown metric(s): "
            + ", ".join(unsupported)
            + ". Registered main metrics: "
            + ", ".join(sorted(ALL_MAIN_METRICS))
        )

    run_config = {
        "input": str(input_csv),
        "output": str(output_root),
        "metrics": metrics,
        "voice_timbre_min_speech_sec": args.voice_timbre_min_speech_sec,
        "voice_timbre_device": args.voice_timbre_device,
        "model_cache": args.model_cache,
        "paste_per_sample": args.paste_per_sample,
        "auto_paste": not args.disable_auto_paste,
    }
    (output_root / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "input_manifest.csv").write_text(input_csv.read_text(encoding="utf-8-sig"), encoding="utf-8")

    for metric in metrics:
        evaluator = load_evaluator(metric)
        kwargs = {}
        if metric == "voice_timbre_similarity":
            kwargs.update(
                min_speech_sec=args.voice_timbre_min_speech_sec,
                device=args.voice_timbre_device,
                model_cache=args.model_cache,
            )
        if metric == "speech_lip_synchronization":
            kwargs["stable_face_filter"] = not args.disable_stable_face_filter
        if metric == "entity_fidelity":
            kwargs["auto_paste"] = not args.disable_auto_paste
            if args.paste_per_sample:
                kwargs["paste_per_sample"] = args.paste_per_sample
        metric_dir = evaluator(samples=samples, output_root=output_root, **kwargs)
        marker = "implemented" if metric in IMPLEMENTED_METRICS else "interface-only"
        print(f"[done] {metric} ({marker}): {metric_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multiref-compass")
    sub = parser.add_subparsers(dest="command", required=True)

    evaluate = sub.add_parser("evaluate", help="Run MultiRef-Compass metrics from a CSV manifest.")
    evaluate.add_argument("--input", required=True, help="Input CSV path.")
    evaluate.add_argument("--output", required=True, help="Output run directory.")
    evaluate.add_argument("--metrics", default="all", help="Comma-separated metric names or all.")
    evaluate.add_argument("--model-cache", default=None, help="Directory for downloaded/cached model weights.")
    evaluate.add_argument("--voice-timbre-min-speech-sec", type=float, default=2.5)
    evaluate.add_argument("--voice-timbre-device", choices=["auto", "cpu", "cuda"], default="auto")
    evaluate.add_argument(
        "--disable-stable-face-filter",
        action="store_true",
        help="Disable the planned stable-face filter before speech_lip_synchronization.",
    )
    evaluate.add_argument(
        "--paste-per-sample",
        default=None,
        help=(
            "Paste Artifact per-sample CSV used to produce paste-adjusted Entity Fidelity. "
            "Only applies when --metrics includes entity_fidelity."
        ),
    )
    evaluate.add_argument(
        "--disable-auto-paste",
        action="store_true",
        help="Disable automatic Paste Artifact judging before entity_fidelity.",
    )
    evaluate.set_defaults(func=run_evaluate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
