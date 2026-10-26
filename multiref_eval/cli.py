import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from finaleval.common.registry import MAIN_METRICS, ALL_MAIN_METRICS
from .validation import audit_dataset, validate_manifest

# The remaining original wrappers depend on Gemini or unresolved export wiring.
AUTOMATIC = {"visual_quality", "audio_quality", "voice_timbre_similarity"}

def metric_names(value):
    names = sorted(ALL_MAIN_METRICS) if value == "all" else list(dict.fromkeys(x.strip() for x in value.split(",") if x.strip()))
    if not names or set(names) - ALL_MAIN_METRICS:
        raise ValueError("Select registered metric names; use list-metrics")
    return names

def check_readiness(metrics):
    blocked = set(metrics) - AUTOMATIC
    if blocked:
        raise ValueError("Not integrated for open judges / export wiring: " + ", ".join(sorted(blocked)) + ". Original code is retained, but no Gemini fallback is allowed by this entry point.")

def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser(prog="multiref-eval")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-metrics")
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--config", default="configs/local.example.json")
    audit = sub.add_parser("audit-dataset")
    audit.add_argument("--root", required=True)
    for name in ("validate", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--input", required=True)
        p.add_argument("--metrics", required=True)
        if name == "evaluate":
            p.add_argument("--output", required=True)
            p.add_argument("--config", default="configs/local.example.json")
    args = parser.parse_args()
    try:
        if args.command == "list-metrics":
            emit([{"name": m.name, "group": m.group, "status": "original_backend_requires_environment" if m.name in AUTOMATIC else "adaptation_pending"} for m in MAIN_METRICS])
        elif args.command == "doctor":
            config = json.loads(Path(args.config).read_text())
            emit({"python": sys.version.split()[0], "ffmpeg": shutil.which("ffmpeg"),
                  "packages": {m: importlib.util.find_spec(m) is not None for m in ("torch", "torchaudio", "speechbrain", "soundfile", "yaml")},
                  "paths": {k: {"path": config[k], "exists": Path(config[k]).exists()} for k in ("dataset_root", "model_cache", "runtime_cache")},
                  "judges": config["judges"], "note": "Discovery only; weights, CUDA and backend execution are not verified."})
        elif args.command == "audit-dataset":
            result = audit_dataset(args.root)
            emit(result)
            if result["missing_visual_files"] or result["duplicate_ids"]:
                return 2
        else:
            metrics = metric_names(args.metrics)
            samples, errors = validate_manifest(args.input, metrics)
            if errors:
                emit({"valid": False, "samples": len(samples), "errors": errors})
                return 2
            if args.command == "validate":
                emit({"valid": True, "samples": len(samples), "note": "File existence only; not media decode or semantic validation."})
                return 0
            check_readiness(metrics)
            config = json.loads(Path(args.config).read_text())
            output = Path(args.output).resolve()
            if output.exists() and any(output.iterdir()):
                raise ValueError("Output directory must be new or empty; stale scores cannot be reused")
            output.mkdir(parents=True, exist_ok=True)
            emit({"metrics": metrics, "output": str(output)})
            env = os.environ.copy()
            env["MULTIREF_COMPASS_MODEL_CACHE"] = str(Path(config["model_cache"]).expanduser().resolve())
            env["MULTIREF_COMPASS_CACHE_ROOT"] = str(Path(config["runtime_cache"]).expanduser().resolve())
            # Record input identity without including endpoint credentials.
            (output / "adaptation_run.json").write_text(json.dumps({"metrics": metrics, "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(), "upstream_revision": "03da121e88e241da38b41324291dfa7b8773d132", "scoring": "original", "judges_used": []}, indent=2))
            subprocess.run([sys.executable, "-m", "finaleval.cli", "evaluate", "--input", str(Path(args.input).resolve()), "--output", str(output), "--metrics", ",".join(metrics), "--model-cache", env["MULTIREF_COMPASS_MODEL_CACHE"]], env=env, check=True)
            failures = []
            expected = {(s.sample_id, s.model_name) for s in samples}
            for metric in metrics:
                path = output / metric / "per_sample.csv"
                if not path.is_file():
                    failures.append(f"{metric}: missing results")
                    continue
                with path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if len(rows) != len(samples) or {(r.get("sample_id"), r.get("model_name")) for r in rows} != expected:
                    failures.append(f"{metric}: result coverage mismatch")
                if any(r.get("status") not in ("scored", "skipped") or (r.get("status") == "scored" and not r.get("score")) for r in rows):
                    failures.append(f"{metric}: failed or invalid rows; inspect per_sample.csv")
            if failures:
                emit({"status": "failed", "errors": failures})
                return 2
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
