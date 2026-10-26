import csv
import hashlib
import json
import pathlib
import tempfile
import unittest
from multiref_eval.cli import check_readiness, metric_names
from multiref_eval.validation import validate_manifest, audit_dataset

class IntegrationGuards(unittest.TestCase):
    def test_original_scoring_files_unchanged(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        lock = json.loads((root / "upstream.lock.json").read_text())
        patches = json.loads((root / "upstream.patches.json").read_text())
        for name, expected in lock["file_sha256"].items():
            expected = patches.get(name, {}).get("sha256", expected)
            if name.startswith("finaleval/"):
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected, name)

    def test_unknown_and_unadapted_metrics_rejected(self):
        with self.assertRaises(ValueError):
            metric_names("made_up")
        for name in metric_names("all"):
            if name not in {"visual_quality", "audio_quality", "voice_timbre_similarity"}:
                with self.assertRaises(ValueError):
                    check_readiness([name])

    def test_relative_paths_duplicates_and_required_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "generated.mp4").write_bytes(b"fixture")
            path = root / "input.csv"
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sample_id", "model_name", "prompt", "video_path"])
                writer.writerow(["1", "model", "test", "generated.mp4"])
            samples, errors = validate_manifest(path, ["visual_quality"])
            self.assertFalse(errors)
            self.assertEqual(samples[0].video_path, str(root / "generated.mp4"))
            _, errors = validate_manifest(path, ["voice_timbre_similarity"])
            self.assertTrue(any("reference audio" in e for e in errors))
            with path.open("a") as f:
                f.write("1,model,test,missing.mp4\n")
            _, errors = validate_manifest(path, ["visual_quality"])
            self.assertTrue(any("duplicate" in e for e in errors))
            self.assertTrue(any("missing video_path" in e for e in errors))

    def test_dataset_audio_is_reported_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "data").mkdir()
            source = root / "data/samples.jsonl"
            source.write_text(json.dumps({"sample_id": "x", "prompt": "Use @Audio1", "image_1": "missing.jpg"}))
            before = source.read_bytes()
            result = audit_dataset(root)
            self.assertEqual(result["audio_reference_sample_ids"], ["x"])
            self.assertEqual(len(result["missing_visual_files"]), 1)
            self.assertEqual(before, source.read_bytes())

if __name__ == "__main__":
    unittest.main()
