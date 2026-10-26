# Model Weights and External Assets

Model weights are not committed to this repository. Keep downloaded weights and
runtime caches outside the source tree when possible.

Recommended cache configuration:

```bash
export MULTIREF_COMPASS_MODEL_CACHE=/path/to/multiref-compass-model-cache
export MULTIREF_COMPASS_CACHE_ROOT=/path/to/multiref-compass-runtime-cache
```

`MULTIREF_COMPASS_MODEL_CACHE` is used by public entrypoints when a model cache
is exposed. `MULTIREF_COMPASS_CACHE_ROOT` is forwarded to legacy backends for
dependency downloads, conda environments, HuggingFace cache, torch cache, and
temporary runtime files. Legacy `FINALEVAL_*` variables are still accepted for
compatibility.

## Summary

| Component | Metrics | Required Files / Models | Download Method |
|---|---|---|---|
| DOVER++ | `visual_quality`, `audio_quality` video branch | `DOVER_plus_plus.pth`, `DOVER.pth` | Auto-download via BQ setup or manual download from DOVER release / HuggingFace. |
| AudioBox Aesthetics | `audio_quality` audio branch | `checkpoint.pt` | Auto-download from Meta public file URL. |
| Aesthetic Predictor V2.5 / ImageBind | optional BQ aesthetic branch | local converted checkpoint plus ImageBind weights | Auto-prepared by BQ setup when enabled. |
| SpeechBrain ECAPA-TDNN | `voice_timbre_similarity` | `speechbrain/spkrec-ecapa-voxceleb` | HuggingFace download on first use or pre-download with `huggingface-cli`. |
| LatentSync SyncNet | `speech_lip_synchronization` | `syncnet_v2.model` plus auxiliary LatentSync assets | HuggingFace download from `ByteDance/LatentSync-1.6` or manual placement. |
| Entity Fidelity detectors | `entity_fidelity` automatic branches | YOLO-World, CurricularFace, anime face detector weights | Manual placement; paths are configurable. |
| MLLM judges | MLLM metrics | no local model weights | Requires Gemini/Vertex credentials instead of local weights. |

## Basic Quality Weights

Used by:

- `visual_quality`
- `audio_quality`

The Basic Quality backend expects DOVER++ files under:

```text
finaleval/basic_quality/backends/legacy_bq/Objective/Video/DOVER/pretrained_weights/
  DOVER.pth
  DOVER_plus_plus.pth
```

The backend setup script downloads them with:

```text
https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth
https://huggingface.co/teowu/DOVER/resolve/main/DOVER_plus_plus.pth?download=true
```

If HuggingFace is slow or blocked, set a mirror before running:

```bash
export BQ_HF_MIRROR_ENDPOINT=https://hf-mirror.com
```

Manual download:

```bash
mkdir -p finaleval/basic_quality/backends/legacy_bq/Objective/Video/DOVER/pretrained_weights

curl -L \
  https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth \
  -o finaleval/basic_quality/backends/legacy_bq/Objective/Video/DOVER/pretrained_weights/DOVER.pth

curl -L \
  "https://huggingface.co/teowu/DOVER/resolve/main/DOVER_plus_plus.pth?download=true" \
  -o finaleval/basic_quality/backends/legacy_bq/Objective/Video/DOVER/pretrained_weights/DOVER_plus_plus.pth
```

AudioBox Aesthetics is used by the audio-quality branch. The backend downloads:

```text
https://dl.fbaipublicfiles.com/audiobox-aesthetics/checkpoint.pt
```

Default cache location:

```text
$MULTIREF_COMPASS_CACHE_ROOT/basic-quality-cache/weights/audiobox-aesthetics/checkpoint.pt
```

Manual download:

```bash
mkdir -p "$MULTIREF_COMPASS_CACHE_ROOT/basic-quality-cache/weights/audiobox-aesthetics"
curl -L \
  https://dl.fbaipublicfiles.com/audiobox-aesthetics/checkpoint.pt \
  -o "$MULTIREF_COMPASS_CACHE_ROOT/basic-quality-cache/weights/audiobox-aesthetics/checkpoint.pt"
```

Verification:

```bash
test -s finaleval/basic_quality/backends/legacy_bq/Objective/Video/DOVER/pretrained_weights/DOVER_plus_plus.pth
test -s "$MULTIREF_COMPASS_CACHE_ROOT/basic-quality-cache/weights/audiobox-aesthetics/checkpoint.pt"
```

## Voice Timbre Similarity

Used by:

- `voice_timbre_similarity`

Default model:

```text
speechbrain/spkrec-ecapa-voxceleb
```

The public evaluator uses `speechbrain.inference.speaker.EncoderClassifier`.
On first use, SpeechBrain downloads the model from HuggingFace into the model
cache:

```bash
export MULTIREF_COMPASS_MODEL_CACHE=/path/to/multiref-compass-model-cache
MULTIREF_COMPASS_METRICS=voice_timbre_similarity \
bash scripts/run_auto_from_csv.sh examples/demo.csv outputs/timbre_demo
```

Manual pre-download with HuggingFace:

```bash
python -m pip install -U huggingface_hub
huggingface-cli download speechbrain/spkrec-ecapa-voxceleb \
  --local-dir "$MULTIREF_COMPASS_MODEL_CACHE/spkrec-ecapa-voxceleb"
```

Then run with:

```bash
MULTIREF_COMPASS_MODEL_CACHE="$MULTIREF_COMPASS_MODEL_CACHE/spkrec-ecapa-voxceleb" \
MULTIREF_COMPASS_METRICS=voice_timbre_similarity \
bash scripts/run_auto_from_csv.sh examples/demo.csv outputs/timbre_demo
```

Device selection:

```bash
export VOICE_TIMBRE_DEVICE=auto   # auto, cpu, or cuda
export VOICE_TIMBRE_MIN_SPEECH_SEC=2.5
```

Verification:

```bash
python - <<'PY'
from speechbrain.inference.speaker import EncoderClassifier
EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"},
)
print("speechbrain ecapa ok")
PY
```

## Speech-Lip Synchronization / VSLS

Used by:

- `speech_lip_synchronization`

The VSLS backend uses LatentSync's SyncNet model. The internal shell expects:

```bash
export LATENTSYNC_DIR=/path/to/LatentSync
export MODEL_PATH="$LATENTSYNC_DIR/checkpoints/auxiliary/syncnet_v2.model"
```

Required minimum file:

```text
$LATENTSYNC_DIR/checkpoints/auxiliary/syncnet_v2.model
```

Recommended LatentSync auxiliary files:

```text
$LATENTSYNC_DIR/checkpoints/auxiliary/syncnet_v2.model
$LATENTSYNC_DIR/checkpoints/auxiliary/sfd_face.pth
$LATENTSYNC_DIR/checkpoints/auxiliary/koniq_pretrained.pkl
$LATENTSYNC_DIR/checkpoints/whisper/tiny.pt
```

Manual download from HuggingFace:

```bash
export LATENTSYNC_DIR=/path/to/LatentSync
mkdir -p "$LATENTSYNC_DIR/checkpoints/auxiliary" "$LATENTSYNC_DIR/checkpoints/whisper"

curl -L \
  https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/auxiliary/syncnet_v2.model \
  -o "$LATENTSYNC_DIR/checkpoints/auxiliary/syncnet_v2.model"

curl -L \
  https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/auxiliary/sfd_face.pth \
  -o "$LATENTSYNC_DIR/checkpoints/auxiliary/sfd_face.pth"

curl -L \
  https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/auxiliary/koniq_pretrained.pkl \
  -o "$LATENTSYNC_DIR/checkpoints/auxiliary/koniq_pretrained.pkl"

curl -L \
  https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/whisper/tiny.pt \
  -o "$LATENTSYNC_DIR/checkpoints/whisper/tiny.pt"
```

Run:

```bash
LATENTSYNC_DIR=/path/to/LatentSync \
MODEL_PATH=/path/to/LatentSync/checkpoints/auxiliary/syncnet_v2.model \
VSLS_DEVICE=cuda \
MULTIREF_COMPASS_METRICS=speech_lip_synchronization \
bash scripts/run_auto_from_csv.sh examples/demo.csv outputs/vsls_demo
```

If Silero VAD downloads are blocked, use a simpler VAD backend:

```bash
export VSLS_VAD_BACKEND=energy
```

Verification:

```bash
test -s "$LATENTSYNC_DIR/checkpoints/auxiliary/syncnet_v2.model"
```

## Entity Fidelity Automatic Branches

Used by:

- `entity_fidelity`

The automatic Entity Fidelity backend can use face recognition, body/object
detection, scene/object branches, and optional cartoon face detectors. The
default paths expected by the legacy backend are:

```text
ef_eval/model_assets/OpenS2V-Weight/glint360k_curricular_face_r101_backbone.bin
ef_eval/model_assets/OpenS2V-Weight/yolo_world_v2_l_image_prompt_adapter-719a7afb.pth
ef_eval/model_assets/anime_face_detectors/mmdet3/mmdet_anime-face_yolov3.pth
ef_eval/model_assets/anime_face_detectors/yolov8/yolov8x6_animeface.pt
```

These paths are historical defaults. For an open-source checkout, place the
files under a cache directory and pass explicit paths through the backend
environment or wrapper configuration.

Recommended layout:

```text
$MULTIREF_COMPASS_MODEL_CACHE/entity_fidelity/
  OpenS2V-Weight/
    glint360k_curricular_face_r101_backbone.bin
    yolo_world_v2_l_image_prompt_adapter-719a7afb.pth
  anime_face_detectors/
    mmdet3/mmdet_anime-face_yolov3.pth
    yolov8/yolov8x6_animeface.pt
```

Important: this repository does not redistribute these weights. Download them
from the upstream projects or the model release associated with the EF backend,
then configure paths explicitly.

Path variables used by the legacy EF shell:

```bash
export CURRICULAR="$MULTIREF_COMPASS_MODEL_CACHE/entity_fidelity/OpenS2V-Weight/glint360k_curricular_face_r101_backbone.bin"
export YOLO_CKPT="$MULTIREF_COMPASS_MODEL_CACHE/entity_fidelity/OpenS2V-Weight/yolo_world_v2_l_image_prompt_adapter-719a7afb.pth"
```

Additional Python backend options include:

```text
--curricular-checkpoint
--yoloworld-checkpoint
--hysts-yolov3-checkpoint
--yolov8-anime-face-checkpoint
```

Verification:

```bash
test -s "$CURRICULAR"
test -s "$YOLO_CKPT"
```

## MLLM-Only Metrics

The following metrics do not require local model weights, but they do require
Gemini/Vertex credentials:

- `anatomical_consistency`
- `detail_preservation`
- `detail_binding`
- `visual_event_sound_alignment`
- `source_correctness`
- `visual_task_following`
- `audio_task_following`
- `temporal_order_following`
- `speech_content_accuracy`

Configure:

```bash
export MULTIREF_COMPASS_GEMINI_CREDENTIALS=/path/to/service-account.json
export GOOGLE_CLOUD_LOCATION=global
```

Run one MLLM metric:

```bash
MULTIREF_COMPASS_METRICS=visual_event_sound_alignment \
bash scripts/run_auto_from_csv.sh examples/demo.csv outputs/ac_demo
```

## Offline or Restricted-Network Environments

If the machine cannot access HuggingFace, GitHub, or public model URLs:

1. Download weights on a machine with internet access.
2. Copy them into `MULTIREF_COMPASS_MODEL_CACHE` or the metric-specific expected path.
3. Set explicit environment variables such as `LATENTSYNC_DIR`, `MODEL_PATH`,
   `CURRICULAR`, `YOLO_CKPT`, and `VOICE_TIMBRE_DEVICE`.
4. Use `VSLS_VAD_BACKEND=energy` if Silero VAD cannot download its model.

For Basic Quality, the backend also supports:

```bash
export BQ_HF_MIRROR_ENDPOINT=https://hf-mirror.com
export BQ_GITHUB_MIRROR_PREFIX=
```

## Quick Dependency Check

```bash
python - <<'PY'
from pathlib import Path
checks = {
    "DOVER++": Path("finaleval/basic_quality/backends/legacy_bq/Objective/Video/DOVER/pretrained_weights/DOVER_plus_plus.pth"),
}
for name, path in checks.items():
    print(f"{name}: {'ok' if path.exists() and path.stat().st_size > 0 else 'missing'} -> {path}")
PY
```

For a full run, inspect each metric's `per_sample.csv`. Missing weights or
blocked downloads are reported as `status=error` with the backend error in the
`reason` column.
