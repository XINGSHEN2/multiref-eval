# MultiRef-Compass

![MultiRef-Compass overview](overall1.png)
[Dataset](https://huggingface.co/datasets/zxhhhhhh/MultiRef-Compass)
[Paper](https://arxiv.org/abs/2607.14189)

MultiRef-Compass is an open evaluation toolkit for multi-reference multimodal
video generation. It is designed to support reproducible, model-agnostic
assessment of generated videos under text, visual-reference, and
audio-reference conditions. The toolkit provides a fixed taxonomy of 14 public
metrics, a stable CSV input schema, and a unified output format for both
sample-level diagnosis and model-level ranking.

## Overview

- **Four evaluation dimensions.** MultiRef-Compass groups metrics into Basic
  Quality, Entity Fidelity, Audio-Video Consistency, and Instruction Following.
- **Fourteen public metrics.** The released taxonomy covers perceptual quality,
  anatomical plausibility, reference fidelity, audio-visual alignment, voice
  timbre similarity, and instruction following.
- **Unified input and output schema.** All metrics consume the same CSV manifest
  and write the same public output structure:
  `per_sample.csv`, `model_summary.csv`, `ranking.md`, and `details.json`.
- **Hybrid automatic and MLLM-based evaluation.** MultiRef-Compass combines
  classical media-analysis pipelines, learned quality models, speaker
  embeddings, and multimodal language model judges.


## Design Principles

MultiRef-Compass is intended for comparative research rather than ad hoc
inspection. The public interface is therefore kept deliberately simple:

- Inputs are stored in a flat CSV manifest so that the same generated videos can
  be evaluated by different metric groups.
- Metrics report both per-sample evidence and model-level aggregates.
- Optional references, such as images, videos, or audio clips, are supplied
  through explicit columns rather than dataset-specific hidden files.
- Metric backends may evolve, but the public metric names, required columns, and
  output file names are treated as stable.

## Metric Taxonomy

MultiRef-Compass reports 14 main metrics.

| Group | Metric | Code Name |
|---|---|---|
| Basic Quality | Visual Quality | `visual_quality` |
| Basic Quality | Audio Quality | `audio_quality` |
| Basic Quality | Anatomical Consistency | `anatomical_consistency` |
| Entity Fidelity | Entity Fidelity | `entity_fidelity` |
| Entity Fidelity | Detail Preservation | `detail_preservation` |
| Entity Fidelity | Detail Binding | `detail_binding` |
| Audio-Video Consistency | Speech-Lip Synchronization / VSLS | `speech_lip_synchronization` |
| Audio-Video Consistency | Visual Event-Sound Alignment | `visual_event_sound_alignment` |
| Audio-Video Consistency | Source Correctness | `source_correctness` |
| Audio-Video Consistency | Voice Timbre Similarity | `voice_timbre_similarity` |
| Instruction Following | Visual Task Following | `visual_task_following` |
| Instruction Following | Audio Task Following | `audio_task_following` |
| Instruction Following | Temporal Order Following | `temporal_order_following` |
| Instruction Following | Speech Content Accuracy | `speech_content_accuracy` |

Auxiliary components are not counted as standalone main metrics:

- Paste Artifact is normalized as a penalty and multiplied into the final
  Entity Fidelity score.
- Stable Speaking Frontal Face is an optional pre-filter for
  Speech-Lip Synchronization / VSLS.
- Visual Task Following may report category breakdowns such as Important
  Action, Wearing Requirement, Basic Requirement, and Other Detail. These are
  breakdown columns, not extra metrics.

## Installation

Clone the repository and install the Python package:

```bash
git clone <your-multiref-compass-repo>
cd MultiRef_Compass
python -m pip install -e .
python -m pip install -r requirements.txt
```

System requirements:

- Python 3.10 or newer is recommended.
- `ffmpeg` must be available on `PATH`.
- CUDA is recommended for video/audio model backends, but some MLLM-only
  checks can run without GPU.

Verify the package import:

```bash
python -m multiref-compass.cli --help
# or, after editable installation:
multiref-compass --help
```

## Credentials

MLLM metrics use Gemini/Vertex. Configure credentials with:

```bash
export MULTIREF_COMPASS_GEMINI_CREDENTIALS=/path/to/service-account.json
export GOOGLE_CLOUD_LOCATION=global
```

The preferred public variable is `MULTIREF_COMPASS_GEMINI_CREDENTIALS` when
using `scripts/run_auto_from_csv.sh`. The legacy
`FINALEVAL_GEMINI_CREDENTIALS` variable is still accepted for compatibility with
lower-level backends.

Do not commit credential files to the repository.

## Model Weights

Model weights are not committed to this repository. Use
`MULTIREF_COMPASS_MODEL_CACHE` to keep downloaded or manually installed weights
outside the source tree:

```bash
export MULTIREF_COMPASS_MODEL_CACHE=/path/to/multiref-compass-model-cache
export MULTIREF_COMPASS_CACHE_ROOT=/path/to/multiref-compass-runtime-cache
```

Common dependencies:

| Component | Used By | Notes |
|---|---|---|
| DOVER++ weights | `visual_quality`, `audio_quality` | Required by Basic Quality video assessment. |
| SpeechBrain ECAPA-TDNN | `voice_timbre_similarity` | Default model: `speechbrain/spkrec-ecapa-voxceleb`. |
| LatentSync / SyncNet weights | `speech_lip_synchronization` | Required for automatic VSLS scoring. |
| Detector / recognition weights | `entity_fidelity` | Required by automatic entity fidelity branches. |

See [docs/model_weights.md](docs/model_weights.md) for weight placement and
cache configuration.

## Input CSV

MultiRef-Compass uses fixed column names. Aliases are intentionally not
inferred.

Recommended columns:

```text
sample_id,model_name,prompt,video_path,board,chinese_prompt,has_speech,has_sound,has_music,reference_audio_path,reference_image_paths,reference_video_path,reference_json_path
```

Minimum common columns:

```text
sample_id,model_name,prompt
```

Paths may be absolute or relative to the CSV file. `reference_image_paths`
accepts either a JSON list string or a semicolon-separated list:

```text
["assets/ref1.png","assets/ref2.jpg"]
```

```text
assets/ref1.png;assets/ref2.jpg
```

`board` is optional metadata. It is only used for grouping, filtering, or
board-level reporting. If it is missing, MultiRef-Compass tries to infer it from
`sample_id` when the ID contains a pattern such as `board2_113`; otherwise the
board field is left empty. No metric should require `board` to compute a score.

### Required Columns by Metric

| Metric | Required Columns | Optional Columns | Skipped When |
|---|---|---|---|
| `visual_quality` | `video_path` | `board` for grouping only | Video is missing. |
| `audio_quality` | `video_path` | `has_sound`, `has_music` | Video or audio track is missing. |
| `anatomical_consistency` | `video_path`, `prompt` | `board` for grouping only | Video is missing. |
| `entity_fidelity` | `video_path` plus visual references | `reference_image_paths`, `reference_video_path`, `reference_json_path` | Visual references are missing. |
| `detail_preservation` | `video_path`, `prompt` plus visual references | `board` for grouping only | Visual references are missing. |
| `detail_binding` | `video_path`, `prompt` plus visual references | `board` for grouping only | Visual references are missing. |
| `speech_lip_synchronization` | `video_path` | `has_speech` | No speech segment is available. |
| `visual_event_sound_alignment` | `video_path` | `prompt` as context only, `has_sound` | No audible non-speech sound with a visible source is present. |
| `source_correctness` | `video_path`, `prompt` | `has_sound`, `has_speech` | Video is missing. |
| `voice_timbre_similarity` | `video_path`, `reference_audio_path` | `has_speech` | Reference audio or generated speech is missing. |
| `visual_task_following` | `video_path`, `prompt` | `board` for grouping only | Video is missing. |
| `audio_task_following` | `video_path`, `prompt` | `has_sound`, `has_music` | No audio requirement is specified. |
| `temporal_order_following` | `video_path`, `prompt` | `board` for grouping only | No temporal order requirement is specified. |
| `speech_content_accuracy` | `video_path`, `prompt` | `has_speech` | No expected speech content is specified. |

## Quick Start

Run all metrics on the demo CSV:

```bash
MULTIREF_COMPASS_METRICS=all \
bash scripts/run_auto_from_csv.sh examples/demo.csv outputs/demo
```

Run a subset:

```bash
MULTIREF_COMPASS_METRICS=visual_event_sound_alignment,visual_task_following \
bash scripts/run_auto_from_csv.sh examples/demo.csv outputs/mllm_subset
```

Run one metric:

```bash
MULTIREF_COMPASS_METRICS=voice_timbre_similarity \
VOICE_TIMBRE_DEVICE=cpu \
bash scripts/run_auto_from_csv.sh examples/demo.csv outputs/timbre_demo
```

Use the Python CLI directly:

```bash
multiref-compass evaluate \
  --input examples/demo.csv \
  --output outputs/demo_cli \
  --metrics all
```

Disable the stable face pre-filter for VSLS:

```bash
multiref-compass evaluate \
  --input examples/demo.csv \
  --output outputs/vsls_no_stable_face \
  --metrics speech_lip_synchronization \
  --disable-stable-face-filter
```

Run Entity Fidelity with a Paste Artifact penalty:

```bash
multiref-compass evaluate \
  --input examples/demo.csv \
  --output outputs/entity_fidelity_with_paste \
  --metrics entity_fidelity
```

By default, `entity_fidelity` first runs the Paste Artifact MLLM judge, then
passes the generated `paste_artifact_per_sample.csv` to the EF backend. The
paste-adjusted EF score is:

```text
final_entity_fidelity = base_entity_fidelity * paste_penalty
```

If you already have a paste per-sample CSV and want to reuse it, pass it
explicitly:

```bash
multiref-compass evaluate \
  --input examples/demo.csv \
  --output outputs/entity_fidelity_with_existing_paste \
  --metrics entity_fidelity \
  --paste-per-sample /path/to/paste_artifact_per_sample.csv
```

`--paste-per-sample` only affects `entity_fidelity`. Other metrics ignore it.
To disable automatic paste judging and run base EF only:

```bash
multiref-compass evaluate \
  --input examples/demo.csv \
  --output outputs/entity_fidelity_base_only \
  --metrics entity_fidelity \
  --disable-auto-paste
```


## Output Format

Each run writes a single output directory:

```text
outputs/demo/
  run_config.json
  input_manifest.csv
  metrics/
    visual_task_following/
      per_sample.csv
      model_summary.csv
      ranking.md
      details.json
    ...
```

Common files:

| File | Description |
|---|---|
| `run_config.json` | Input path, output path, selected metrics, and runtime options. |
| `input_manifest.csv` | Resolved input snapshot used by the run. |
| `metrics/<metric>/per_sample.csv` | Sample-level scores and reasons. |
| `metrics/<metric>/model_summary.csv` | Aggregated score per model. |
| `metrics/<metric>/ranking.md` | Markdown ranking table. |
| `metrics/<metric>/details.json` | Backend metadata, thresholds, and diagnostic information. |

Common `per_sample.csv` fields:

| Field | Description |
|---|---|
| `sample_id` | Sample identifier. |
| `model_name` | Generator/model name. |
| `video_path` | Generated video path. |
| `metric_name` | Metric code name. |
| `score` | Final 1-5 score, empty if not scored. |
| `status` | `scored`, `skipped`, or `error`. |
| `reason` | Skip reason, error message, or judging rationale. |
| `raw_score` | Backend score before public normalization, when available. |

Metric-specific diagnostic fields may be appended.



## Repository Layout

```text
MultiRef_Compass/
  basic_quality/
  entity_fidelity/
  audio_video_consistency/
  instruction_following/
  common/
scripts/
  run_auto_from_csv.sh
  run_all_from_csv.sh
  run_mllm_from_csv.sh
examples/
  demo.csv
prompts/
  rejudge.md
docs/
  metrics.md
  model_weights.md
  output_format.md
```

Backend scripts under `*/backends/` are implementation details and may
change. The public interface is the CSV schema, CLI, and output format described
in this README.

## Citation

MultiRef-Compass is released as a research toolkit for reproducible evaluation
of multi-reference multimodal video generation. We build on the following two prior works:

### OpenS2V-Nexus

```bibtex
@article{yuan2025opens2v,
  title={OpenS2V-Nexus: A Detailed Benchmark and Million-Scale Dataset for Subject-to-Video Generation},
  author={Yuan, Shenghai and He, Xianyi and Deng, Yufan and Ye, Yang and Huang, Jinfa and Lin, Bin and Luo, Jiebo and Yuan, Li},
  journal={arXiv preprint arXiv:2505.20292},
  year={2025}
}
```

### T2AV-Compass

```bibtex
@inproceedings{cao2026t2avcompass,
  title         = {T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation},
  author        = {Cao, Zhe and Wang, Tao and Wang, Jiaming and Wang, Yanghai and Zhang, Yuanxing and Chen, Jialu and Deng, Miao and Wang, Jiahao and Guo, Yubin and Liao, Chenxi and Zhang, Yize and Zhang, Zhaoxiang and Liu, Jiaheng},
  booktitle     = {International Conference on Machine Learning (ICML)},
  year          = {2026},
  eprint        = {2512.21094},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2512.21094},
}
```

## License

See [LICENSE](LICENSE).
