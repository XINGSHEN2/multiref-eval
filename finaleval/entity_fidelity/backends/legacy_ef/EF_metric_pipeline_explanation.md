# Entity Fidelity Metric Pipeline Explanation

This document summarizes how the EF metrics are computed in `finaleval/EF`, including the automated entity-fidelity branches, the MLLM-based metrics, how they are combined, and how human/object/background fidelity is measured for real and cartoon subjects.

## 1. Overall Structure

The EF pipeline contains two complementary parts:

1. Automated Entity Fidelity, which measures reference similarity through detection, cropping, visual embeddings, and temporal consistency.
2. MLLM-based evaluation, which uses a Gemini judge for detail preservation, binding, and paste artifact quality.

The usual inputs are CSV/XLSX rows with:

```text
board, seq, model, video_path
```

The pipeline resolves the corresponding sample metadata JSON, loads reference images for human/object/scene entities, samples video frames, computes branch scores, and writes per-video JSON plus summary CSV/ranking files.

Main entry points:

```text
EF/run_entity_fidelity_from_xlsx.sh
EF/run_mllm_detail_binding_from_xlsx.sh
EF/run_mllm_paste_artifact_from_xlsx.sh
EF/summary_py/generate_ef_paste_summary.py
```

## 2. Automated EF Pipeline

The automated EF pipeline is implemented mainly in:

```text
EF/eval_py/run_entity_fidelity_from_xlsx.py
EF/eval_py/run_entity_fidelity.py
EF/eval_py/multi_entity_fidelity.py
EF/eval_py/face_branch.py
EF/eval_py/body_branch.py
EF/eval_py/object_branch.py
EF/eval_py/scene_branch.py
```

For each video, the pipeline samples frames, computes frame-level similarity scores, aggregates them into branch-level scores, and then adds long-term consistency correction.

Default sampled frames:

```text
num_frames = 32
```

The main automated branches are:

```text
Human_EF
Face_EF
Body_EF
Object_EF
Scene_EF
```

After temporal correction, the summary contains:

```text
Human_EF_with_LTC
Object_EF_with_LTC
Scene_EF_with_LTC
EF_with_LTC
```

## 3. Human Fidelity

Human fidelity is split into real-human and cartoon-human logic.

### 3.1 Real Human

For real human references, the pipeline uses both face and body evidence.

Face branch:

1. Detect faces with `InsightFace FaceAnalysis`.
2. Extract face embeddings using ElasticFace R100.
3. The face recognizer used by `run_entity_fidelity_from_xlsx.py` is:

```text
elasticface_r100
```

with checkpoint:

```text
ef_eval/model_assets/face_recognizers/elasticface_r100.pth
```

4. Compute cosine similarity between the video-frame face embedding and reference face embeddings.
5. This produces frame-level `Face_EF_t`.

Body branch:

1. Detect the person/body region using YOLO-World.
2. Crop the detected body box from each sampled video frame.
3. Embed the crop with SigLIP for the final body score. DINOv2 is also computed as a diagnostic component, but it is not the selected overall body score in this pipeline.
4. Compare the SigLIP body embedding with the reference body embedding using cosine similarity.
5. This produces frame-level `Body_EF_t`; `Body_DINOv2_t` is kept only as a component/diagnostic output.

Final body/object/scene embedding model used for the main score:

```text
SigLIP: google/siglip-base-patch16-224 local snapshot
```

DINOv2 is loaded and written to component fields such as `Body_DINOv2`, `Object_DINOv2`, and `Scene_DINOv2`; the branch overall similarity used for EF is SigLIP.

For real humans, the frame-level human score is:

```text
EF_t = face_weight * Face_EF_t + (1 - face_weight) * Body_EF_t
```

In `run_entity_fidelity_from_xlsx.py`, the default is:

```text
face_weight = 0.6
```

If either face or body is unavailable, the available branch is used.

### 3.2 Cartoon Human

Cartoon subjects are handled differently because real face recognizers are unreliable for cartoon faces.

Cartoon head branch:

1. Detect cartoon/anime faces using the YOLOv8 anime-face detector.
2. Detector used:

```text
yolov8-anime-face
```

3. Crop the detected cartoon head region.
4. Embed the crop using the DINOv2 component of the image embedding branch.
5. Compare video cartoon-head embeddings with reference cartoon-head embeddings.
6. This produces `Cartoon_Head_EF_t`. In the current code this head score is the DINOv2 head similarity, not the SigLIP body score.

Cartoon body branch:

1. Detect the cartoon body/person region using YOLO-World.
2. Crop the body region.
3. Embed with SigLIP for the final body score. DINOv2 is still recorded as a diagnostic component.
4. Compare to reference body embeddings.
5. This produces `Body_EF_t`.

For cartoon references, low-quality body matches are filtered with:

```text
cartoon_min_body_ef = 0.6
```

If the body score is below this threshold, the frame is dropped. This prevents a small matching head from incorrectly producing a high human score when the full character does not match.

When both cartoon head and body are available:

```text
EF_t = face_weight * Cartoon_Head_EF_t + (1 - face_weight) * Body_EF_t
```

If no cartoon head is detected, the score falls back to body evidence only.

### 3.3 Multi-Human Samples

For samples with multiple human entities, the pipeline uses `MultiEntityFidelityEvaluator`.

Process:

1. Detect all candidate body boxes and face/head boxes in each sampled frame.
2. For each reference entity and each detection, compute face/head/body similarities.
3. Build candidate assignments.
4. Assign detections to reference entities greedily by match score.
5. Avoid assigning one detection to multiple entities.
6. Track identity switches across frames.
7. Compute per-entity EF, then aggregate across entities.

The assignment match score is:

```text
match_score = 0.4 * face_like_score + 0.6 * body_score
```

For missing entities, the code assigns a fallback score of `0.5` when aggregating entity scores.

## 4. Object Fidelity

Object fidelity is implemented in:

```text
EF/eval_py/object_branch.py
```

Object references come from sample JSON entries where:

```text
image_items[].type == "object"
```

Process:

1. Load the object reference image.
2. Build an object detection prompt from its description.
3. Use YOLO-World as an open-vocabulary detector with this text prompt.
4. Detect the object in each sampled video frame.
5. Filter invalid tiny boxes.
6. Crop the detected object box.
7. Embed the crop using SigLIP for the final object score. DINOv2 is also recorded as `Object_DINOv2`.
8. Compare the crop embedding to the reference object embedding.
9. Average valid frame scores to get `Object_EF`.

The object branch outputs:

```text
Object_EF
Object_DINOv2
Object_SigLIP
valid_object_ratio
```

If there are multiple object references, the final object branch score is the mean of valid object scores.

Object cropping is required for video frames. The reference object image is generally embedded directly without an additional mask.

## 5. Background / Scene Fidelity

Scene fidelity is implemented in:

```text
EF/eval_py/scene_branch.py
```

Scene references come from sample JSON entries where:

```text
image_items[].type == "scene"
```

Process:

1. Embed the reference scene image using SigLIP for the final scene score. DINOv2 is also recorded as `Scene_DINOv2`.
2. Embed each sampled video frame as a full image.
3. Compare frame embeddings with the reference scene embedding. The code uses `Scene_SigLIP_t` as `Scene_EF_t`.
4. Use the best matching frame as the scene score.

The aggregation is:

```text
Scene_EF = max(frame-level scene similarity)
```

The output also records:

```text
Scene_EF_mean
best_frame_index
Scene_DINOv2
Scene_SigLIP
```

Scene/background does not use cropping or masking. It is measured on the whole frame because the target is global environment similarity.

## 6. Branch Combination

For each video, branch scores are combined in `combine_video_branches`.

Base branch weights are:

```text
human: 0.50
object: 0.25
scene: 0.25
```

If a branch is missing, available branch weights are renormalized.

Example:

- human + object + scene: `0.50 / 0.25 / 0.25`
- human + scene only: human and scene are renormalized from `0.50 / 0.25`
- human only: final EF is human EF

The output records:

```text
branch_scores
branch_weights
Overall_EF
```

## 7. Long-Term Consistency

Long-term consistency is added by:

```text
EF/tools/add_long_term_consistency.py
```

It evaluates whether EF stays stable across sampled frames. It uses:

- mean score
- standard deviation
- minimum score
- early-frame mean
- late-frame mean
- late/early ratio
- collapse ratio
- normalized std

The LTC score is clamped to `[0, 1]` and then applied as:

```text
EF_with_LTC = mean_EF * (0.7 + 0.3 * LTC)
```

This penalizes videos where the subject starts faithful but later collapses, drifts, disappears, or changes identity.

The pipeline computes LTC for:

```text
Human_EF_with_LTC
Object_EF_with_LTC
Scene_EF_with_LTC
EF_with_LTC
```

## 8. MLLM Metrics

MLLM metrics live under:

```text
EF/mllm/
```

The main MLLM metrics are:

```text
Detail Preservation
Binding
Paste Artifact
```

Anatomical consistency has been moved to:

```text
finaleval/BQ/
```

### 8.1 Detail Preservation

Detail preservation checks whether fine-grained reference details are preserved in the generated video. It is judged by a Gemini MLLM prompt and outputs model-level and sample-level rankings.

This metric captures issues that embedding similarity may miss, such as missing accessories, wrong local attributes, or incomplete reference details.

### 8.2 Binding

Binding checks whether entities, attributes, and interactions are correctly bound. Typical errors include:

- attributes assigned to the wrong person
- object attached to the wrong subject
- clothing or identity mixed across characters
- reference object appearing but linked to the wrong actor

This is also judged by Gemini and reported as a separate MLLM metric.

### 8.3 Paste Artifact

Paste artifact is used as a correction to automated EF.

It checks whether the generated subject/object looks pasted, cut out, paper-like, poorly integrated, or inconsistent with the environment.

Typical judged artifacts:

- sticker-like subject
- cutout edges
- halo around boundaries
- sliding 2D layer
- poor contact shadows
- inconsistent lighting
- inconsistent perspective
- subject not physically integrated with the background

The judge outputs a 1-5 quality score and a penalty coefficient:

```text
5 -> 1.00
4 -> 0.95
3 -> 0.90
2 -> 0.80
1 -> 0.65
```

The penalty is applied to EF summaries as:

```text
Final_EF = Base_EF * paste_penalty
```

## 9. MLLM + Automated EF Combination

The strongest coupling between MLLM and automated EF is paste artifact penalty.

Detail preservation and binding are reported as independent MLLM metrics. They do not directly change the automated embedding EF score.

Paste artifact does modify EF by multiplying a base EF score:

```text
Mean(Human, Object, Background) * Paste Penalty
(0.5 Human + 0.5 Other) * Paste Penalty
(0.7 Body + 0.3 Other) * Paste Penalty
(0.7 Face + 0.3 Other) * Paste Penalty
```

where:

```text
Other = mean(Object_EF_with_LTC, Scene_EF_with_LTC)
```

The summary script computes per-row metrics, averages within each board by model, then averages over available boards.

## 10. Common Ranking Formulas

The board-level ranking script uses:

```text
Other = mean(Object_EF_with_LTC, Scene_EF_with_LTC)
Board total = 0.7 * Human_EF_with_LTC + 0.3 * Other
```

If no object/background branch exists, board total falls back to human.

The final EF + paste summary can also report:

```text
mean_human_object_background
human_0p5_other_0p5
face_plus_other
body_plus_other
object_only
background_only
```

and paste-adjusted versions of the first four.

## 11. Models Used

The automated EF pipeline uses these model families:

| Purpose | Model / Method |
|---|---|
| Real face detection | InsightFace FaceAnalysis |
| Real face recognition | ElasticFace R100 |
| Body / object / scene final embedding | SigLIP |
| Body / object / scene diagnostic component | DINOv2 |
| Person/body detection | YOLO-World |
| Object detection | YOLO-World with object text prompt |
| Cartoon face/head detection | YOLOv8 anime face |
| MLLM judge | Gemini 3.1 Pro Preview (`gemini-3.1-pro-preview`) |

## 12. Cropping and Masking

Cropping/masking is used selectively.

Required cropping:

- real face crop/alignment through InsightFace
- real body crop from YOLO-World person box
- cartoon head crop from YOLOv8 anime-face detector
- cartoon body crop from YOLO-World person/body box
- object crop from YOLO-World object box

Reference-side masking:

- real human reference body uses precomputed SAM masks when the corresponding mask file exists:

```text
datasets/human_masks_sam3/masks
```

- if a mask exists, the reference human image is masked before SigLIP body embedding
- if no mask exists, the full reference image is used

No cropping/masking:

- scene/background branch uses full-frame embeddings
- MLLM detail/binding/paste artifact judge uses video/manifest-level inputs rather than embedding crops

## 13. Practical Interpretation

Automated EF measures visual similarity to reference images through model embeddings. It is strong for identity, body appearance, object appearance, and scene similarity, but it can over-score pasted or physically implausible subjects.

MLLM paste artifact compensates for this by judging whether the subject/object is visually integrated into the scene. Detail preservation and binding provide additional semantic checks that are not captured by cosine similarity alone.

In short:

```text
Automated EF = reference similarity
LTC = temporal stability
Paste Artifact = visual integration penalty
Detail / Binding = MLLM semantic fidelity diagnostics
```
