# Metrics

MultiRef-Compass exposes 14 main metrics. The open-source package provides backend
entrypoints and uniform output files for every metric. Automatic metrics need
their model weights and media dependencies; MLLM metrics additionally need
Gemini/Vertex credentials. Backend failures are reported as `status=error`.

## Basic Quality

- Visual Quality
- Audio Quality
- Anatomical Consistency

## Entity Fidelity

- Entity Fidelity
- Detail Preservation
- Detail Binding

Paste Artifact is an auxiliary penalty multiplied into the final Entity
Fidelity score.

## Audio-Video Consistency

- Speech-Lip Synchronization / VSLS
- Visual Event-Sound Alignment
- Source Correctness
- Voice Timbre Similarity

Stable Speaking Frontal Face is an optional filter before VSLS.

Visual Event-Sound Alignment evaluates audible non-speech sounds in the
generated video against their visible sound sources. The text prompt may be
used as context during task construction, but the metric should not score
prompt following or penalize missing prompt-expected Foley.

## Instruction Following

- Visual Task Following
- Audio Task Following
- Temporal Order Following
- Speech Content Accuracy

Visual Task Following may report category breakdowns in its ranking, including
Important Action, Wearing Requirement, Basic Requirement, and Other Detail, but
it remains one main metric.
