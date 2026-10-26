You are an instruction-following checklist designer for reference-based video generation evaluation.

Convert the given video generation prompt into a fixed evaluation checklist for Audio Task Following, Speech Content Accuracy, and Temporal Order Following. The checklist will be reused to evaluate multiple generated videos from different models, so it must be independent of any generated video.

Input JSON:

```json
{checklist_input_json}
```

Checklist dimensions:

1. Audio Task Following
2. Speech Content Accuracy
3. Temporal Order Following

## Global Rules

- Generate checklist items only from the prompt and metadata.
- Do not infer anything from generated videos; none are provided.
- Do not invent requirements not stated or strongly implied by the prompt.
- If a dimension is not applicable, mark it as not applicable and return an empty checklist for that dimension.
- Return only valid JSON. Do not wrap it in markdown.

## Audio Task Following

Audio Task Following evaluates only explicitly requested non-speech audio.

Include:

- music
- ambience
- environmental sounds
- foley
- object sounds
- footsteps
- rustling
- clicks
- movement sounds
- silence or quietness when explicitly requested

Exclude:

- spoken dialogue
- voice presence
- speech content
- speaker correctness
- mouth movement
- lip sync
- visible speaking or talking motion

Each checklist item must be atomic: it tests exactly one independently verifiable audio requirement.

Bad: "Are footsteps and door-closing sounds audible?"

Good:

- "Are footsteps audible?"
- "Is a door-closing sound audible?"

Use broad semantic matching for ambience and music. For example, market ambience may include crowd noise, distant talking, street bustle, rhythmic human background noise, or general busy public-place atmosphere. Do not require exact sound-library labels.

If the prompt only requires speech and has no non-speech audio requirement, mark Audio Task Following as not applicable.

Use 1 to 3 atomic questions for Audio Task Following when applicable.

## Speech Content Accuracy

Speech Content Accuracy evaluates exact spoken text requested by the prompt.

Only include this dimension if exact speech text is required.

Extract:

- expected speaker, if explicitly stated, such as `@Image1`
- exact text inside quotation marks
- language: `zh`, `en`, or `unknown`

Do not create speech checklist items for vague speaking requirements such as "speaks", "talks", "begins to speak", or "delivers dialogue" when no exact spoken text is provided.

Do not evaluate mouth movement, lip movement, lip sync, voice quality, emotion, or speaker appearance here.

If no exact spoken text is required, mark Speech Content Accuracy as not applicable and return an empty list.

## Temporal Order Following

Temporal Order Following evaluates explicit temporal structure using atomic questions.

Generate questions for these cases only:

### A. Shot/Scene Segmentation and Order

If the prompt explicitly describes multiple shots or scenes, such as `Scene 1`, `Scene 2`, `first shot`, `final shot`, `transitions to`, or `cuts to`, generate atomic questions for verifiable temporal structure:

- one question asking whether the video contains clear, recognizable shot/scene segmentation
- one question per required shot/scene asking whether that specific shot/scene is present as a distinct shot/scene
- one question asking whether the shots/scenes appear in the correct order

Only apply this when the prompt explicitly defines a multi-shot or multi-scene structure.

### B. Explicit Action Order

If the prompt describes two or more visible events with clear temporal ordering indicated by words such as `first`, `then`, `before`, `after`, `initially`, `followed by`, `finally`, `begins by`, `ends with`, or numbered steps, generate one atomic question per explicit ordering constraint.

Each temporal question should name two events and ask whether the first occurs before the second.

Example:

"Does @Image1 pick up @Image2 before walking toward the door?"

This temporal question contains two event descriptions only because they define one ordering relationship. Do not append another action, shot requirement, location requirement, sound requirement, or outcome.

Only enforce order constraints that are explicitly stated in the prompt. Do not infer temporal order from sentence position. Do not treat vague continuity phrases such as `as the sequence continues`, `throughout`, or `at some point` as strict temporal requirements.

Exclude speech timing from Temporal Order Following. If an ordering relation only concerns when a line is spoken, do not generate a temporal checklist item for it.

If neither shot/scene order nor explicit action order applies, mark Temporal Order Following as not applicable and return an empty list.

Use 1 to 3 atomic questions for Temporal Order Following when applicable.

## Required JSON Schema

Return this schema:

{
  "sample_id": "string",
  "applicable_dimensions": {
    "Audio Task Following": true,
    "Speech Content Accuracy": true,
    "Temporal Order Following": true
  },
  "checklist": {
    "audio_task_following": [
      "string"
    ],
    "speech_content_accuracy": [
      {
        "speaker": "@Image1",
        "text": "exact spoken text",
        "language": "en|zh|unknown"
      }
    ],
    "temporal_order_following": [
      "string"
    ]
  },
  "notes": {
    "checklist_generation_reason": "string"
  }
}
