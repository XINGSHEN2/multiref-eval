# Visual Task Following MLLM Judge

You are evaluating a generated reference-based video for **Visual Task Following**.

Use only the generated video and the provided IF spec. Do not reward content merely because it appears in the prompt. Do not judge visual quality unless it prevents judging whether the instruction was followed.

This dimension only evaluates whether the requested **visible subjects, objects, scenes, actions, interactions, spatial relations, states, or framings** are present and used as instructed.

Do **not** evaluate reference identity similarity, face similarity, clothing fidelity, texture fidelity, source-reference consistency, speech content, non-speech audio, or event order. Event order belongs to a separate temporal dimension.

## Evidence-first scoring

For each checklist item, do **not** decide the score first.

First identify the visible evidence:

1. What requested subject, object, scene, action, relation, state, or framing is visible?
2. What part of the requirement is satisfied?
3. What part is missing, unclear, incomplete, or contradicted?
4. Which evidence level best matches the video: no evidence, weak related evidence, partial evidence, mostly complete evidence, or complete direct evidence?

Only after this evidence analysis, assign the item score using the evidence gate below.

Do not choose a score first and then justify it. The score must be derived from visible evidence.

## Scoring rules

Score each checklist item independently. Each item score must be an integer from **1 to 5**. The dimension-level `score` must be the arithmetic mean of applicable item scores.

Use this evidence gate:

* **1 = No evidence**: the requirement is completely absent, clearly wrong, or contradicted.
* **2 = Weak related evidence**: something relevant is visible, but the core requirement is mostly missing or not completed.
* **3 = Partial evidence**: a meaningful part is visible, but an important part is missing, incomplete, ambiguous, or incorrect.
* **4 = Mostly complete evidence**: the core requirement is clearly completed, but some meaningful secondary detail is missing, unclear, or slightly wrong.
* **5 = Complete direct evidence**: the full atomic requirement is clearly, directly, and unambiguously shown.

Use **1** only when there is no related visible evidence at all. If any related subject, object, action, state, relation, attempt, or final result is visible, use **2 or 3**, not 1.

Use **5** only when the full requirement is directly and clearly visible. If the evidence is implied, brief, partially occluded, ambiguous, or missing any meaningful detail, use **4 or lower**, not 5.

Most mixed, partial, ambiguous, implied, or briefly shown cases should receive **2, 3, or 4**, not 1 or 5.

## Leniency rules

Do not over-penalize tiny details unless they are core requirements. Be lenient with small hand/finger motions, subtle expressions, exact gaze, precise pose angle, minor color/material/texture differences, prop styling, and incidental background details.

For hold, carry, use, wear, or interact-with actions, prioritize whether the subject visibly performs the intended functional role. Do not penalize fine-grained object category or material differences if the visible object functions as the requested prop.

For action completion, partial motion, visible action trend, plausible intermediate state, final state implying the action, brief contact, reaching, leaning, placing, adjusting, or positioning can support **4**. 

Do not bind actions to shot labels. If the requested visual action or state appears clearly in any shot, give visual credit. Shot order and event order should not be judged here.

## Reason requirements

Each item reason must mention:

1. the visible evidence;
2. what is satisfied;
3. what is missing, unclear, incomplete, or wrong;
4. why the selected score follows from the evidence.

For score **1**, explain why there is no related visible evidence instead of an adjacent score.
For score **5**, explain why the full requirement is directly and completely visible instead of an adjacent score.
For scores **2/3/4**, explain why the evidence fits that intermediate level instead of an adjacent score.

Do not use generic reasons such as “partially correct”, “mostly accurate”, or “some details are missing” without concrete visible evidence.

## Evaluate this IF spec

```json
{if_eval_spec_json}
```

## Output format

Return only one valid JSON object. Do not change the JSON structure.

```text
{
  "Visual Task Following": {
    "applicable": boolean,
    "score": number_or_null,
    "reason": string,
    "item_results": [
      {
        "question": string,
        "score": integer_1_to_5_or_null,
        "reason": string
      }
    ]
  }
}
```

## Procedure

1. Read only the `visual_task_following` checklist.
2. Check applicability using `applicable_dimensions`.
3. For every applicable visual checklist item, first analyze visible evidence.
4. Assign each item score using the evidence gate.
5. Score every item independently.
6. Compute the dimension-level `score` as the arithmetic mean of applicable item scores. Do not round or adjust it to match an overall impression.
7. If not applicable, return `"applicable": false`, `"score": null`, and a brief reason.
8. Return only the final JSON object.
