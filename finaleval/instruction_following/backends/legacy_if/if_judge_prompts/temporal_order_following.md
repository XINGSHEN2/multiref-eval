# Temporal Order Following MLLM Judge

You are evaluating a generated reference-based video for **Temporal Order Following**.

Use only the generated video and the provided IF spec. Do not reward content merely because it appears in the prompt. Do not penalize visual quality unless it prevents comparing visible events, shots, or scenes.

This dimension evaluates only whether the generated video follows the requested **temporal order, shot progression, scene progression, and required multi-shot structure**.

Do **not** evaluate:

* speech content;
* audio presence or audio correctness;
* visual resemblance or identity similarity;
* general visual task completion;
* reference consistency.

Visual content is considered here only when it is needed to determine event order, shot order, or whether a required shot/scene exists as part of a temporal structure.

## Item Type Classification
### Type A: Order-comparison item
Asks whether one event/action/shot occurs before or after another. Evaluate only if both events are visible and assessable. If either is missing or unclear, mark `applicable: false`, `score: null` — a missing event is not a reversal or misordering.

### Type B: Required-shot or required-scene presence item
Asks whether a required shot or scene exists as a recognizable distinct structural segment in sequence. If clearly absent or only scattered content without independent segmentation, keep `applicable: true` and assign score **1**. Ordinary missing content should not be penalized here unless the shot/scene is defined as temporal structure requirement.

### Type C: Shot/scene segmentation item
Asks whether the video has a recognizable multi-shot or multi-scene structure and valid transitions.
A single continuous shot should not receive severe penalty unless multi-shot structure is an explicit core requirement.

## Score Scale — Type A (Order Comparison)
- 1: Both events visible, clearly reversed order.
- 2: Both events visible, substantial violation of requested order (dominant sequence wrong, not clean full reversal).
- 3: Both assessable, but heavy overlap/repetition/unclear boundaries; cannot confirm dominant order.
- 4: Correct dominant order, only brief displacement or minor incidental overlap/out-of-order occurrence.
- 5: Requested pairwise order unambiguous and fully preserved.

## Score Scale — Type B & C (Shot/Scene Structure)
- 1: Required shot/scene absent, or content appears but never forms distinct requested segment.
- 2: Only weak, fragmentary attempt of target shot/scene.
- 3: Partially recognizable but framing, boundaries, or placement ambiguous/incomplete.
- 4: Clearly present and distinct, with one minor framing/position boundary issue.
- 5: Clearly present as recognizable independent structural segment matching requirements.

## General Rules
- Dimension-level score = arithmetic mean of applicable and assessable item scores, do not manually round based on overall impression.
- Each item reason must cite concrete visual evidence (events order, shot boundaries) instead of vague summaries.
- Do not double penalize content omissions already covered by Visual Task Following, unless independent shot segmentation is required by temporal checklist.

## Evaluate this IF spec

```json
{if_eval_spec_json}
```

---

## Output format

Return only one valid JSON object with this structure. The values below are type descriptions, not literal values to copy.

```text
{
  "Temporal Order Following": {
    "applicable": boolean,
    "score": number_or_null,
    "reason": string,
    "item_results": [
      {
        "question": string,
        "applicable": boolean,
        "score": integer_1_to_5_or_null,
        "reason": string
      }
    ]
  }
}
```

---

## Evaluation procedure

1. Read only the `temporal_order_following` checklist in the IF spec.
2. Check whether Temporal Order Following is applicable according to `applicable_dimensions`.
3. Produce one `item_results` entry for every temporal checklist item.
4. Ignore speech/dialogue timing items if they appear by mistake; mark them not applicable.
5. Classify each item as Type A, Type B, or Type C.
6. For Type A order-comparison items, compare order only if both events are visible and assessable. If either event is missing or unclear, set `"applicable": false` and `"score": null`.
7. For Type B required-shot/scene presence items, always evaluate whether the requested distinct shot/scene exists. If clearly absent or non-distinct, keep `"applicable": true` and assign score 1.
8. For Type C segmentation items, evaluate whether the requested temporal structure is recognizable and distinct.
9. Use the appropriate score scale for the item type.
10. Set the dimension-level `score` to the arithmetic mean of applicable and assessable item scores. Do not round or adjust it based on an overall impression.
11. If no temporal items are assessable, set `"applicable": false`, `"score": null`, and explain briefly.
12. Return only the final JSON object.
