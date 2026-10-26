# Audio Task Following MLLM Judge

You are evaluating a generated reference-based video's Audio Task Following.

Use only the generated video/audio and the provided IF spec. Do not reward content merely because it appears in the prompt. Do not penalize audio quality unless it prevents detecting whether a requested sound is present.

Important scoring rules:

- Each checklist item receives an integer score from 1 to 5.
- The dimension-level `score` must be the arithmetic mean of all applicable item scores and may be a decimal. Do not assign it independently.
- Audio Task Following evaluates only explicitly requested non-speech audio: sound effects, ambience, environmental sounds, foley, object sounds, and background music.
- Speech Content Accuracy is handled separately. Ignore speech, dialogue, narration, voiceover, spoken text, speaker identity, and lip sync.
- Use broad semantic matching. A sound does not need an exact sound-library match; judge whether it plausibly matches the requested scene, action, or music requirement.
- Give score 5 when the requested non-speech sound is clearly audible and broadly matches the requirement.Do not reduce the score merely because a matching sound is faint, brief, partially masked, noisy, or not prominent. If it is still identifiable, give score 5.
- Give score 1 when the requested sound is clearly absent or clearly does not match the requirement.
- Use scores 2 to 4 only when evidence is genuinely partial or uncertain, or when only part of the requested audio requirement is present.
- Do not penalize extra or unrequested audio. Unrelated speech, dialogue, music, ambience, or sound effects must not lower the score.
- The absence of sound effects should not be assumed merely because background music and sound effects are played simultaneously
- Do not evaluate visual task completion or temporal ordering in this dimension.

Item score scale:

- 1: The requested sound is clearly absent or clearly incorrect.
- 2: There is weak evidence of the requested sound, but most of the requirement is not satisfied.
- 3: The requested sound is partially present or genuinely ambiguous.
- 4: The requested sound is mostly present and recognizable, with a minor substantive mismatch.
- 5: The requested sound is clearly audible and broadly satisfies the requirement.

Evaluate this IF spec:

```json
{if_eval_spec_json}
```

Return only one valid JSON object with this structure. The values below are type descriptions, not literal values to copy:

```text
{
  "Audio Task Following": {
    "applicable": boolean,
    "score": number_or_null,
    "reason": string,
    "item_results": [
      {"question": string, "score": integer_1_to_5_or_null, "reason": string}
    ]
  }
}
```

Evaluation procedure:

1. Read only the `audio_task_following` checklist in the IF spec.
2. Produce one `item_results` entry for every applicable explicitly requested non-speech audio item.
3. Ignore speech-related checklist items if they appear by mistake, and do not let them affect the score.
4. Assign each applicable item an integer score from 1 to 5 using the item score scale above.
5. Set the dimension-level `score` to the arithmetic mean of all applicable item scores.
6. The dimension reason must summarize how the item scores support the dimension score.
7. If the dimension is not applicable according to `applicable_dimensions`, set `"applicable": false`, `"score": null`, and explain briefly.
8. Return only the final JSON object.
