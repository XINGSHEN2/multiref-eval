# Source Correctness Judge

Judge only whether audible speech/voice can be reliably attributed to the expected visible speaker.

Do not judge transcript wording, language, lip-sync quality, mouth timing, speech timing, or reference identity similarity. Mouth/lip/jaw movement may be used only as positive evidence for identifying who is speaking.

## Applicability

Score an item only when the speech source can be reliably bound to a visible speaker.

Set `source_match: "not_applicable"` and `score: null` when:

- speech is off-screen, voiceover-like, narrated, or not visibly bound to a speaker;
- no speech/voice is audible;
- the relevant speaker/source is too small, occluded, blurred, cropped, shown from behind, hidden by a cutaway, or otherwise unreliable;
- multiple visible people could plausibly be speaking and no specific speaking source can be confirmed;
- the only evidence against the expected speaker is missing/unclear mouth movement, static mouth, poor lip sync, or timing mismatch.

Exception: if clear visible mouth/lip/jaw movement reliably identifies who is producing the audible speech, score the item normally even if the view is profile, partially occluded, or otherwise limited.

## Scoring

- 1: clear positive evidence that a different visible speaker is producing the speech.
- 2: do not use for ordinary source uncertainty; use `not_applicable` instead.
- 3: expected speaker is reliably the source, but the voice gender clearly mismatches the expected speaker's gender presentation.
- 4: do not use for ordinary source uncertainty; use `not_applicable` instead.
- 5: expected speaker is reliably the source, with no clear voice-gender mismatch.

Do not penalize age, identity, role, accent, language, nationality, race, ethnicity, regional background, style, missing words, changed words, translated words, or incomplete speech.

Do not infer `wrong_person` merely because the expected speaker's mouth does not move, lip sync is poor, or the speech sounds like a voiceover. Use `wrong_person` only when another visible speaker is clearly producing the speech.

## Output

Return exactly one `source_results` item for each checklist item. Set the dimension-level `score` to the mean of non-null item scores. If every item is `not_applicable`, set top-level `"applicable": false` and `"score": null`.

Evaluate this JSON:

```json
{ac_eval_spec_json}
```

Return only one valid JSON object:

```json
{
  "dimension": "Source Correctness",
  "applicable": true,
  "score": null,
  "reason": "Brief evidence from the generated video/audio.",
  "source_results": [
    {
      "source_id": "speech_1",
      "expected_speaker": "@Image1",
      "expected_speaker_reference": {
        "placeholder": "@Image1",
        "type": "human",
        "description": "reference image description"
      },
      "audible_sound_or_speech": "brief description of audible speech/voice, or empty string",
      "observed_source": "who appears to produce the speech",
      "source_match": "correct|wrong_person|ambiguous|not_applicable",
      "timbre_match": "matched|mismatched|not_inferable|unclear",
      "score": null,
      "reason": "Brief evidence for this item."
    }
  ],
  "source_errors": []
}
```
