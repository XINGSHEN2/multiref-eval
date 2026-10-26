# Voice Timbre Similarity Judge

Judge only whether the audible speech voice in the generated video has a similar timbre to the input/reference audio.

Use the reference audio as the target voice. Use the generated video's audible speech as the candidate voice.

Focus only on broad voice timbre characteristics:

- vocal gender presentation
- apparent age range
- pitch range
- brightness or darkness
- thinness or thickness
- breathy, nasal, hoarse, clear, rough, soft, or resonant qualities
- overall speaker-like voice texture

Do not judge transcript correctness, language correctness, lip sync, mouth timing, visual identity, source correctness, event sounds, music, or background noise except when they make the voice impossible to hear.

## Applicability

Score only when both the reference audio and the generated video contain audible speech or voice that is clear enough to compare.

Set `applicable: false` and `score: null` when:

- the reference audio is missing, unreadable, silent, or has no speech/voice;
- the generated video has no audible speech/voice;
- either voice is too short, too noisy, too overlapped, or too unclear to compare;
- multiple generated speakers overlap and the comparable voice cannot be isolated;
- the generated audio is only music, environment sound, or event sound.

Language, wording, accent, emotion, speed, and recording quality may differ. Ignore those differences unless they prevent timbre comparison.

## Scoring

Use only these scores:

- 1: different timbre. The generated speech clearly sounds like a different voice from the reference audio.
- 3: uncertain or partially similar. The voices share only coarse traits, or evidence is mixed/limited.
- 5: similar timbre. The generated speech clearly preserves the broad voice/timbre characteristics of the reference audio.

Do not use scores 2 or 4.

## Output

Return exactly one `voice_results` item unless the task is not applicable. Set the dimension-level `score` to the item score.

Evaluate this JSON:

```json
{ac_eval_spec_json}
```

Return only one valid JSON object:

```json
{
  "dimension": "Voice Timbre Similarity",
  "applicable": true,
  "score": 5,
  "reason": "Brief evidence comparing only the generated speech timbre to the reference audio timbre.",
  "voice_results": [
    {
      "voice_id": "speech_voice",
      "reference_voice_description": "Brief description of the reference voice.",
      "generated_voice_description": "Brief description of the generated video speech voice.",
      "voice_similarity": "similar|partial_or_uncertain|different|not_inferable",
      "score": 5,
      "reason": "Brief timbre-only comparison."
    }
  ],
  "voice_errors": []
}
```
