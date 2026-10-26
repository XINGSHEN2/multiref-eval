# Stable Speaking Frontal Face Judge

Judge whether the generated video is suitable for automatic speech-lip / VSLS evaluation.

Use only the generated video and its audio as evidence. Focus on the time span when speech or voice is audible. The item should pass only when the visible speaking person is consistently shown in a half-body / upper-body / bust-style framing or closer, with a stable frontal or near-frontal face while speaking.

## Pass Condition

Set `stable_speaking_frontal_face: true` only if all conditions are met:

- audible speech or voice is present;
- the speaker is visible during the speech;
- the speaker is framed as half-body / upper-body / bust-style or closer during speech, not as a tiny full-body or distant figure;
- the speaker's face is frontal or near-frontal for most of the speaking duration;
- the mouth/lower face area is visible enough to support lip-sync evaluation;
- the face is not too small, heavily blurred, heavily occluded, cropped out, turned away, or shown mainly in profile;
- the shot is stable enough that the face can be tracked without frequent cuts, large camera motion, fast head turns, or severe scale changes;
- if multiple people are visible, the speaking person can be identified reliably.

## Fail / Not Applicable

Set `stable_speaking_frontal_face: false` when:

- there is no audible speech;
- the speech is voiceover-like, off-screen, narrated, or not bound to a visible speaker;
- the speaker is mainly side-facing, back-facing, too small, distant full-body, occluded, blurred, cropped, or outside the frame;
- the video cuts away from the face during speech;
- the mouth is not visible enough for lip-sync evaluation;
- multiple visible people could be speaking and the speaker cannot be identified reliably.

Use `applicable: false` when there is no audible speech or no visible speaking person. Otherwise use `applicable: true`.

## Scoring

- 5: clear pass; half-body/upper-body or closer framing with stable frontal or near-frontal speaking face for most speech.
- 4: mostly pass; minor pose/motion/framing issue but still suitable.
- 3: borderline; sometimes usable but not stable enough for strict automatic evaluation.
- 2: mostly unsuitable; face is often not frontal/stable/visible.
- 1: clearly unsuitable.

For CSV filtering, only scores `>= 4` with `stable_speaking_frontal_face: true` should be kept.

Evaluate this JSON:

```json
{ac_eval_spec_json}
```

Return only one valid JSON object:

```json
{
  "dimension": "Stable Speaking Frontal Face",
  "applicable": true,
  "score": 5,
  "stable_speaking_frontal_face": true,
  "speaker_visible": true,
  "speaker_identifiable": true,
  "frontal_or_near_frontal": true,
  "mouth_visible": true,
  "half_body_or_closer": true,
  "stable_track": true,
  "speech_time_observed": "brief description of when speech is heard",
  "observed_speaker": "brief description of the visible speaker, or empty string",
  "failure_modes": [],
  "reason": "Brief evidence from the generated video/audio."
}
```

Do not return keys other than those shown above. Use JSON booleans and numbers, not strings, for boolean and score fields.
