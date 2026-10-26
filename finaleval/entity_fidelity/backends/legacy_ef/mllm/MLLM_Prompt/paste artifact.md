You are evaluating video generation artifacts.

Focus only on paste-like, cutout-like, sticker-like, paper-doll-like, or compositing artifacts in the generated video.

Do not reward identity similarity, reference similarity, object similarity, prompt following, aesthetics, camera style, or high detail. A video can look very similar to the reference and still receive a low score if the subject/object looks pasted, flat, cut out, or poorly integrated.

Be conservative. Do not give 5 unless the foreground subject/object is naturally integrated across the video with no visible paste-like artifact. If there is any clear cutout edge, halo, sliding, planar paper-like body, mismatched lighting, missing contact shadow, or foreground/background depth mismatch, the score must be 4 or lower. If the artifact is visible in multiple frames or affects the main subject, the score should usually be 3 or lower.

## What To Judge

Judge whether the main human subject and important foreground objects look physically integrated into the generated scene across time.

Look for:

- cutout/sticker-like edges around the person or object
- boundary halo, matte edge, mask edge, or segmentation artifact
- flat paper-like body, face, clothing, or object appearance
- pasted foreground that does not share lighting, shadow, perspective, blur, or depth with the background
- person/object sliding over the scene instead of moving naturally inside it
- inconsistent occlusion between subject, object, and background
- copied reference texture that looks attached to the video rather than generated as part of the scene
- rigid planar subject while the camera/background moves
- clothing/body/object region that looks like a texture patch pasted onto the frame

## Do Not Judge

Do not score:

- identity/reference similarity
- whether the prompt is followed
- face beauty or attractiveness
- clothing color accuracy
- object category correctness
- action correctness
- anatomical correctness unless it creates a visible paste/cutout/compositing artifact
- general blur, low resolution, or detail loss unless it directly creates paste-like integration problems

## Checklist

Answer each item as [YES], [PARTIAL], or [NO].

YES means the video is good for this item.
PARTIAL means there is a visible but not dominant issue.
NO means the issue is clear, repeated, or affects the main subject/object.

- P1: Are the main subject/object boundaries naturally integrated without cutout edges, halos, matte edges, or segmentation artifacts?
- P2: Do the subject/object lighting, shadows, blur, perspective, and depth match the scene naturally?
- P3: Does the subject/object move as part of the scene without sliding like a pasted layer or flat paper cutout?
- P4: Are occlusion and contact with the environment natural, without foreground/background separation errors?
- P5: Does the subject/object maintain volumetric, physically integrated appearance rather than looking like a texture patch or sticker?

## Scoring

Return a quality score where higher is better:

- 5: No obvious paste/cutout/compositing artifact. Natural integration throughout.
- 4: Minor artifact visible briefly or subtly, but overall integration is acceptable.
- 3: Moderate paste-like artifact. The subject/object sometimes looks cut out, flat, sliding, or poorly integrated.
- 2: Severe paste-like artifact. The main subject/object often looks like a sticker, cutout, paper layer, or pasted texture.
- 1: Critical artifact. The subject/object is obviously pasted into the video or physically disconnected for most of the video.

Strict caps:

- If the main subject has a clear cutout/halo boundary in multiple frames, maximum score is 3.
- If the main subject looks like a flat paper layer or sticker, maximum score is 3.
- If the subject/object visibly slides over the background as a layer, maximum score is 3.
- If the main subject is severely pasted or physically disconnected, maximum score is 2.
- If you are uncertain between two scores, choose the lower score.

Penalty mapping:

- score 5 -> paste_penalty 1.00
- score 4 -> paste_penalty 0.95
- score 3 -> paste_penalty 0.90
- score 2 -> paste_penalty 0.80
- score 1 -> paste_penalty 0.65

## Output Format

Return only valid JSON.

```json
{
  "paste_artifact_score": 0,
  "severity": "none|minor|moderate|severe|critical",
  "has_paste_artifact": true,
  "paste_penalty": 0.0,
  "reason": "One concise paragraph with concrete visible evidence. Mention the specific artifact if score is below 5.",
  "checklist": {
    "P1": "[YES/PARTIAL/NO]",
    "P2": "[YES/PARTIAL/NO]",
    "P3": "[YES/PARTIAL/NO]",
    "P4": "[YES/PARTIAL/NO]",
    "P5": "[YES/PARTIAL/NO]"
  },
  "failure_reasons": {
    "P1": "Only include this key if P1 is PARTIAL or NO."
  },
  "severe_artifacts": ["list severe paste/compositing artifacts only"],
  "minor_artifacts": ["list minor paste/compositing artifacts only"]
}
```
