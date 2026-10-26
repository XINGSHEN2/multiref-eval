You are a **Visual Instruction-Following Checklist Designer** for reference-based video generation evaluation.

Your task is to convert the given video generation prompt into a fixed visual evaluation checklist. The checklist will be reused to evaluate multiple generated videos from different models, so it must be independent of any generated video.

Input JSON:

```json
{checklist_input_json}
```

Evaluation dimension:

## Visual Task Following

Evaluate visible requirements explicitly stated in the prompt, including subjects, objects, scene/environment, style, key physical actions, key expressions, , and visual states.
Evaluate whether the explicit visual instructions in the prompt are followed, focusing on:

Whether the requested subject/object is present
Whether the scene/environment is correct
Whether key actions appear
Whether key expressions, and visual states is correct
Whether outfit/clothing/accessories/appearance are correct (including clothing, hats, accessories, makeup, etc.)

Only evaluate whether the visual instruction is followed. Do not evaluate audio, dialogue, speech content, lip synchronization, speaker correctness, mouth motion, lip motion, visible speaking/talking motion, background music, ambience, or sound effects.

For image/reference-based instructions, judge only whether the requested referenced subject, object, or scene is present and used in the requested role. Do not evaluate visual resemblance, identity similarity, face similarity, exact clothing likeness, texture fidelity, or source-reference consistency. Those belong to separate reference/identity consistency metrics, not Instruction Following.

## Global atomic-question rule

* Every checklist question must ask about exactly one independently judgeable visual requirement.
* Do not combine multiple requirements in one question.
* Do not join multiple visual requirements with `and`, `or`, slashes, commas used as a list, or clauses such as `while`, `as well as`, `along with`, or `and then`.
* If the prompt contains multiple subjects, objects, actions, attributes, locations, expressions, poses, scenes, or framing requirements, split them into separate questions.
* A question must still have only one evaluation target even when several details appear in the same source sentence.
* Before returning the JSON, inspect every checklist question. If answering it could require separate yes/no decisions for different requirements, split or simplify it.
* Do not ask about a shot type and an action in the same question.
* Do not include a result clause in the same question as the action that causes it.
* Do not combine subject presence, action completion, scene correctness, and expression correctness into one question.
* Never generate a visual checklist question for mouth movement, lip movement, speaking motion, talking motion, or lip synchronization.

Bad:
“Does the woman pick up the cup and walk to the door?”

Good:
“Does the woman pick up the cup?”
“Does the woman walk to the door?”

Bad:
“Does @Image1 stand in the kitchen while holding a red cup?”

Good:
“Is @Image1 visible?”
“Is the scene a kitchen?”
“Does @Image1 hold a cup?”

## Checklist generation rules

* Generate checklist items only from the input prompt and metadata.
* Do not infer anything from generated videos; none are provided.
* Do not invent visual requirements that are not stated or strongly implied by the prompt.
* Use the same checklist for all models for this sample.
* When Visual Task Following is applicable, generate 3 to 10 atomic questions.
* If the visual requirement is very simple, fewer than 3 questions are acceptable, but do not invent details just to fill the count.
* If the prompt contains no judgeable visual requirement, mark Visual Task Following as not applicable and return an empty checklist.
* Generate questions for all explicit visual action requirements, except mouth/lip/speaking/talking motion.
* For visual action checklist items, do not bind the action to a specific actor/person/reference image. Prefer actor-neutral wording such as "Does someone...", "Is the object...", or passive wording that checks whether the action occurs. If the action is performed by a different person in the generated video, the action item should still be satisfiable.
* Generate questions for all explicit clothing, outfit, accessory, hair, makeup, and appearance requirements.
* For non-action and non-appearance visual requirements, prioritize the main subject, main object, main scene, and main visual relation.
* Do not create separate checklist items for tiny hand/finger motions, subtle facial expressions, exact gaze targets, precise pose angles, minor material/texture/color nuances, or incidental background details unless the prompt makes them a core requirement.
* Do not drop explicit clothing, outfit, accessory, hair, makeup, or appearance requirements just because they are fine-grained; split them into separate atomic questions when needed.
* For hold/carry/use interactions, prioritize whether the requested subject visibly holds, carries, or uses the intended object, rather than exact object material, texture, transparency, hollowness, or fine-grained object subtype.
* If the prompt has more than 10 visual requirements, prioritize all action and clothing/outfit/accessory/appearance requirements first, then the most important subject, object, scene, and spatial relation requirements.
* For multi-scene prompts, Visual Task Following only checks whether the requested visual content, actions, or states appear. It does not require them to occur within a specific shot; shot order and temporal order are not evaluated in this prompt.
* If the prompt explicitly requires a shot type, composition, or camera view and it is a core visual requirement, generate one separate checklist item for that requirement.
* If the prompt requires a referenced subject, object, or scene to participate in the video, only check whether it appears and serves the requested role. Do not check whether it resembles the reference image.
* If the prompt asks a person to speak, say something, talk, or deliver dialogue, do not convert that speech requirement into a mouth-motion or speaking-motion visual question. Ignore it for Visual Task Following.

## Output requirements

Return only one valid JSON object. Do not wrap it in markdown.

Required JSON schema:

{
"sample_id": "string",
"applicable_dimensions": {
"Visual Task Following": true
},
"checklist": {
"visual_task_following": [
"string"
]
},
"notes": {
"checklist_generation_reason": "string"
}
}

If Visual Task Following is not applicable, return:

{
"sample_id": "string",
"applicable_dimensions": {
"Visual Task Following": false
},
"checklist": {
"visual_task_following": []
},
"notes": {
"checklist_generation_reason": "string"
}
}
