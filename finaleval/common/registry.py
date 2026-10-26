from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    name: str
    group: str
    implemented: bool
    auxiliary: bool = False


MAIN_METRICS: list[MetricSpec] = [
    MetricSpec("visual_quality", "basic_quality", True),
    MetricSpec("audio_quality", "basic_quality", True),
    MetricSpec("anatomical_consistency", "basic_quality", True),
    MetricSpec("entity_fidelity", "entity_fidelity", True),
    MetricSpec("detail_preservation", "entity_fidelity", True),
    MetricSpec("detail_binding", "entity_fidelity", True),
    MetricSpec("speech_lip_synchronization", "audio_video_consistency", True),
    MetricSpec("visual_event_sound_alignment", "audio_video_consistency", True),
    MetricSpec("source_correctness", "audio_video_consistency", True),
    MetricSpec("voice_timbre_similarity", "audio_video_consistency", True),
    MetricSpec("visual_task_following", "instruction_following", True),
    MetricSpec("audio_task_following", "instruction_following", True),
    MetricSpec("temporal_order_following", "instruction_following", True),
    MetricSpec("speech_content_accuracy", "instruction_following", True),
]

AUXILIARY_METRICS: list[MetricSpec] = [
    MetricSpec("paste_artifact_penalty", "entity_fidelity", False, auxiliary=True),
    MetricSpec("stable_speaking_frontal_face", "audio_video_consistency", False, auxiliary=True),
]

ALL_MAIN_METRICS = {metric.name for metric in MAIN_METRICS}
IMPLEMENTED_METRICS = {metric.name for metric in MAIN_METRICS if metric.implemented}
CALLABLE_METRICS = ALL_MAIN_METRICS
