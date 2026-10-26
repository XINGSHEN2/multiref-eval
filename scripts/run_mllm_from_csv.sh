#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'MSG'
MLLM metrics are registered in this open-source skeleton but are not migrated yet.
Planned metrics:
  visual_event_sound_alignment, source_correctness,
  detail_preservation, detail_binding, anatomical_consistency,
  visual/audio/temporal/speech instruction-following metrics.
MSG
exit 2

