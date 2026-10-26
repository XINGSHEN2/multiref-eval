#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[moved] Anatomical belongs to BQ now; forwarding to ${ROOT}/BQ/run_mllm_anatomical_from_xlsx.sh" >&2
exec bash "${ROOT}/BQ/run_mllm_anatomical_from_xlsx.sh"
