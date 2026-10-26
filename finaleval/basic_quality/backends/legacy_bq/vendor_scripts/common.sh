#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${CODE_ROOT}/.." && pwd)"
CACHE_BASE="${FINALEVAL_CACHE_ROOT:-${XDG_CACHE_HOME:-/tmp/finaleval-cache}}"
CACHE_ROOT="${BQ_CACHE_ROOT:-${CACHE_BASE}/basic-quality-cache}"
CONDA_ROOT="${BQ_CONDA_ROOT:-${CACHE_BASE}/conda}"

export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-${CONDA_ROOT}/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${CONDA_ROOT}/pkgs}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${CACHE_ROOT}/matplotlib}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export BQ_DOWNLOAD_DIR="${BQ_DOWNLOAD_DIR:-${CACHE_ROOT}/downloads}"
export BQ_GITHUB_MIRROR_PREFIX="${BQ_GITHUB_MIRROR_PREFIX:-}"
export BQ_HF_MIRROR_ENDPOINT="${BQ_HF_MIRROR_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT="${HF_ENDPOINT:-${BQ_HF_MIRROR_ENDPOINT}}"
export BQ_CORE_ENV="${BQ_CORE_ENV:-basic-quality-core}"
export BQ_USE_ACTIVE_ENV="${BQ_USE_ACTIVE_ENV:-0}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    export PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    export PYTHON_BIN="$(command -v python)"
  elif [[ -x "${HOME}/miniconda3/bin/python" ]]; then
    export PYTHON_BIN="${HOME}/miniconda3/bin/python"
  fi
fi

VIDEO_EXTENSIONS=(mp4 avi mov mkv webm m4v)

ensure_cache_layout() {
  mkdir -p \
    "${CACHE_ROOT}" \
    "${CONDA_ENVS_PATH}" \
    "${CONDA_PKGS_DIRS}" \
    "${HF_HOME}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${TORCH_HOME}" \
    "${XDG_CACHE_HOME}" \
    "${PIP_CACHE_DIR}" \
    "${MPLCONFIGDIR}" \
    "${TMPDIR}" \
    "${BQ_DOWNLOAD_DIR}"
}

ensure_conda() {
  if [[ "${BQ_USE_ACTIVE_ENV}" == "1" ]]; then
    return 0
  fi
  if ! command -v conda >/dev/null 2>&1; then
    local candidate
    for candidate in \
      "${BQ_CONDA_EXE:-}" \
      "/opt/conda/bin/conda" \
      "${HOME}/miniconda3/bin/conda" \
      "${HOME}/anaconda3/bin/conda"; do
      if [[ -n "${candidate}" && -x "${candidate}" ]]; then
        export PATH="$(dirname "${candidate}"):${PATH}"
        break
      fi
    done
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found. Install Miniconda/Anaconda or set BQ_CONDA_EXE." >&2
    exit 1
  fi
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1090
  source "${conda_base}/etc/profile.d/conda.sh"
}

conda_env_exists() {
  local env_name="$1"
  local env_root="${CONDA_ENVS_PATH%%:*}"
  conda env list | awk '{print $1}' | grep -Fxq "${env_name}" || [[ -d "${env_root}/${env_name}" ]]
}

conda_env_prefix() {
  local env_name="$1"
  local env_root="${CONDA_ENVS_PATH%%:*}"
  printf '%s/%s\n' "${env_root}" "${env_name}"
}

create_env_if_missing() {
  local env_name="$1"
  local python_version="$2"
  if ! conda_env_exists "${env_name}"; then
    conda create -y --prefix "$(conda_env_prefix "${env_name}")" "python=${python_version}"
  fi
}

conda_run_in() {
  local env_name="$1"
  shift
  if [[ "${BQ_USE_ACTIVE_ENV}" == "1" ]]; then
    if [[ "${1:-}" == "python" && -n "${PYTHON_BIN:-}" ]]; then
      shift
      "${PYTHON_BIN}" "$@"
      return
    fi
    "$@"
    return
  fi
  local env_prefix
  env_prefix="$(conda_env_prefix "${env_name}")"
  if [[ -d "${env_prefix}" ]]; then
    conda run --no-capture-output -p "${env_prefix}" "$@"
  else
    conda run --no-capture-output -n "${env_name}" "$@"
  fi
}

resolve_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  elif [[ -e "${PROJECT_ROOT}/${path}" ]]; then
    printf '%s\n' "${PROJECT_ROOT}/${path}"
  elif [[ -e "${CODE_ROOT}/${path}" ]]; then
    printf '%s\n' "${CODE_ROOT}/${path}"
  else
    printf '%s\n' "${PROJECT_ROOT}/${path}"
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: required file not found: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: required directory not found: ${path}" >&2
    exit 1
  fi
}

count_video_files() {
  local dir="$1"
  local count=0
  local ext
  shopt -s nullglob nocaseglob
  for ext in "${VIDEO_EXTENSIONS[@]}"; do
    for _ in "${dir}"/*."${ext}"; do
      count=$((count + 1))
    done
  done
  shopt -u nullglob nocaseglob
  printf '%s\n' "${count}"
}

require_video_files() {
  local dir="$1"
  local count
  count="$(count_video_files "${dir}")"
  if [[ "${count}" -eq 0 ]]; then
    echo "ERROR: no video files found in ${dir}" >&2
    exit 1
  fi
}

github_fast_url() {
  local url="$1"
  if [[ -n "${BQ_GITHUB_MIRROR_PREFIX}" ]] && [[ "${url}" == https://github.com/* || "${url}" == https://raw.githubusercontent.com/* ]]; then
    printf '%s/%s\n' "${BQ_GITHUB_MIRROR_PREFIX%/}" "${url}"
  else
    printf '%s\n' "${url}"
  fi
}

download_fast_url() {
  local url="$1"
  if [[ -n "${BQ_GITHUB_MIRROR_PREFIX}" ]] && [[ "${url}" == https://github.com/* || "${url}" == https://raw.githubusercontent.com/* ]]; then
    github_fast_url "${url}"
  elif [[ -n "${BQ_HF_MIRROR_ENDPOINT}" ]] && [[ "${url}" == https://huggingface.co/* ]]; then
    printf '%s/%s\n' "${BQ_HF_MIRROR_ENDPOINT%/}" "${url#https://huggingface.co/}"
  else
    printf '%s\n' "${url}"
  fi
}

download_file() {
  local url="$1"
  local destination="$2"
  mkdir -p "$(dirname "${destination}")"
  if [[ -s "${destination}" ]]; then
    return 0
  fi
  local tmp="${destination}.part"
  local fast_url
  fast_url="$(download_fast_url "${url}")"
  local curl_args=(
    --fail
    --location
    --http1.1
    --connect-timeout 30
    --retry 10
    --retry-delay 5
    --retry-max-time 1800
    --retry-all-errors
    --continue-at -
    --output "${tmp}"
  )
  if ! curl "${curl_args[@]}" "${fast_url}"; then
    curl "${curl_args[@]}" "${url}"
  fi
  mv "${tmp}" "${destination}"
}

ensure_system_deps() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y ffmpeg git curl wget libgl1 pkg-config build-essential libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libswresample-dev
}

ensure_core_env() {
  if [[ "${BQ_USE_ACTIVE_ENV}" == "1" ]]; then
    return 0
  fi
  local env_name="${BQ_CORE_ENV}"
  local marker="${CACHE_ROOT}/markers/${env_name}-core-v2.ready"
  local aesthetic_dir="${CODE_ROOT}/Objective/Video/aesthetic-predictor-v2-5"
  local audiobox_dir="${CODE_ROOT}/Objective/Audio/audiobox-aesthetics"
  local imagebind_dir="${CODE_ROOT}/Objective/Similarity/ImageBind-main"
  local audiobox_checkpoint="${CACHE_ROOT}/weights/audiobox-aesthetics/checkpoint.pt"
  local pytorchvideo_tar="${BQ_DOWNLOAD_DIR}/pytorchvideo-6cdc929315aab1b5674b6dcf73b16ec99147735f.tar.gz"
  if [[ -f "${marker}" ]] && conda_env_exists "${env_name}"; then
    return 0
  fi
  create_env_if_missing "${env_name}" 3.10
  mkdir -p "$(dirname "${marker}")"
  conda_run_in "${env_name}" pip install --upgrade pip
  conda_run_in "${env_name}" pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
  conda_run_in "${env_name}" pip install numpy==1.26.4 scipy pandas pyyaml scikit-learn seaborn matplotlib tqdm librosa soundfile opencv-python-headless pillow requests huggingface_hub
  conda_run_in "${env_name}" pip install timm ftfy regex einops iopath types-regex decord transformers==4.48.0 "accelerate>=0.30" "safetensors>=0.5.3" "rich>=13.9.4" "setuptools<81" submitit
  conda_run_in "${env_name}" pip install -e "${aesthetic_dir}"
  conda_run_in "${env_name}" pip install -e "${audiobox_dir}"
  download_file "https://dl.fbaipublicfiles.com/audiobox-aesthetics/checkpoint.pt" "${audiobox_checkpoint}"
  download_file "https://github.com/facebookresearch/pytorchvideo/archive/6cdc929315aab1b5674b6dcf73b16ec99147735f.tar.gz" "${pytorchvideo_tar}"
  mkdir -p "${CACHE_ROOT}/weights/imagebind"
  ln -sfn "${CACHE_ROOT}/weights/imagebind" "${imagebind_dir}/.checkpoints"
  conda_run_in "${env_name}" pip install "${pytorchvideo_tar}"
  conda_run_in "${env_name}" pip install -e "${imagebind_dir}" --no-deps
  conda_run_in "${env_name}" python -c "from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip; convert_v2_5_from_siglip('${aesthetic_dir}/models/aesthetic_predictor_v2_5.pth', low_cpu_mem_usage=True, trust_remote_code=True); print('aesthetic predictor ready')"
  conda_run_in "${env_name}" python -c "from audiobox_aesthetics.infer import initialize_predictor; initialize_predictor('${audiobox_checkpoint}'); print('audiobox checkpoint ready')"
  (cd "${imagebind_dir}" && conda_run_in "${env_name}" python -c "from imagebind.models import imagebind_model; imagebind_model.imagebind_huge(pretrained=True); print('imagebind checkpoint ready')")
  touch "${marker}"
}

ensure_aesthetic_env() {
  ensure_core_env
}

ensure_dover_env() {
  if [[ "${BQ_USE_ACTIVE_ENV}" == "1" ]]; then
    return 0
  fi
  local env_name="basic-quality-dover"
  local pkg_dir="${CODE_ROOT}/Objective/Video/DOVER"
  create_env_if_missing "${env_name}" 3.10
  conda_run_in "${env_name}" pip install --upgrade pip
  conda_run_in "${env_name}" pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
  conda_run_in "${env_name}" pip install -r "${pkg_dir}/requirements.txt"
  conda_run_in "${env_name}" pip install "numpy<2"
  conda_run_in "${env_name}" pip install -e "${pkg_dir}" --no-deps
  download_file "https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth" "${pkg_dir}/pretrained_weights/DOVER.pth"
  download_file "${HF_ENDPOINT%/}/teowu/DOVER/resolve/main/DOVER_plus_plus.pth?download=true" "${pkg_dir}/pretrained_weights/DOVER_plus_plus.pth"
}

ensure_audiobox_env() {
  ensure_core_env
}

ensure_nisqa_env() {
  ensure_core_env
}

ensure_imagebind_env() {
  ensure_core_env
}

ensure_synchformer_env() {
  local env_name="basic-quality-synchformer"
  local pkg_dir="${CODE_ROOT}/Objective/Similarity/Synchformer-main"
  local marker="${CACHE_ROOT}/markers/${env_name}-v1.ready"
  local cfg_path="${pkg_dir}/logs/sync_models/24-01-04T16-39-21/cfg-24-01-04T16-39-21.yaml"
  local ckpt_path="${pkg_dir}/logs/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt"
  if [[ -f "${marker}" ]] && conda_env_exists "${env_name}" && [[ -s "${cfg_path}" ]] && [[ -s "${ckpt_path}" ]]; then
    return 0
  fi
  if conda_env_exists "${env_name}" && [[ -s "${cfg_path}" ]] && [[ -s "${ckpt_path}" ]] \
      && conda_run_in "${env_name}" python -c "import av, omegaconf, torch, torchvision" >/dev/null 2>&1; then
    mkdir -p "$(dirname "${marker}")"
    touch "${marker}"
    return 0
  fi
  create_env_if_missing "${env_name}" 3.10
  conda_run_in "${env_name}" pip install --upgrade pip
  conda_run_in "${env_name}" pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
  conda_run_in "${env_name}" pip install numpy==1.26.4 scipy pandas matplotlib scikit-learn requests tqdm ffmpeg-python omegaconf==2.2.3 einops timm==0.6.7 transformers==4.27.4 wandb "setuptools<81"
  if ! conda_run_in "${env_name}" python -c "import av" >/dev/null 2>&1; then
    if ! conda install -y -n "${env_name}" -c conda-forge av=10.0.0; then
      conda_run_in "${env_name}" pip install av==10.0.0
    fi
  fi
  mkdir -p "${pkg_dir}/logs/sync_models/24-01-04T16-39-21"
  download_file "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-04T16-39-21/cfg-24-01-04T16-39-21.yaml" "${cfg_path}"
  download_file "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt" "${ckpt_path}"
  mkdir -p "$(dirname "${marker}")"
  touch "${marker}"
}

ensure_latentsync_env() {
  local env_name="basic-quality-latentsync"
  local pkg_dir="${CODE_ROOT}/Objective/Similarity/LatentSync"
  create_env_if_missing "${env_name}" 3.10
  conda_run_in "${env_name}" pip install --upgrade pip
  conda_run_in "${env_name}" pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
  conda_run_in "${env_name}" pip install -r "${pkg_dir}/requirements.txt"
  mkdir -p "${pkg_dir}/checkpoints/auxiliary" "${pkg_dir}/checkpoints/whisper"
  download_file "${HF_ENDPOINT%/}/ByteDance/LatentSync-1.6/resolve/main/auxiliary/syncnet_v2.model" "${pkg_dir}/checkpoints/auxiliary/syncnet_v2.model"
  download_file "${HF_ENDPOINT%/}/ByteDance/LatentSync-1.6/resolve/main/auxiliary/sfd_face.pth" "${pkg_dir}/checkpoints/auxiliary/sfd_face.pth"
  download_file "${HF_ENDPOINT%/}/ByteDance/LatentSync-1.6/resolve/main/auxiliary/koniq_pretrained.pkl" "${pkg_dir}/checkpoints/auxiliary/koniq_pretrained.pkl"
  download_file "${HF_ENDPOINT%/}/ByteDance/LatentSync-1.6/resolve/main/whisper/tiny.pt" "${pkg_dir}/checkpoints/whisper/tiny.pt"
}
