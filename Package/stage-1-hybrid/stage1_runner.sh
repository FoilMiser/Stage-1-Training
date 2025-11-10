#!/usr/bin/env bash
# stage1_runner.sh — Stage-1 Hybrid runner (precompute | train)
# Adds support for a zipped HuggingFace-style STUDENT checkpoint in GCS.

set -Eeuo pipefail
IFS=$'\n\t'
export PYTHONUNBUFFERED=1

log()   { printf "[stage1][%(%F %T)T] %s\n" -1 "$*"; }
fatal() { printf "[stage1][%(%F %T)T][FATAL] %s\n" -1 "$*" >&2; exit 2; }
trap 'log "ERROR: failed (line ${BASH_LINENO[*]})"' ERR

# -------------------- Mode --------------------
MODE="${1:-train}"   # precompute | train

# -------------------- Cloud + paths (overridable) --------------------
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value core/project 2>/dev/null || echo liquid-llm)}"
REGION="${REGION:-northamerica-northeast2}"
BUCKET="${BUCKET:-gs://liquid-llm-bucket-2}"

# Package/toolkit zips in GCS (adjust if yours live elsewhere)
PKG_ZIP="${PKG_ZIP:-$BUCKET/sandbox/stage-1-hybrid/stage-1-hybrid.zip}"
TOOLKIT_ZIP="${TOOLKIT_ZIP:-$BUCKET/sandbox/preprocess-toolkit/preprocess-toolkit-stage1-1-0.zip}"

# Dataset manifests (pass-through; optional)
LM_MANIFEST="${LM_MANIFEST:-$BUCKET/datasets/stage1/manifests/stage1_lm.filesliced.jsonl}"
MC_MANIFEST="${MC_MANIFEST:-$BUCKET/datasets/stage1/manifests/stage1_math_code.jsonl}"

# -------------------- Student checkpoint --------------------
STUDENT_ZIP_URI="${STUDENT_ZIP_URI:-gs://liquid-llm-bucket-2/stage1/Checkpoints/student.zip}"
STUDENT_DIR="${STUDENT_DIR:-/opt/models/student}"
STUDENT_DIR_GCS="${STUDENT_DIR_GCS:-}"

# -------------------- Runtime hygiene --------------------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export ARROW_NUM_THREADS="${ARROW_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_ENABLE_HF_TRANSFER=1

# -------------------- Local layout --------------------
ROOT="/opt"
CODE_ROOT="$ROOT/code"
PKG_DIR="$CODE_ROOT/stage-1-hybrid"
TOOLKIT_DIR="$ROOT/toolkit"
DATA_ROOT="$ROOT/data"
mkdir -p "$CODE_ROOT" "$TOOLKIT_DIR" "$DATA_ROOT" "$(dirname "$STUDENT_DIR")"

# -------------------- Helpers --------------------
download_unzip() { # $1=gs://...zip  $2=dest_dir
  local uri="$1" dest="$2"
  [[ "$uri" == gs://* ]] || fatal "Expected GCS URI for zip, got: $uri"
  mkdir -p "$dest"
  local z="/tmp/$(basename "$uri")"
  log "Downloading $uri → $z"
  gcloud storage cp "$uri" "$z"
  log "Unzipping $z → $dest"
  unzip -q -o "$z" -d "$dest"
}

flatten_if_nested_hf_root() { # $1=dir that should contain config.json
  local out="$1"
  local inner
  inner="$(find "$out" -maxdepth 2 -type f -name config.json | head -n1 || true)"
  [[ -z "$inner" ]] && return 0
  local root; root="$(dirname "$inner")"
  if [[ "$root" != "$out" ]]; then
    log "Flattening nested HF folder: moving $root → $out"
    shopt -s dotglob
    tmp="$out.__tmp__"; rm -rf "$tmp"; mkdir -p "$tmp"
    mv "$root"/* "$tmp"/
    rm -rf "$out"/*
    mv "$tmp"/* "$out"/
    rmdir "$tmp" || true
    shopt -u dotglob
  fi
}

check_hf_folder() { # $1=dir; verify minimal HF files
  local d="$1"
  [[ -d "$d" ]] || fatal "Student folder missing: $d"
  [[ -f "$d/config.json" ]] || fatal "Student folder lacks config.json: $d"
  if compgen -G "$d/model.safetensors*" > /dev/null; then
    :
  elif [[ -f "$d/pytorch_model.bin" || -f "$d/pytorch_model.pt" ]]; then
    :
  else
    fatal "No model weights found (safetensors shards or pytorch_model.*) in: $d"
  fi
  if [[ ! -f "$d/tokenizer.json" && ! -f "$d/tokenizer.model" && ! -f "$d/vocab.json" ]]; then
    log "WARN: tokenizer files not found in student folder (may still load if tokenizer is elsewhere)."
  fi
}

# -------------------- Fetch package + toolkit --------------------
log "Fetching runner assets…"
gcloud storage cp "$PKG_ZIP" /tmp/pkg.zip
gcloud storage cp "$TOOLKIT_ZIP" /tmp/toolkit.zip || true

log "Extracting package → $PKG_DIR"
rm -rf "$PKG_DIR"; mkdir -p "$PKG_DIR"
unzip -q /tmp/pkg.zip -d "$CODE_ROOT"

# Some zips may nest the repo; ensure "$PKG_DIR" exists
if [[ ! -d "$PKG_DIR" ]]; then
  found="$(find "$CODE_ROOT" -maxdepth 2 -type d -name 'stage-1-hybrid' | head -n1 || true)"
  [[ -n "$found" ]] || fatal "Could not locate extracted stage-1-hybrid package directory."
  mv "$found" "$PKG_DIR"
fi

if [[ -f /tmp/toolkit.zip ]]; then
  log "Extracting toolkit → $TOOLKIT_DIR"
  rm -rf "$TOOLKIT_DIR"; mkdir -p "$TOOLKIT_DIR"
  unzip -q /tmp/toolkit.zip -d "$TOOLKIT_DIR"
fi

# -------------------- Put code on PYTHONPATH (fixes import resolution) --------------------
export PYTHONPATH="/opt/code/stage-1-hybrid:${PYTHONPATH:-}"
log "PYTHONPATH=$PYTHONPATH"

# -------------------- Quiet the noisy sitecustomize (optional) --------------------
# If sitecustomize imports pythonjsonlogger, make sure it's present.
python - <<'PY' >/dev/null 2>&1 || true
import sys, subprocess
try:
    import pythonjsonlogger  # noqa: F401
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "python-json-logger"])
PY

# -------------------- Hugging Face token via Secret Manager (no gcloud prompts) --------------------
HF_TOKEN_SECRET="${HF_TOKEN_SECRET:-}"   # e.g. "projects/982634794264/secrets/hf-token"
if [[ -n "${HF_TOKEN_SECRET:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  log "Fetching HF token from Secret Manager: ${HF_TOKEN_SECRET##*/}"
  python - <<PY
import sys
from google.cloud import secretmanager
name = "${HF_TOKEN_SECRET}/versions/latest" if "/versions/" not in "${HF_TOKEN_SECRET}" else "${HF_TOKEN_SECRET}"
client = secretmanager.SecretManagerServiceClient()
payload = client.access_secret_version(request={"name": name}).payload.data.decode("utf-8").strip()
print(payload)
PY
  HF_TOKEN_RC=$?
  if [[ "$HF_TOKEN_RC" -ne 0 ]]; then
    # Install the client lib once and retry
    python -m pip install -q google-cloud-secretmanager
    HUGGING_FACE_HUB_TOKEN="$(python - <<PY
from google.cloud import secretmanager
name = "${HF_TOKEN_SECRET}/versions/latest" if "/versions/" not in "${HF_TOKEN_SECRET}" else "${HF_TOKEN_SECRET}"
print(secretmanager.SecretManagerServiceClient()
      .access_secret_version(request={"name": name})
      .payload.data.decode("utf-8").strip())
PY
)"
  else
    HUGGING_FACE_HUB_TOKEN="$(python - <<'PY'
import sys
print(sys.stdin.read().strip())
PY
)"
  fi
  export HUGGING_FACE_HUB_TOKEN
fi

# -------------------- Prepare STUDENT (zip or directory) --------------------
STUDENT_ARG=()
if [[ -n "${STUDENT_ZIP_URI:-}" ]]; then
  log "Preparing student from zip: $STUDENT_ZIP_URI"
  rm -rf "$STUDENT_DIR"; mkdir -p "$STUDENT_DIR"
  download_unzip "$STUDENT_ZIP_URI" "$STUDENT_DIR"
  flatten_if_nested_hf_root "$STUDENT_DIR"
  check_hf_folder "$STUDENT_DIR"
  STUDENT_ARG=(--student-model-path "$STUDENT_DIR")
elif [[ -n "${STUDENT_DIR_GCS:-}" ]]; then
  log "Preparing student from GCS directory: $STUDENT_DIR_GCS"
  rm -rf "$STUDENT_DIR"; mkdir -p "$STUDENT_DIR"
  gcloud storage cp -r "$STUDENT_DIR_GCS"/* "$STUDENT_DIR"/
  flatten_if_nested_hf_root "$STUDENT_DIR"
  check_hf_folder "$STUDENT_DIR"
  STUDENT_ARG=(--student-model-path "$STUDENT_DIR")
else
  log "No STUDENT_ZIP_URI/STUDENT_DIR_GCS provided; relying on Python defaults."
fi

# -------------------- Python args --------------------
COMMON_ARGS=("${STUDENT_ARG[@]}")
[[ -n "${LM_MANIFEST:-}" ]] && COMMON_ARGS+=(--lm-manifest "$LM_MANIFEST")
[[ -n "${MC_MANIFEST:-}" ]] && COMMON_ARGS+=(--mc-manifest "$MC_MANIFEST")
[[ -n "${LM_TEACHER_ID:-}" ]] && COMMON_ARGS+=(--lm-teacher-id "$LM_TEACHER_ID")
[[ -n "${MC_TEACHER_ID:-}" ]] && COMMON_ARGS+=(--mc-teacher-id "$MC_TEACHER_ID")
[[ -n "${TEACHER_CACHE_DIR:-}" ]] && COMMON_ARGS+=(--teacher-cache-dir "$TEACHER_CACHE_DIR")
[[ -n "${TORCH_DTYPE:-}" ]] && COMMON_ARGS+=(--torch-dtype "$TORCH_DTYPE")      # e.g., bfloat16
[[ -n "${GRAD_ACCUM_STEPS:-}" ]] && COMMON_ARGS+=(--grad-accum-steps "$GRAD_ACCUM_STEPS")
[[ -n "${CONFIG_DS_JSON:-}" ]] && COMMON_ARGS+=(--deepspeed "$CONFIG_DS_JSON")

# -------------------- Run --------------------
cd "$PKG_DIR"

if [[ "$MODE" == "precompute" ]]; then
  log "Running precompute (hybrid logits) with args: ${COMMON_ARGS[*]}"
  python -m stage1_hybrid.precompute_hybrid_logits "${COMMON_ARGS[@]}"
  log "Precompute completed."
elif [[ "$MODE" == "train" ]]; then
  TRAIN_ENTRY="${TRAIN_ENTRY:-stage1_hybrid.train_distill}"
  log "Running train ($TRAIN_ENTRY) with args: ${COMMON_ARGS[*]}"
  python -m "$TRAIN_ENTRY" "${COMMON_ARGS[@]}"
  log "Training completed."
else
  fatal "Unknown MODE: $MODE (expected: precompute | train)"
fi

log "Done."
