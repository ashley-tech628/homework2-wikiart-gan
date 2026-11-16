#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR=$(dirname -- "$0")
cd -- "$PACKAGE_DIR"
PACKAGE_DIR=$PWD
ACTION="${1:-train}"
if [ "$ACTION" != train ] && [ "$ACTION" != evaluate ] && [ "$ACTION" != preview ]; then
  echo 'Usage: bash run_datahub.sh [train|evaluate|preview]'; exit 2
fi
ROOT="${HW2_TMP_ROOT:-/tmp/homework2-full-wikiart-${USER:-user}}"
SCRATCH="$ROOT/stylegan2"
PERSIST="${HW2_PERSIST:-$HOME/homework2-stylegan2-wikiart}"
VENV="${HW2_VENV:-/tmp/homework2-venv}"
mkdir -p "$SCRATCH" "$PERSIST"
exec 9>"$PERSIST/run.lock"
flock -n 9 || { echo 'Another run is already using this experiment directory.'; exit 1; }
export TMPDIR="$SCRATCH/tmp" XDG_CACHE_HOME="$SCRATCH/cache"
export TORCH_HOME="$SCRATCH/torch" MPLCONFIGDIR="$SCRATCH/mpl"
export KAGGLEHUB_CACHE="$SCRATCH/kagglehub" HF_HOME="$SCRATCH/hf"
export PIP_CACHE_DIR="$SCRATCH/pip" PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-backend:cudaMallocAsync}"
export TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT="${TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT:-128}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$KAGGLEHUB_CACHE"
echo "Images/downloads/caches/env: $ROOT and $VENV (temporary)."
echo "Durable checkpoints/results: $PERSIST (home); allow about 2 GiB for atomic saves."
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
if [ ! -x "$VENV/bin/python" ]; then python3 -m venv --system-site-packages "$VENV"; fi
PY="$VENV/bin/python"
if ! "$PY" -c 'import torch,torchvision; assert torch.cuda.is_available(); print(torch.__version__,torch.ones(1,device="cuda")+1)'; then
  "$PY" -m pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
fi
"$PY" -m pip install --no-cache-dir --no-deps pytorch-ignite==0.5.4
"$PY" -m pip install --no-cache-dir 'numpy<3' scipy pillow matplotlib requests click packaging 'kagglehub==1.0.3'
"$PY" "$PACKAGE_DIR/pretrained.py" --cache "$SCRATCH/pretrained"
INITIAL="$SCRATCH/pretrained/initial.pt"
if [ ! -f "$PERSIST/pretrained_before_finetuning_psi1.png" ]; then
  "$PY" "$PACKAGE_DIR/sample.py" --checkpoint "$INITIAL" --output "$PERSIST/pretrained_before_finetuning_psi1.png"
fi
if [ "$ACTION" = preview ]; then
  echo "Public-pretraining preview (not your fine-tuned result): $PERSIST/pretrained_before_finetuning_psi1.png"; exit 0
fi
if [ -n "${WIKIART_SOURCE:-}" ]; then
  SOURCE="$WIKIART_SOURCE"
else
  SOURCE="$ROOT/raw"
  "$PY" "$PACKAGE_DIR/download_data.py" --output "$SOURCE"
fi
MANIFEST="$SCRATCH/dataset_manifest.json"
"$PY" "$PACKAGE_DIR/data.py" --source "$SOURCE" --output "$MANIFEST"
if [ "$ACTION" = train ]; then
  "$PY" "$PACKAGE_DIR/train.py" --initial "$INITIAL" --manifest "$MANIFEST" \
    --output "$SCRATCH/run" --persist "$PERSIST" --kimg "${HW2_KIMG:-80}" \
    --max-hours "${HW2_MAX_HOURS:-2.5}" --batch "${HW2_BATCH:-16}" \
    --microbatch "${HW2_MICROBATCH:-2}" --workers "${HW2_WORKERS:-2}"
fi
if [ -f "$PERSIST/best_finetuned.pt" ]; then
  "$PY" "$PACKAGE_DIR/sample.py" --checkpoint "$PERSIST/best_finetuned.pt" --output "$PERSIST/best_grid_psi1.png"
  "$PY" "$PACKAGE_DIR/sample.py" --checkpoint "$PERSIST/best_finetuned.pt" --output "$PERSIST/best_grid_psi07.png" --psi .7
  COMPLETE=$("$PY" -c 'import json,sys; s=json.load(open(sys.argv[1])); print(int(s["nimg"]>=s["target_nimg"]))' "$PERSIST/status.json")
  if [ "$ACTION" = evaluate ] || [ "$COMPLETE" = 1 ]; then
    "$PY" "$PACKAGE_DIR/evaluate.py" --checkpoint "$INITIAL" --manifest "$MANIFEST" \
      --cache "$SCRATCH/run/metric_cache" --output "$PERSIST/pretrained_test_metrics.json" --count "${HW2_EVAL_IMAGES:-1000}"
    "$PY" "$PACKAGE_DIR/evaluate.py" --checkpoint "$PERSIST/best_finetuned.pt" --manifest "$MANIFEST" \
      --cache "$SCRATCH/run/metric_cache" --output "$PERSIST/finetuned_test_metrics.json" --count "${HW2_EVAL_IMAGES:-1000}"
  fi
fi
"$PY" "$PACKAGE_DIR/package_results.py" --run "$PERSIST" --output "$HOME/homework2-stylegan2-results.zip"
cat "$PERSIST/status.json"
echo
echo 'If status=session_time_limit: run the same command again to resume, including after a server restart.'
echo 'If status=quality_guard: inspect the saved best result; do not automatically extend training.'
echo "Download: $HOME/homework2-stylegan2-results.zip"
