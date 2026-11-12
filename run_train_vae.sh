#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"     # Set DEVICE=cpu for CPU
DEVICE="${DEVICE:-auto}"
PYTHON="${PYTHON:-python}"
CONFIG_FILE="${CONFIG_FILE:-configs/vae_waveform_16k_5s.yaml}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
RESUME="${RESUME:-}"

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
args=(--config "$CONFIG_FILE" --device "$DEVICE")
if [[ -n "$EXPERIMENT_NAME" ]]; then args+=(--experiment-name "$EXPERIMENT_NAME"); fi
if [[ -n "$RESUME" ]]; then args+=(--resume "$RESUME"); fi
exec "$PYTHON" train.py "${args[@]}" "$@"
