#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"     # Set DEVICE=cpu for CPU
DEVICE="${DEVICE:-auto}"
PYTHON="${PYTHON:-python}"
CONFIG_FILE="${CONFIG_FILE:-configs/vae_waveform_16k_5s.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/vae_waveform_16k_5s/checkpoints/best.pt}"

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
exec "$PYTHON" evaluate.py --config "$CONFIG_FILE" --device "$DEVICE" --checkpoint "$CHECKPOINT" "$@"
