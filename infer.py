"""Reconstruct a mono WAV deterministically at the input rate and length."""
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from vae.checkpoint import load_model
from vae.dataset import load_audio, resample_waveform, write_audio
from vae.utils import add_runtime_args, load_config, resolve_device


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser, checkpoint=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    config = load_config(args.config)
    device = resolve_device(args.device, config)
    model, _ = load_model(args.checkpoint, config, device)
    data = config["data"]
    rate = data["sample_rate"]
    audio = load_audio(args.input, rate, data["max_full_seconds"])
    prediction = model(audio.waveform[None].to(device)).waveform[0].cpu()
    prediction = resample_waveform(prediction, rate, audio.source_rate)
    prediction = F.pad(prediction[:audio.source_length], (0, max(0, audio.source_length - prediction.numel())))
    write_audio(args.output, prediction, audio.source_rate)
    print(f"Wrote {args.output} ({audio.source_length} samples at {audio.source_rate} Hz)")


if __name__ == "__main__":
    main()
