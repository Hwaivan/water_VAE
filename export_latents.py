"""Export full-file posterior means and optionally fit training-only latent statistics."""
import argparse
import hashlib
from pathlib import Path

import torch

from vae.checkpoint import FORMAT_VERSION, load_model, save_checkpoint
from vae.dataset import load_audio, read_manifest
from vae.utils import add_runtime_args, append_jsonl, load_config, resolve_device, write_json


class ChannelMoments:
    """Merge per-file population moments in float64, weighting each latent frame."""

    def __init__(self, channels):
        self.count = 0
        self.mean = torch.zeros(channels, dtype=torch.float64)
        self.m2 = torch.zeros_like(self.mean)

    def update(self, latent):
        latent = latent.double()
        frames = latent.shape[-1]
        variance, mean = torch.var_mean(latent, dim=-1, correction=0)
        total = self.count + frames
        delta = mean - self.mean
        self.m2 += variance * frames + delta.square() * (self.count * frames / total)
        self.mean += delta * (frames / total)
        self.count = total

    def summary(self):
        std = (self.m2 / self.count).sqrt()
        return {
            "frame_count": self.count,
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "normalization_std": std.clamp_min(1e-6).tolist(),
        }


def training_entries(data):
    manifest = Path(data["train_manifest"])
    entries = read_manifest(manifest, data["manifest_mode"], data.get("audio_root"))
    clean_paths = {entry.clean_path for entry in entries}
    for split in ("valid", "test"):
        heldout = data.get(f"{split}_manifest")
        if heldout and Path(heldout).is_file():
            overlap = clean_paths.intersection(
                entry.clean_path for entry in read_manifest(
                    heldout, data["manifest_mode"], data.get("audio_root"),
                )
            )
            if overlap:
                raise ValueError(f"Training and {split} share a clean path: {min(overlap)}")
    return manifest, entries


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser, checkpoint=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", type=Path, help="Export this manifest without fitting statistics")
    mode.add_argument("--fit-train-stats", action="store_true", help="Export the checkpoint's train manifest and fit per-channel statistics")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device, config)
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        stream.seek(0)
        model, state = load_model(stream, config, device)
    source = {
        "path": str(args.checkpoint.resolve()),
        "sha256": digest.hexdigest(),
        "format_version": FORMAT_VERSION,
        "epoch": state["epoch"],
        "global_step": state["global_step"],
    }
    data = config["data"]
    moments = None
    if args.fit_train_stats:
        manifest, entries = training_entries(state["config"]["data"])
        moments = ChannelMoments(model.latent_channels)
    else:
        manifest = args.manifest.resolve()
        entries = read_manifest(manifest, data["manifest_mode"], data.get("audio_root"))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    for index, entry in enumerate(entries):
        audio = load_audio(entry.clean_path, data["sample_rate"], data["max_full_seconds"])
        mu, _ = model.encode(audio.waveform[None].to(device))
        latent = mu[0].cpu().contiguous()
        if not latent.isfinite().all():
            raise FloatingPointError(f"Non-finite latent for {entry.file_id}")
        file_name = f"{index:06d}.pt"
        metadata = {
            "file_id": entry.file_id, "clean_path": str(entry.clean_path),
            "waveform_length": audio.waveform.numel(), "sample_rate": data["sample_rate"],
            "source_length": audio.source_length, "source_sample_rate": audio.source_rate,
            "latent_shape": list(latent.shape), "downsample_factor": model.downsample_factor,
            "checkpoint": source, "latent_file": file_name,
        }
        save_checkpoint(output / file_name, {**metadata, "mu": latent})
        append_jsonl(output / "index.jsonl", metadata)
        if moments is not None:
            moments.update(latent)
    if moments is not None:
        write_json(output / "latent_stats.json", {
            "split": "train", "manifest": str(manifest), "checkpoint": source,
            "sample_rate": data["sample_rate"], "downsample_factor": model.downsample_factor,
            **moments.summary(),
        })
    print(f"Exported {len(entries)} posterior means to {output}")


if __name__ == "__main__":
    main()
