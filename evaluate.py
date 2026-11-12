"""Evaluate a waveform VAE using deterministic posterior-mean reconstruction."""
import argparse
import json
from pathlib import Path

from vae.checkpoint import load_model
from vae.dataset import CleanAudioDataset, make_loader
from vae.engine import evaluate_batches
from vae.losses import VAELoss
from vae.utils import add_runtime_args, load_config, resolve_device, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser, checkpoint=True)
    parser.add_argument("--split", default="test", choices=("valid", "test"))
    parser.add_argument("--manifest", "--test-manifest", type=Path)
    parser.add_argument("--full-audio", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.manifest:
        config["data"][f"{args.split}_manifest"] = str(args.manifest.resolve())
    device = resolve_device(args.device, config)
    model, state = load_model(args.checkpoint, config, device)
    dataset = CleanAudioDataset(config, args.split, full_audio=args.full_audio, report_invalid=True)
    criterion = VAELoss(config).to(device)
    output = args.output_dir or Path(config["experiment"]["output_dir"]) / "evaluation" / args.split
    output.mkdir(parents=True, exist_ok=False)
    summary, rows = evaluate_batches(
        model, criterion, make_loader(dataset, config, device), config, device,
        state["global_step"], examples_dir=output / "reconstructions",
    )
    summary.update(
        split=args.split, checkpoint=str(args.checkpoint.resolve()), epoch=state["epoch"],
        global_step=state["global_step"], sample_rate=dataset.sample_rate,
        manifest=config["data"][f"{args.split}_manifest"],
        segment_samples=dataset.segment_length, max_full_seconds=dataset.max_seconds,
    )
    with (output / "per_file.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["valid_count"]:
        raise SystemExit("No valid reconstruction metrics. See per_file.jsonl.")


if __name__ == "__main__":
    main()
