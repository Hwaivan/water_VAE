"""Train or resume clean waveform VAE pretraining."""
import argparse
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import torch
import yaml

from vae.checkpoint import (
    FORMAT_VERSION, capture_rng, load_checkpoint, make_scaler, restore_rng, save_checkpoint,
)
from vae.dataset import CleanAudioDataset, make_loader
from vae.engine import evaluate_batches, train_batch
from vae.losses import VAELoss
from vae.utils import add_runtime_args, append_jsonl, build_model, load_config, resolve_device, seed_everything


def initialize_run(output, resume, config):
    last = output / "checkpoints" / "last.pt"
    existing = output.exists() and any(output.iterdir())
    state = load_checkpoint(resume, config) if resume else None
    if existing:
        if not state or not last.is_file() or resume.resolve() != last.resolve():
            raise ValueError(f"{output} already contains a run; resume its last.pt or select a fresh --output-dir")
    if state:
        for section in ("data", "loss"):
            if state["config"][section] != config[section]:
                raise ValueError(f"Resume requires matching checkpoint {section} configuration")
        if state["config"]["experiment"]["seed"] != config["experiment"]["seed"]:
            raise ValueError("Resume requires matching experiment.seed")
        for key in ("batch_size", "learning_rate", "weight_decay", "gradient_clip_norm"):
            if state["config"]["training"][key] != config["training"][key]:
                raise ValueError(f"Resume requires matching training.{key}")
        if state["epoch"] >= config["training"]["epochs"]:
            raise ValueError("Increase training.epochs above the saved epoch to resume")
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("logs", "checkpoints", "reconstructions"):
        (output / directory).mkdir(exist_ok=True)
    if state:
        best = output / "checkpoints" / "best.pt"
        if state["epoch"] == state["best_epoch"]:
            save_checkpoint(best, state)
        elif not existing:
            source_best = resume.with_name("best.pt")
            previous_best = load_checkpoint(source_best, config) if source_best.is_file() else None
            if (previous_best and previous_best["run_id"] == state["run_id"]
                    and previous_best["epoch"] == state["best_epoch"]):
                shutil.copyfile(source_best, best)
            else:
                # Start the new run from the best reconstruction checkpoint available to it.
                state["best_epoch"] = state["epoch"]
                state["best_metric"] = state["validation_score"]
                save_checkpoint(best, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, help="Total number of epochs including completed epochs")
    parser.add_argument("--experiment-name")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.experiment_name:
        config["experiment"]["name"] = args.experiment_name
        config["experiment"]["output_dir"] = str(
            Path(config["experiment"]["output_dir"]).with_name(args.experiment_name)
        )
    if args.output_dir:
        config["experiment"]["output_dir"] = str(args.output_dir.resolve())
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        config["training"]["epochs"] = args.epochs

    output = Path(config["experiment"]["output_dir"])
    training = config["training"]
    seed_everything(config["experiment"]["seed"])
    device = resolve_device(args.device, config)
    train = CleanAudioDataset(config, "train")
    valid = CleanAudioDataset(config, "valid", report_invalid=True)
    model = build_model(config).to(device)
    criterion = VAELoss(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    scaler = make_scaler(device.type == "cuda" and training["amp"])
    state = initialize_run(output, args.resume, config)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "logs" / "train.log", encoding="utf-8")],
    )
    logger = logging.getLogger("vae")
    start_epoch, step, best, best_epoch, run_id = 1, 0, float("inf"), 0, uuid4().hex
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scaler.is_enabled() and state["scaler"]:
            scaler.load_state_dict(state["scaler"])
        restore_rng(state["rng_state"], device)
        start_epoch, step = state["epoch"] + 1, state["global_step"]
        best, best_epoch, run_id = state["best_metric"], state["best_epoch"], state["run_id"]
        logger.info("Resumed %s at epoch %d, optimizer step %d", args.resume, start_epoch, step)
    config["runtime"] = {
        "device": str(device),
        "amp_enabled": scaler.is_enabled(),
        "parameters": sum(param.numel() for param in model.parameters()),
        "downsample_factor": model.downsample_factor,
        "latent_hz": config["data"]["sample_rate"] / model.downsample_factor,
    }
    with (output / "runtime_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    logger.info("%s train=%d valid=%d", config["runtime"], len(train), len(valid))

    for epoch in range(start_epoch, training["epochs"] + 1):
        model.train()
        sums, counts = defaultdict(float), defaultdict(int)
        skipped, initial_step = 0, step
        for batch in make_loader(train, config, device, epoch):
            metrics, labeled_count, updated = train_batch(
                model, criterion, batch, optimizer, scaler, device, step + 1,
                training["gradient_clip_norm"],
            )
            step += int(updated)
            skipped += int(not updated)
            for key, value in metrics.items():
                if value is not None:
                    count = labeled_count if key == "classification_loss" else len(batch["items"])
                    sums[key] += value * count
                    counts[key] += count
        if step == initial_step:
            raise FloatingPointError(f"Epoch {epoch} made no optimizer updates")
        validation, rows = evaluate_batches(
            model, criterion, make_loader(valid, config, device), config, device, step,
            examples_dir=output / "reconstructions" / f"epoch_{epoch:04d}",
        )
        for row in rows:
            if not row["valid"]:
                logger.warning("Invalid validation sample %s: %s", row["file_id"], row["error"])
        if not validation["valid_count"]:
            raise ValueError("Validation contains no valid reconstruction metrics")
        score = validation["means"]["reconstruction_loss"]
        improved = score < best
        if improved:
            best, best_epoch = score, epoch
        payload = {
            "format_version": FORMAT_VERSION, "run_id": run_id,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "rng_state": capture_rng(device),
            "epoch": epoch, "global_step": step, "config": config,
            "best_metric": best, "best_epoch": best_epoch, "validation_score": score,
        }
        save_checkpoint(output / "checkpoints" / "last.pt", payload)
        if improved:
            save_checkpoint(output / "checkpoints" / "best.pt", payload)
        training_means = {key: value / counts[key] for key, value in sums.items()}
        training_means.setdefault("classification_loss", None)
        append_jsonl(output / "metrics.jsonl", {
            "epoch": epoch, "global_step": step, "skipped_amp_steps": skipped,
            "beta": criterion.beta(step), "train": training_means, "validation": validation,
        })
        logger.info("Epoch %d step=%d skipped=%d train_vae=%.5f train_cls=%s valid_reconstruction=%.5f",
                    epoch, step, skipped, training_means["vae_loss"],
                    training_means["classification_loss"], score)
        if validation["posterior_collapse_suspected"]:
            logger.warning("Validation mean KL below posterior collapse threshold")


if __name__ == "__main__":
    main()
