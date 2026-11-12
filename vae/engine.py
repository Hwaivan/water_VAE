"""Length-aware training batches and deterministic reconstruction evaluation."""
import math

import torch

from .dataset import length_groups, write_audio
from .metrics import audio_metrics, summarize


def train_batch(model, criterion, batch, optimizer, scaler, device, step, clip_norm):
    optimizer.zero_grad(set_to_none=True)
    chunks, class_chunks = [], []
    for _, clean, labels in length_groups(batch, device):
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            result = model(clean)
        terms, ce, _, labeled = criterion(result, clean, labels, step)
        chunks.append(terms)
        class_chunks.append(ce[labeled])
    means = {key: torch.cat([chunk[key] for chunk in chunks]).mean() for key in chunks[0]}
    class_losses = torch.cat(class_chunks)
    labeled_count = class_losses.numel()
    total = means["vae_loss"]
    if labeled_count:
        classification = class_losses.mean()
        total = total + criterion.classification_weight * classification
    values = {key: float(value.detach()) for key, value in means.items()}
    values["total_loss"] = float(total.detach())
    values["classification_loss"] = float(classification.detach()) if labeled_count else None
    if not all(value is None or math.isfinite(value) for value in values.values()):
        raise FloatingPointError(f"Non-finite training loss or posterior statistics at step {step}")

    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    if clip_norm or not scaler.is_enabled():
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip_norm or float("inf"), error_if_nonfinite=not scaler.is_enabled(),
        )
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return values, labeled_count, scaler.get_scale() >= scale_before


@torch.inference_mode()
def evaluate_batches(model, criterion, loader, config, device, step, *, examples_dir=None):
    model.eval()
    rows = []
    example_limit = config["evaluation"]["save_examples"] if examples_dir else 0
    examples = 0
    for batch in loader:
        batch_rows = [None] * len(batch["items"])
        for indices, clean, labels in length_groups(batch, device):
            result = model(clean)
            terms, ce, correct, labeled = criterion(result, clean, labels, step)
            for offset, index in enumerate(indices):
                item = batch["items"][index]
                row = {key: item[key] for key in ("file_id", "path", "length")}
                row.update(audio_metrics(
                    result.waveform[offset], clean[offset], criterion.transforms[0],
                    criterion.eps, config["metrics"]["silence_rms"],
                ))
                values = {key: float(value[offset]) for key, value in terms.items()}
                if labeled[offset]:
                    values["classification_loss"] = float(ce[offset])
                    values["class_correct"] = int(correct[offset])
                if all(math.isfinite(value) for value in values.values()):
                    row.update(values)
                else:
                    row.update(valid=False, error="non_finite_loss_or_posterior")
                batch_rows[index] = row
                if row["valid"] and examples < example_limit:
                    rate = config["data"]["sample_rate"]
                    name = f"{examples:03d}"
                    write_audio(examples_dir / f"{name}_clean.wav", clean[offset], rate)
                    write_audio(examples_dir / f"{name}_reconstructed.wav", result.waveform[offset], rate)
                    examples += 1
        rows.extend(batch_rows)
        rows.extend({**item, "valid": False} for item in batch["errors"])
    summary = summarize(rows)
    kl = summary["means"].get("kl")
    summary["posterior_collapse_suspected"] = (
        kl < config["metrics"]["collapse_kl_threshold"] if kl is not None else None
    )
    summary["beta"] = criterion.beta(step)
    return summary, rows
