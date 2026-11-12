"""Reconstruction metrics with explicit per-example validity."""
import math
from collections import defaultdict

import torch

from .losses import spectral_convergence


def audio_metrics(estimate, reference, transform, spectral_eps, silence_rms):
    estimate, reference = estimate.float(), reference.float()
    if estimate.shape != reference.shape:
        raise ValueError("Audio metrics require aligned waveforms of equal length")
    if not (estimate.isfinite().all() and reference.isfinite().all()):
        return {"valid": False, "error": "non_finite_waveform"}
    signal = reference.square().mean()
    if signal <= silence_rms ** 2:
        return {"valid": False, "error": "silent_or_low_energy_reference"}
    eps = 1e-8
    noise = (estimate - reference).square().mean()
    sdr = -10 * (noise / signal).clamp_min(eps).log10()
    ref_centered = reference - reference.mean()
    est_centered = estimate - estimate.mean()
    ref_variance = ref_centered.square().mean()
    si_snr = None
    si_status = "low_variance_reference"
    if ref_variance > silence_rms ** 2:
        projection = (est_centered * ref_centered).mean() / ref_variance * ref_centered
        projected_power = projection.square().mean()
        residual_power = (est_centered - projection).square().mean()
        if est_centered.square().mean() <= silence_rms ** 2:
            si_snr = 10 * math.log10(eps)
            si_status = "low_variance_estimate"
        else:
            ratio = projected_power / (residual_power + eps * projected_power)
            si_snr = float(10 * ratio.clamp_min(eps).log10())
            si_status = "valid"
    predicted, target = transform(torch.stack((estimate, reference))).abs().chunk(2)
    sc = spectral_convergence(predicted, target, spectral_eps)
    log_difference = 20 * (
        predicted.clamp_min(spectral_eps).log10() - target.clamp_min(spectral_eps).log10()
    )
    lsd = log_difference.square().mean(dim=1).sqrt().mean()
    metrics = {
        "sdr_db": float(sdr),
        "si_snr_db": si_snr,
        "stft_convergence": float(sc.item()),
        "log_spectral_distance_db": float(lsd),
    }
    if not all(value is None or math.isfinite(value) for value in metrics.values()):
        return {"valid": False, "error": "non_finite_metric"}
    return {"valid": True, "si_snr_status": si_status, **metrics}


def summarize(rows):
    fields = defaultdict(list)
    valid_count, labeled_count = 0, 0
    for row in rows:
        if not row["valid"]:
            continue
        valid_count += 1
        labeled_count += int(row.get("class_correct") is not None)
        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in ("valid", "length"):
                fields[key].append(value)
    return {
        "valid_count": valid_count,
        "invalid_count": len(rows) - valid_count,
        "si_snr_count": len(fields.get("si_snr_db", [])),
        "labeled_count": labeled_count,
        "means": {key: math.fsum(values) / len(values) for key, values in fields.items()},
    }
