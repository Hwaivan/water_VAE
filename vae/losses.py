"""Per-example waveform, spectral and posterior losses."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .stft import STFT


def diagonal_kl(mu, logvar):
    mu, logvar = mu.float(), logvar.float()
    return 0.5 * (mu.square() + logvar.expm1() - logvar).mean(dim=(1, 2))


def spectral_convergence(estimate, target, eps, *, ignore_small_reference=False):
    error = torch.linalg.vector_norm(estimate - target, dim=(1, 2))
    reference = torch.linalg.vector_norm(target, dim=(1, 2))
    floor = eps * math.sqrt(target.shape[1] * target.shape[2])
    convergence = error / reference.clamp_min(floor)
    if ignore_small_reference:
        convergence = torch.where(reference > floor, convergence, 0.0)
    return convergence


class VAELoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        loss = config["loss"]
        self.waveform_weight = loss["waveform_weight"]
        self.mrstft_weight = loss["mrstft_weight"]
        self.kl_weight = loss["kl_weight"]
        self.warmup_steps = loss["kl_warmup_steps"]
        self.classification_weight = loss["classification_weight"]
        self.eps = loss["spectral_eps"]
        self.transforms = nn.ModuleList(STFT(**settings) for settings in loss["stft_resolutions"])

    def beta(self, step):
        return self.kl_weight * (min(1, step / self.warmup_steps) if self.warmup_steps else 1)

    def forward(self, result, target, labels, step):
        estimate = result.waveform.float()
        target = target.float()
        wave = (estimate - target).abs().mean(dim=1)
        spectral, log_magnitude = [], []
        for transform in self.transforms:
            estimated, reference = transform(torch.cat((estimate, target))).abs().chunk(2)
            spectral.append(spectral_convergence(
                estimated, reference, self.eps, ignore_small_reference=True
            ))
            log_magnitude.append(
                ((estimated + self.eps).log() - (reference + self.eps).log()).abs().mean(dim=(1, 2))
            )
        convergence = torch.stack(spectral).mean(dim=0)
        log_mag = torch.stack(log_magnitude).mean(dim=0)
        mrstft = convergence + log_mag
        kl = diagonal_kl(result.mu, result.logvar)
        reconstruction = self.waveform_weight * wave + self.mrstft_weight * mrstft
        vae = reconstruction + self.beta(step) * kl
        terms = {
            "waveform_l1": wave,
            "mrstft": mrstft,
            "mrstft_convergence": convergence,
            "mrstft_log_magnitude": log_mag,
            "kl": kl,
            "reconstruction_loss": reconstruction,
            "vae_loss": vae,
            "posterior_mu_mean": result.mu.mean(dim=(1, 2)),
            "posterior_mu_rms": result.mu.square().mean(dim=(1, 2)).sqrt(),
            "posterior_std_mean": (0.5 * result.logvar).exp().mean(dim=(1, 2)),
        }
        labeled = labels >= 0 if result.logits is not None else torch.zeros_like(labels, dtype=torch.bool)
        ce = wave.new_zeros(labels.shape)
        correct = torch.zeros_like(labeled)
        if labeled.any():
            logits = result.logits[labeled].float()
            ce[labeled] = F.cross_entropy(logits, labels[labeled], reduction="none")
            correct[labeled] = logits.argmax(dim=-1) == labels[labeled]
        return terms, ce, correct, labeled
