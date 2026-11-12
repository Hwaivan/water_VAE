"""Aligned convolutional waveform VAE."""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(math.gcd(8, channels, channels // 2), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=2, dilation=2),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.layers(x)


@dataclass
class VAEOutput:
    waveform: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor
    logits: torch.Tensor | None


class WaveformVAE(nn.Module):
    """Use [B,T] mono waveforms and [B,D,ceil(T/stride)] latents."""

    def __init__(
        self, latent_channels, channels, strides, classification_head=False, num_classes=None
    ):
        super().__init__()
        if not strides or len(channels) != len(strides) + 1:
            raise ValueError("model.channels must contain len(model.strides) + 1 widths")
        if any(stride < 2 for stride in strides) or min(*channels, latent_channels) < 2:
            raise ValueError("Model widths and strides must be at least 2")
        if classification_head and (num_classes is None or num_classes < 2):
            raise ValueError("model.classification_head requires model.num_classes >= 2")
        self.downsample_factor = math.prod(strides)
        self.latent_channels = latent_channels
        encoder = [nn.Conv1d(1, channels[0], 7, padding=3)]
        decoder = [nn.Conv1d(latent_channels, channels[-1], 3, padding=1)]
        stages = list(zip(channels[:-1], channels[1:], strides))
        for narrow, wide, stride in stages:
            encoder.extend([
                nn.Conv1d(narrow, wide, stride + 2, stride=stride, padding=1),
                ResidualBlock(wide),
            ])
        for narrow, wide, stride in reversed(stages):
            decoder.extend([
                ResidualBlock(wide),
                nn.ConvTranspose1d(wide, narrow, stride + 2, stride=stride, padding=1),
            ])
        decoder.append(nn.Conv1d(channels[0], 1, 7, padding=3))
        self.encoder = nn.Sequential(*encoder)
        self.mu_head = nn.Conv1d(channels[-1], latent_channels, 1)
        self.logvar_head = nn.Conv1d(channels[-1], latent_channels, 1)
        self.decoder = nn.Sequential(*decoder)
        self.classifier = nn.Linear(latent_channels, num_classes) if classification_head else None

    def encode(self, waveform):
        """Right-pad to the stride product and return float32 mu and logvar."""
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform[:, 0]
        if waveform.ndim != 2 or not waveform.shape[-1]:
            raise ValueError("Expected nonempty waveform [B,T] or [B,1,T]")
        pad = -waveform.shape[-1] % self.downsample_factor
        features = self.encoder(F.pad(waveform[:, None], (0, pad)))
        return self.mu_head(features).float(), self.logvar_head(features).float().clamp(-12, 8)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def decode(self, latent, length=None):
        """Decode [B,D,T_latent]. Crop only the right alignment pad if requested."""
        if latent.ndim != 3 or latent.shape[1] != self.latent_channels or not latent.shape[-1]:
            raise ValueError("Expected nonempty latent [B,latent_channels,T_latent]")
        capacity = latent.shape[-1] * self.downsample_factor
        if length is not None and not capacity - self.downsample_factor < length <= capacity:
            raise ValueError(f"Length {length} is incompatible with latent capacity {capacity}")
        return self.decoder(latent)[:, 0, :length]

    def forward(self, waveform, *, sample=None):
        mu, logvar = self.encode(waveform)
        if sample is None:
            sample = self.training
        z = self.reparameterize(mu, logvar) if sample else mu
        logits = self.classifier(mu.mean(dim=-1)) if self.classifier is not None else None
        return VAEOutput(self.decode(z, waveform.shape[-1]), mu, logvar, z, logits)
