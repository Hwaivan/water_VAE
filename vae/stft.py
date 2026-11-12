"""Float32 STFT shared by spectral training losses and evaluation."""
import torch
from torch import nn
from torch.nn import functional as F


class STFT(nn.Module):
    def __init__(
        self, n_fft, win_length, hop_length, window="hann", center=True,
        normalized=False, onesided=True,
    ):
        super().__init__()
        if window != "hann" or not 1 <= hop_length <= win_length <= n_fft or n_fft < 2:
            raise ValueError("STFT requires Hann and 1 <= hop_length <= win_length <= n_fft")
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, waveform):
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            waveform = waveform.float()
            length = waveform.shape[-1]
            if not self.center and length < self.n_fft:
                waveform = F.pad(waveform, (0, self.n_fft - length))
            return torch.stft(
                waveform, n_fft=self.n_fft, hop_length=self.hop_length,
                win_length=self.win_length, window=self.window.float(),
                center=self.center, normalized=self.normalized, onesided=self.onesided,
                pad_mode="reflect" if length > self.n_fft // 2 else "constant",
                return_complex=True,
            )
