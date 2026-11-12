import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from scipy.io import wavfile

from vae.checkpoint import make_scaler
from vae.dataset import CleanAudioDataset, collate_batch, load_audio, read_manifest
from vae.engine import train_batch
from vae.losses import VAELoss, diagonal_kl
from vae.metrics import audio_metrics
from vae.utils import WORKSPACE, build_model, load_config, resolve_device


CONFIG = WORKSPACE / "configs" / "vae_smoke.yaml"


class WaveformVAETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.config = load_config(CONFIG)

    def test_variable_lengths_and_reparameterization(self):
        model = build_model(self.config)
        for batch, length in [(1, 1), (1, 16), (3, 17), (2, 803)]:
            waveform = torch.randn(batch, length)
            result = model(waveform)
            self.assertEqual(tuple(result.mu.shape), (batch, 4, (length + 15) // 16))
            self.assertEqual(result.logvar.shape, result.mu.shape)
            self.assertEqual(result.waveform.shape, waveform.shape)
            self.assertFalse(torch.equal(result.mu, result.z))
            model.eval()
            deterministic = model(waveform)
            torch.testing.assert_close(deterministic.z, deterministic.mu)
            torch.testing.assert_close(model.decode(deterministic.mu, length), deterministic.waveform)
            self.assertEqual(model.decode(deterministic.mu).shape[-1], 16 * result.mu.shape[-1])
            model.train()

    def test_manifest_label_and_resampling(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            signal = np.full(61, 8192, dtype=np.int16)
            path = root / "clean signal.wav"
            wavfile.write(path, 8000, np.stack((signal, signal // 2), axis=1))
            clean = root / "clean.list"
            clean.write_text("clean signal.wav\n", encoding="utf-8")
            paired = root / "paired.list"
            paired.write_text("nonexistent noisy.wav\tclean signal.wav\n", encoding="utf-8")
            tsv = root / "labels.tsv"
            data = self.config["data"]
            data["valid_manifest"] = str(clean)
            self.config["model"]["num_classes"] = 3
            clean_item = CleanAudioDataset(self.config, "valid")[0]
            self.assertEqual(clean_item["file_id"], "clean signal")
            self.assertEqual(clean_item["length"], 122)
            self.assertEqual(clean_item["label"], -1)
            self.assertAlmostEqual(float(clean_item["waveform"][61]), 0.1875, places=3)
            data.update(manifest_mode="paired", valid_manifest=str(paired), class_labels={"valid": str(tsv)})
            tsv.write_text("file_id\tclass_id\nnonexistent noisy\t2\n", encoding="utf-8")
            paired_item = CleanAudioDataset(self.config, "valid")[0]
            self.assertEqual(paired_item["file_id"], "nonexistent noisy")
            self.assertEqual(paired_item["label"], 2)
            torch.testing.assert_close(clean_item["waveform"], paired_item["waveform"])
            for rows, message in [
                ("clean signal\t1\n", "missing labels"),
                ("nonexistent noisy\t1\nnonexistent noisy\t2\n", "duplicate label"),
                ("nonexistent noisy\t3\n", "outside"),
            ]:
                tsv.write_text(rows, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    CleanAudioDataset(self.config, "valid")
            with self.assertRaisesRegex(ValueError, "full-audio limit"):
                load_audio(path, 16000, max_seconds=0.001)
            truncated = root / "truncated.wav"
            truncated.write_bytes(path.read_bytes()[:-12])
            with self.assertRaisesRegex(ValueError, "Cannot read WAV"):
                load_audio(truncated, 16000)
            paired.write_text("a.wav\tb.wav\nother/a.wav\tc.wav\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate file_id"):
                read_manifest(paired, "paired")

    def test_stft_baseline_and_mrstft_gradients(self):
        criterion = VAELoss(self.config)
        signal = torch.randn(2, 701)
        reference = torch.stft(signal, 512, hop_length=100, win_length=400,
                               window=torch.hann_window(400), return_complex=True)
        torch.testing.assert_close(criterion.transforms[0](signal), reference)
        torch.testing.assert_close(diagonal_kl(torch.ones(2, 3, 4), torch.zeros(2, 3, 4)),
                                   torch.full((2,), 0.5))
        self.assertEqual(criterion.beta(0), 0)
        self.assertEqual(criterion.beta(4), self.config["loss"]["kl_weight"])
        for clean in [signal, torch.zeros(2, 1)]:
            model = build_model(self.config).eval()
            prediction = model(clean)
            labels = torch.tensor([-1, -1])
            losses, _, _, labeled = criterion(prediction, clean, labels, 1)
            self.assertTrue(all(value.isfinite().all() for value in losses.values()))
            self.assertFalse(labeled.any())
            losses["vae_loss"].mean().backward()
            self.assertTrue(all(p.grad.isfinite().all() for p in model.parameters()))

    def test_metrics_no_artificial_high_scores(self):
        criterion = VAELoss(self.config)
        transform = criterion.transforms[0]
        estimate = torch.zeros(500)
        self.assertFalse(audio_metrics(estimate, estimate, transform, 1e-4, 1e-5)["valid"])
        clean = torch.sin(torch.arange(500, dtype=torch.float32))
        row = audio_metrics(estimate, clean, transform, 1e-4, 1e-5)
        self.assertEqual(row["si_snr_db"], -80)
        self.assertAlmostEqual(row["sdr_db"], 0.0)
        exact = audio_metrics(clean, clean, transform, 1e-4, 1e-5)
        self.assertEqual(exact["stft_convergence"], 0)
        self.assertEqual(exact["log_spectral_distance_db"], 0)
        dc = audio_metrics(clean, torch.ones_like(clean), transform, 1e-4, 1e-5)
        self.assertIsNone(dc["si_snr_db"])

    def _two_steps(self, device, amp, classify):
        self.config["model"].update(classification_head=classify, num_classes=3)
        self.config["loss"]["classification_weight"] = 0.2 if classify else 0.0
        model = build_model(self.config).to(device)
        criterion = VAELoss(self.config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scaler = make_scaler(device.type == "cuda" and amp)
        self.assertEqual(scaler.is_enabled(), device.type == "cuda" and amp)
        for step in range(1, 3):
            labels = [2, -1, 0] if classify and step == 1 else [-1, -1, -1]
            batch = collate_batch([
                {"waveform": 0.2 * torch.sin(torch.arange(length, dtype=torch.float32) / 9),
                 "length": length, "label": label}
                for length, label in zip([410, 410, 256], labels)
            ])
            before = model.encoder[0].weight.detach().clone()
            values, count, updated = train_batch(model, criterion, batch, optimizer, scaler,
                                                 device, step, clip_norm=1.0)
            self.assertTrue(updated)
            self.assertFalse(torch.equal(before, model.encoder[0].weight))
            self.assertEqual(count, 2 if classify and step == 1 else 0)
            self.assertTrue(np.isfinite(values["total_loss"]))
            self.assertEqual(values["classification_loss"] is None, count == 0)
            if classify:
                self.assertEqual(model.classifier.weight.grad is None, count == 0)

    def test_cpu_with_and_without_amp_and_optional_head(self):
        device = resolve_device("cpu", self.config)
        for amp, classify in [(False, False), (True, False), (True, True)]:
            self._two_steps(device, amp, classify)
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA requested but unavailable"):
                resolve_device("cuda", self.config)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_with_and_without_amp(self):
        device = resolve_device("cuda:0", self.config)
        for amp in (False, True):
            self._two_steps(device, amp, classify=True)


if __name__ == "__main__":
    unittest.main()
