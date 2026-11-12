"""Generate multitone WAVs and exercise all VAE commands, including exact CPU resume."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml
from scipy.io import wavfile

from vae.utils import WORKSPACE, load_config


def run_smoke(root, device):
    data = root / "data"
    data.mkdir()
    manifests = {}
    for split, sizes in {"train": [1007, 711, 990, 563], "valid": [889], "test": [707]}.items():
        files = []
        for index, size in enumerate(sizes):
            rate = 8000 if split == "test" else 16000
            t = np.arange(size) / rate
            signal = 0.28 * np.sin(2 * np.pi * (231 + 37 * index) * t)
            signal += 0.14 * np.sin(2 * np.pi * 963 * t)
            name = f"{split} {index}.wav"
            wavfile.write(data / name, rate, signal.astype(np.float32))
            files.append(name)
        manifest = data / f"{split}.list"
        manifest.write_text("\n".join(files) + "\n", encoding="utf-8")
        manifests[split] = manifest
    config = load_config(WORKSPACE / "configs" / "vae_smoke.yaml")
    run = root / "run"
    config["experiment"]["output_dir"] = str(run)
    for split, path in manifests.items():
        config["data"][f"{split}_manifest"] = str(path)
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def call(script, *arguments):
        subprocess.run(
            [sys.executable, "-B", str(WORKSPACE / script), "--config", str(config_path),
             "--device", device, *map(str, arguments)], check=True, cwd=root,
        )

    call("train.py")
    last = run / "checkpoints" / "last.pt"
    first = torch.load(last, weights_only=True, map_location="cpu")
    assert first["epoch"] == 1 and first["global_step"] == 2
    call("train.py", "--resume", last, "--epochs", 2)
    resumed = torch.load(last, weights_only=True, map_location="cpu")
    assert resumed["epoch"] == 2 and resumed["global_step"] == 4
    assert resumed["optimizer"]["state"]
    if device == "cpu":
        reference = root / "uninterrupted"
        call("train.py", "--epochs", 2, "--output-dir", reference)
        continuous = torch.load(reference / "checkpoints" / "last.pt", weights_only=True)
        for key, tensor in resumed["model"].items():
            torch.testing.assert_close(tensor, continuous["model"][key], rtol=0, atol=0)

    best = run / "checkpoints" / "best.pt"
    source, output = data / "test 0.wav", root / "reconstructed.wav"
    evaluation = root / "evaluation"
    eval_manifest = data / "test_mixed.list"
    wavfile.write(data / "silence.wav", 16000, np.zeros(128, dtype=np.float32))
    (data / "broken.wav").write_bytes(b"not a WAV")
    eval_manifest.write_text("test 0.wav\nsilence.wav\nbroken.wav\n", encoding="utf-8")
    call("evaluate.py", "--checkpoint", best, "--full-audio",
         "--manifest", eval_manifest, "--output-dir", evaluation)
    summary = json.loads((evaluation / "summary.json").read_text(encoding="utf-8"))
    assert summary["valid_count"] == 1 and summary["invalid_count"] == 2
    assert summary["si_snr_count"] == 1
    for key in ("waveform_l1", "kl", "sdr_db", "si_snr_db", "log_spectral_distance_db", "stft_convergence"):
        assert np.isfinite(summary["means"][key])

    output_again = root / "reconstructed_again.wav"
    for path in (output, output_again):
        call("infer.py", "--checkpoint", best, "--input", source, "--output", path)
    rate, waveform = wavfile.read(output)
    assert rate == 8000 and len(waveform) == 707 and np.isfinite(waveform).all()
    np.testing.assert_array_equal(waveform, wavfile.read(output_again)[1])
    call("export_latents.py", "--checkpoint", best, "--manifest", manifests["test"],
         "--output-dir", root / "test_latents")
    exported = torch.load(root / "test_latents" / "000000.pt", weights_only=True)
    assert exported["file_id"] == "test 0" and exported["waveform_length"] == 1414
    assert tuple(exported["mu"].shape) == (4, 89)
    assert not (root / "test_latents" / "latent_stats.json").exists()
    call("export_latents.py", "--checkpoint", best, "--fit-train-stats",
         "--output-dir", root / "train_latents")
    fitted = json.loads((root / "train_latents" / "latent_stats.json").read_text(encoding="utf-8"))
    train_mu = torch.cat([
        torch.load(path, weights_only=True)["mu"]
        for path in sorted((root / "train_latents").glob("*.pt"))
    ], dim=1).double()
    assert fitted["frame_count"] == train_mu.shape[1] and fitted["split"] == "train"
    np.testing.assert_allclose(fitted["mean"], train_mu.mean(dim=1).numpy())
    np.testing.assert_allclose(fitted["std"], train_mu.std(dim=1, correction=0).numpy())
    assert fitted["checkpoint"]["sha256"] == exported["checkpoint"]["sha256"]
    print(f"VAE smoke test passed on {device}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, help="Keep generated inputs and outputs here")
    args = parser.parse_args()
    if args.output_dir:
        root = args.output_dir.resolve()
        root.mkdir(parents=True, exist_ok=False)
        run_smoke(root, args.device)
    else:
        with TemporaryDirectory(prefix="water_vae_smoke_") as directory:
            run_smoke(Path(directory), args.device)


if __name__ == "__main__":
    main()
