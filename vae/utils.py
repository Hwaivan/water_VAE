"""Configuration, device selection and output helpers."""
import argparse
import json
import math
import random
from pathlib import Path

import torch
import yaml

from .model import WaveformVAE


WORKSPACE = Path(__file__).resolve().parents[1]


def workspace_path(value):
    return (WORKSPACE / Path(value).expanduser()).resolve()


def load_config(path):
    with Path(path).expanduser().open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    data, loss, training = (config[key] for key in ("data", "loss", "training"))
    for key in ("train_manifest", "valid_manifest", "test_manifest", "audio_root"):
        if data.get(key):
            data[key] = str(workspace_path(data[key]))
    data["class_labels"] = {
        split: str(workspace_path(path)) if path else None
        for split, path in (data.get("class_labels") or {}).items()
    }
    experiment = config["experiment"]
    experiment["output_dir"] = str(workspace_path(experiment["output_dir"]))
    rate = data["sample_rate"]
    if not isinstance(rate, int) or rate < 1 or data["num_workers"] < 0:
        raise ValueError("Use a positive integer sample rate and nonnegative num_workers")
    for key in ("segment_seconds", "max_full_seconds"):
        seconds = data[key]
        if key == "segment_seconds" and seconds is None:
            continue
        if not math.isfinite(seconds) or seconds <= 0 or round(seconds * rate) < 1:
            raise ValueError(f"data.{key} must describe at least one sample")
    for key in ("waveform_weight", "mrstft_weight", "kl_weight", "classification_weight"):
        if not math.isfinite(loss[key]) or loss[key] < 0:
            raise ValueError(f"loss.{key} must be finite and nonnegative")
    if not loss["waveform_weight"] + loss["mrstft_weight"] or not loss["stft_resolutions"]:
        raise ValueError("Configure a reconstruction weight and at least one STFT resolution")
    if not math.isfinite(loss["spectral_eps"]) or loss["spectral_eps"] <= 0:
        raise ValueError("loss.spectral_eps must be positive and finite")
    if loss["kl_warmup_steps"] < 0:
        raise ValueError("loss.kl_warmup_steps must be nonnegative")
    if loss["classification_weight"] and not config["model"].get("classification_head"):
        raise ValueError("Positive classification_weight requires model.classification_head")
    if training["epochs"] < 1 or training["batch_size"] < 1:
        raise ValueError("training.epochs and training.batch_size must be positive")
    for key in ("silence_rms", "collapse_kl_threshold"):
        if not math.isfinite(config["metrics"][key]) or config["metrics"][key] <= 0:
            raise ValueError(f"metrics.{key} must be positive and finite")
    return config


def resolve_device(value, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable in this PyTorch environment")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"Unavailable CUDA device: {device}")
    elif device.type == "cpu":
        threads = config["training"].get("cpu_threads")
        if threads is not None:
            torch.set_num_threads(threads)
    else:
        raise ValueError("Device must be auto, cpu, cuda or cuda:N")
    return device


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def build_model(config):
    model = dict(config["model"])
    if model.pop("type") != "waveform_vae":
        raise ValueError("model.type must be waveform_vae")
    return WaveformVAE(**model)


def add_runtime_args(parser: argparse.ArgumentParser, *, checkpoint=False):
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    if checkpoint:
        parser.add_argument("--checkpoint", type=Path, required=True)


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path, record):
    Path(path).write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
