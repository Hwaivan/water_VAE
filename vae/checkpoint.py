"""Versioned checkpoints with atomic replacement and epoch-boundary resume."""
import os
import random
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from .utils import build_model


FORMAT_VERSION = 1


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, config):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported VAE checkpoint version: {path}")
    saved = state["config"]
    if saved["model"] != config["model"]:
        raise ValueError("Checkpoint model configuration differs from requested model")
    if saved["data"]["sample_rate"] != config["data"]["sample_rate"]:
        raise ValueError("Checkpoint sample rate differs from requested data.sample_rate")
    return state


def load_model(path, config, device):
    state = load_checkpoint(path, config)
    model = build_model(config)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state


def capture_rng(device):
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def make_scaler(enabled):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled, init_scale=1024.0)
    return torch.cuda.amp.GradScaler(enabled=enabled, init_scale=1024.0)
