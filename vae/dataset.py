"""WAV input, clean targets and strict optional class labels."""
import math
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


@dataclass(frozen=True)
class ManifestEntry:
    file_id: str
    clean_path: Path


@dataclass(frozen=True)
class Audio:
    waveform: torch.Tensor
    source_rate: int
    source_length: int


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            row = line.rstrip("\r\n")
            if row.strip() and not row.lstrip().startswith("#"):
                yield line_number, row


def read_manifest(path, mode, audio_root=None):
    path = Path(path).expanduser()
    if mode not in ("clean", "paired"):
        raise ValueError("data.manifest_mode must be clean or paired")
    root = Path(audio_root) if audio_root else path.resolve().parent
    entries, ids = [], set()
    for line, row in read_rows(path):
        if mode == "paired":
            fields = row.split("\t") if "\t" in row else row.split()
        else:
            fields = row.split("\t")
        fields = [field.strip() for field in fields]
        if len(fields) != (2 if mode == "paired" else 1) or not all(fields):
            raise ValueError(f"{path}:{line}: expected a {mode} manifest row")
        file_id = Path(fields[0]).stem
        if not file_id or file_id in ids:
            raise ValueError(f"{path}:{line}: empty or duplicate file_id {file_id!r}")
        ids.add(file_id)
        entries.append(ManifestEntry(file_id, (root / Path(fields[-1]).expanduser()).resolve()))
    if not entries:
        raise ValueError(f"Manifest contains no samples: {path}")
    return entries


def read_labels(path, entries, num_classes):
    if num_classes is None or num_classes < 2:
        raise ValueError("Class TSV requires model.num_classes >= 2")
    labels = {}
    for index, (line, row) in enumerate(read_rows(path)):
        fields = [field.strip() for field in row.split("\t")]
        if len(fields) != 2:
            raise ValueError(f"{path}:{line}: expected file_id<TAB>class_id")
        file_id, value = fields
        if index == 0 and fields == ["file_id", "class_id"]:
            continue
        if not file_id or file_id in labels:
            raise ValueError(f"{path}:{line}: empty or duplicate label ID {file_id!r}")
        try:
            class_id = int(value)
        except ValueError as exc:
            raise ValueError(f"{path}:{line}: class_id must be an integer") from exc
        if not 0 <= class_id < num_classes:
            raise ValueError(f"{path}:{line}: class_id {class_id} outside [0, {num_classes})")
        labels[file_id] = class_id
    missing = [entry.file_id for entry in entries if entry.file_id not in labels]
    if missing:
        raise ValueError(f"{path}: missing labels for {len(missing)} IDs, e.g. {missing[:5]}")
    return labels


def resample_waveform(waveform, source_rate, target_rate):
    if source_rate == target_rate:
        return waveform
    divisor = math.gcd(source_rate, target_rate)
    values = resample_poly(waveform.numpy(), target_rate // divisor, source_rate // divisor)
    length = max(1, round(waveform.numel() * target_rate / source_rate))
    return torch.from_numpy(np.ascontiguousarray(values[:length], dtype=np.float32))


def load_audio(path, sample_rate, max_seconds=None):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message="Reached EOF prematurely|Incomplete chunk ID",
                category=wavfile.WavFileWarning,
            )
            source_rate, values = wavfile.read(str(path))
    except (OSError, ValueError, EOFError, NotImplementedError, struct.error,
            UnboundLocalError, wavfile.WavFileWarning) as exc:
        raise ValueError(f"Cannot read WAV {path}: {exc}") from exc
    source_length = len(values)
    if not source_length or source_rate <= 0:
        raise ValueError(f"Empty audio or invalid sampling rate: {path}")
    if max_seconds is not None and source_length > max_seconds * source_rate:
        raise ValueError(f"{path}: audio exceeds the {max_seconds:g} s full-audio limit")
    if values.dtype.kind == "u":
        midpoint = 2 ** (8 * values.dtype.itemsize - 1)
        values = (values.astype(np.float32) - midpoint) / midpoint
    elif values.dtype.kind == "i":
        scale = 2 ** (8 * values.dtype.itemsize - 1)
        values = values.astype(np.float32) / scale
    else:
        values = values.astype(np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1)
    waveform = resample_waveform(
        torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)), source_rate, sample_rate
    )
    if not torch.isfinite(waveform).all():
        raise ValueError(f"Non-finite audio: {path}")
    return Audio(waveform, source_rate, source_length)


def write_audio(path, waveform, sample_rate):
    values = waveform.detach().float().cpu().numpy()
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Cannot write invalid mono waveform: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sample_rate, values)


class CleanAudioDataset(Dataset):
    def __init__(self, config, split, *, full_audio=False, report_invalid=False):
        data = config["data"]
        manifest = data.get(f"{split}_manifest")
        if not manifest:
            raise ValueError(f"Set data.{split}_manifest in the YAML")
        self.entries = read_manifest(manifest, data["manifest_mode"], data.get("audio_root"))
        label_path = data.get("class_labels", {}).get(split)
        self.labels = (
            read_labels(label_path, self.entries, config["model"].get("num_classes"))
            if label_path else {}
        )
        self.sample_rate = data["sample_rate"]
        self.training = split == "train"
        self.segment_length = None
        if not full_audio and (self.training or data["segment_validation"]):
            seconds = data["segment_seconds"]
            self.segment_length = round(seconds * self.sample_rate) if seconds is not None else None
        self.max_seconds = data["max_full_seconds"] if self.segment_length is None else None
        self.report_invalid = report_invalid
        self.seed = config["experiment"]["seed"]
        self.epoch = 0

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        item = {"file_id": entry.file_id, "path": str(entry.clean_path)}
        try:
            audio = load_audio(entry.clean_path, self.sample_rate, self.max_seconds)
        except ValueError as exc:
            if not self.report_invalid:
                raise
            return {**item, "error": str(exc)}
        waveform = audio.waveform
        if self.segment_length is not None:
            excess = max(0, waveform.numel() - self.segment_length)
            start = excess // 2
            if self.training and excess:
                generator = torch.Generator().manual_seed(self.seed + self.epoch * len(self) + index)
                start = int(torch.randint(excess + 1, (), generator=generator))
            waveform = waveform[start:start + self.segment_length]
        return {**item, "waveform": waveform, "length": waveform.numel(),
                "label": self.labels.get(entry.file_id, -1)}


def collate_batch(items):
    valid = [item for item in items if "error" not in item]
    return {
        "items": valid,
        "errors": [item for item in items if "error" in item],
        "waveform": pad_sequence([item["waveform"] for item in valid], batch_first=True) if valid else None,
        "labels": torch.tensor([item["label"] for item in valid], dtype=torch.long),
    }


def make_loader(dataset, config, device, epoch=0):
    dataset.epoch = epoch
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=dataset.training,
        generator=torch.Generator().manual_seed(dataset.seed + epoch),
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )


def length_groups(batch, device):
    """Trim batch padding before convolution so it cannot affect GroupNorm."""
    groups = {}
    for index, item in enumerate(batch["items"]):
        groups.setdefault(item["length"], []).append(index)
    for length, indices in groups.items():
        waveform = batch["waveform"][indices, :length].to(device)
        labels = batch["labels"][indices].to(device)
        yield indices, waveform, labels
