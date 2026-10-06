"""PyTorch datasets over cached recordings."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import Config
from .annotations import intervals_to_labels
from .preprocess import Pipeline
from .store import Recording


def read_window(signal: np.ndarray, start: int, length: int) -> Tuple[np.ndarray, int]:
    """Copy ``signal[:, start:start+length]`` as float32, zero-padding past the end.
    Returns the window and the number of valid (non-padded) samples."""
    n = signal.shape[1]
    stop = min(start + length, n)
    valid = max(0, stop - start)
    out = np.zeros((signal.shape[0], length), dtype=np.float32)
    if valid > 0:
        out[:, :valid] = signal[:, start:stop]
    return out, valid


class TrainWindowDataset(Dataset):
    """Map-style dataset indexed by ``(rec_idx, start)`` tuples coming from
    :class:`lteeg.data.windows.EpochSampler`.

    Returns ``x`` float32 ``(C, W)``, ``y`` int64 ``(W)`` (class per sample, -1 =
    ignore), and integer metadata (``rec``, ``patient``, ``start``) that models may use
    (e.g. patient-conditioned or test-time-adaptive models).
    """

    def __init__(self, cfg: Config, recordings: Sequence[Recording], source, patients: Sequence[str]):
        self.cfg = cfg
        self.recordings: List[Recording] = list(recordings)
        self.source = source
        self.window = cfg.window_samples
        self.fs = cfg.fs
        self.intervals = [r.intervals(self.fs) for r in self.recordings]
        if cfg.task.mode == "binary":
            # every event class collapses to "seizure" = 1
            self.intervals = [np.column_stack([iv[:, :2], np.ones(len(iv), dtype=np.int64)]) if len(iv) else iv
                              for iv in self.intervals]
        self.ignore_margin = int(round(cfg.task.ignore_boundary_sec * self.fs))
        pid = {p: k for k, p in enumerate(patients)}
        self.patient_idx = [pid.get(r.patient, -1) for r in self.recordings]
        self.window_pipeline = Pipeline(cfg.preprocess.window, self.fs) if cfg.preprocess.window else None

    def __len__(self) -> int:  # pragma: no cover - length comes from the sampler
        raise TypeError("TrainWindowDataset is indexed by (rec, start) items from EpochSampler")

    def __getitem__(self, item: Tuple[int, int]) -> Dict[str, Any]:
        rec_idx, start = int(item[0]), int(item[1])
        sig = self.source.open(self.recordings[rec_idx])
        x, valid = read_window(sig, start, self.window)
        y = intervals_to_labels(self.intervals[rec_idx], start, self.window, self.ignore_margin)
        if valid < self.window:
            y[valid:] = -1
        if self.window_pipeline is not None:
            x = self.window_pipeline(x).astype(np.float32)
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "rec": rec_idx,
            "patient": self.patient_idx[rec_idx],
            "start": start,
        }
