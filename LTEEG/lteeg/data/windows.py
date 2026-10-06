"""Window indexing and training-window sampling.

Training windows lie on a grid (``windows.train_stride_sec``) over every training
recording and are categorized by their number of event samples:

* ``background`` - no event sample,
* ``full``       - every sample is an event sample,
* ``boundary``   - partially covered (contains an onset and/or offset).

The ``balanced`` sampler keeps ``boundary_ratio`` of the boundary windows and draws
``full_ratio`` x and ``background_ratio`` x as many full / background windows
(never more than exist), the scheme of the original SeizureTransformer
``get_dataset.py``. Options that are off by default:

* ``redraw_every_epoch`` - re-draw the full/background subsets every epoch, so the
  model sees far more of the (huge) background over training;
* ``group_by: patient`` - apply the quotas within each patient so seizure-rich
  patients do not dominate;
* ``jitter_sec`` - shift each selected window by a random offset (temporal
  augmentation; the category of a window may change slightly).

Selection depends only on (seed, epoch) and the naturally sorted recording list,
so it is reproducible across machines and independent of DataLoader workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config
from .annotations import positive_counts
from .store import Recording

BACKGROUND, BOUNDARY, FULL = 0, 1, 2
CATEGORY_NAMES = {BACKGROUND: "background", BOUNDARY: "boundary", FULL: "full"}


def grid_starts(n_samples: int, window: int, stride: int, include_tail: bool = False) -> np.ndarray:
    """Start indices of all complete windows on a stride grid (optionally plus one
    window aligned to the recording end)."""
    if n_samples < window:
        return np.zeros(0, dtype=np.int64)
    starts = np.arange(0, n_samples - window + 1, stride, dtype=np.int64)
    if include_tail and starts[-1] + window < n_samples:
        starts = np.append(starts, n_samples - window)
    return starts


@dataclass
class WindowIndex:
    rec: np.ndarray  # int32, index into the recording list
    start: np.ndarray  # int64, first sample
    positive: np.ndarray  # int64, event samples inside the window
    category: np.ndarray  # int8
    window: int

    def __len__(self) -> int:
        return len(self.rec)

    def counts(self, mask: Optional[np.ndarray] = None) -> Dict[str, int]:
        cat = self.category if mask is None else self.category[mask]
        return {CATEGORY_NAMES[k]: int(np.sum(cat == k)) for k in (BOUNDARY, FULL, BACKGROUND)}


def build_window_index(recordings: Sequence[Recording], fs: float, window: int, stride: int,
                       include_tail: bool = False) -> WindowIndex:
    recs, starts, pos = [], [], []
    for i, rec in enumerate(recordings):
        n = rec.n_samples_at(fs)
        s = grid_starts(n, window, stride, include_tail)
        if len(s) == 0:
            continue
        recs.append(np.full(len(s), i, dtype=np.int32))
        starts.append(s)
        pos.append(positive_counts(rec.intervals(fs), s, window))
    if not recs:
        raise ValueError("No training window fits into any recording (all shorter than the window?)")
    rec_a, start_a, pos_a = np.concatenate(recs), np.concatenate(starts), np.concatenate(pos)
    cat = np.full(len(rec_a), BOUNDARY, dtype=np.int8)
    cat[pos_a == 0] = BACKGROUND
    cat[pos_a == window] = FULL
    return WindowIndex(rec_a, start_a, pos_a, cat, window)


class WindowSampler:
    """Produces the (recording, start) list for each training epoch."""

    def __init__(self, cfg: Config, recordings: Sequence[Recording], index: WindowIndex):
        self.cfg = cfg
        self.s = cfg.sampling
        self.recordings = list(recordings)
        self.index = index
        self.seed = cfg.sampling_seed
        self.window = index.window
        self.n_samples = np.array([r.n_samples_at(cfg.fs) for r in self.recordings], dtype=np.int64)
        patients = sorted({r.patient for r in self.recordings})
        pid = {p: k for k, p in enumerate(patients)}
        self.patient_of_rec = np.array([pid[r.patient] for r in self.recordings], dtype=np.int32)
        self._fixed: Optional[np.ndarray] = None
        if not self.s.redraw_every_epoch:
            self._fixed = self._select(np.random.default_rng(self.seed))

    # -------------------------------------------------------------- selection
    def _quota_select(self, rng: np.random.Generator, members: np.ndarray) -> List[np.ndarray]:
        cat = self.index.category[members]
        boundary = members[cat == BOUNDARY]
        full = members[cat == FULL]
        background = members[cat == BACKGROUND]
        n_b = int(len(boundary) * self.s.boundary_ratio)
        boundary = boundary if n_b == len(boundary) else rng.permutation(boundary)[:n_b]
        n_f = min(int(n_b * self.s.full_ratio), len(full))
        n_g = min(int(n_b * self.s.background_ratio), len(background))
        return [boundary, rng.permutation(full)[:n_f], rng.permutation(background)[:n_g]]

    def _select(self, rng: np.random.Generator) -> np.ndarray:
        all_idx = np.arange(len(self.index))
        if self.s.name == "all":
            return all_idx
        if self.s.group_by == "patient":
            parts: List[np.ndarray] = []
            rec_patient = self.patient_of_rec[self.index.rec]
            for p in np.unique(rec_patient):
                parts.extend(self._quota_select(rng, all_idx[rec_patient == p]))
        else:
            parts = self._quota_select(rng, all_idx)
        return np.sort(np.concatenate(parts)) if parts else all_idx[:0]

    def selection(self, epoch: int) -> np.ndarray:
        if self._fixed is not None:
            return self._fixed
        return self._select(np.random.default_rng([self.seed, epoch]))

    def epoch_items(self, epoch: int) -> List[Tuple[int, int]]:
        """Shuffled ``(rec_idx, start)`` pairs for one epoch (deterministic in seed/epoch)."""
        sel = self.selection(epoch)
        rng = np.random.default_rng([self.seed, epoch, 1])
        order = rng.permutation(len(sel))
        sel = sel[order]
        rec = self.index.rec[sel].astype(np.int64)
        start = self.index.start[sel].copy()
        if self.s.jitter_sec > 0:
            j = int(round(self.s.jitter_sec * self.cfg.fs))
            start = start + rng.integers(-j, j + 1, size=len(start))
            start = np.clip(start, 0, self.n_samples[rec] - self.window)
        return list(zip(rec.tolist(), start.tolist()))

    def __len__(self) -> int:
        return len(self.selection(0))

    def summary(self) -> Dict[str, object]:
        sel = self.selection(0)
        mask = np.zeros(len(self.index), dtype=bool)
        mask[sel] = True
        pos_frac = float(self.index.positive[sel].sum() / max(1, len(sel) * self.window))
        per_patient: Dict[str, Dict[str, int]] = {}
        names = sorted({r.patient for r in self.recordings})
        rec_patient = self.patient_of_rec[self.index.rec]
        for k, p in enumerate(names):
            m = rec_patient == k
            per_patient[p] = {"available": self.index.counts(m), "selected": self.index.counts(m & mask)}
        return {
            "available": self.index.counts(),
            "selected_epoch0": self.index.counts(mask),
            "windows_per_epoch": int(len(sel)),
            "positive_sample_fraction_epoch0": pos_frac,
            "redraw_every_epoch": self.s.redraw_every_epoch,
            "per_patient": per_patient,
        }


class EpochSampler:
    """torch ``Sampler`` yielding ``(rec_idx, start)`` items. It runs in the main
    process, so per-epoch re-drawing also works with persistent DataLoader workers."""

    def __init__(self, sampler: WindowSampler):
        self.sampler = sampler
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        return iter(self.sampler.epoch_items(self.epoch))

    def __len__(self) -> int:
        return len(self.sampler.selection(self.epoch))
