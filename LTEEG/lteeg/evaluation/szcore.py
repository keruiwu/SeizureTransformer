"""Self-contained SzCORE scoring (Dan et al., Epilepsia 2024).

Re-implements ``timescoring`` 0.0.x (``SampleScoring`` / ``EventScoring``) with the
exact same semantics, including its rounding (Python ``round``, half-to-even) and
resampling conventions, and is cross-checked against the reference package in
``tests/test_szcore.py``. Extras on top of the counts: per-reference-event detection
details (latency, coverage) and the list of false-alarm events.

Sample-based scoring
    Reference/hypothesis events are rasterized at ``sample_fs`` (1 Hz by default) and
    compared sample by sample.

Event-based scoring (defaults = SzCORE recommendations)
    1. events closer than ``min_duration_between_events`` (90 s) are merged;
    2. events longer than ``max_event_duration`` (300 s) are split;
    3. a reference event extended by ``tolerance_start`` (30 s) before and
       ``tolerance_end`` (60 s) after is detected if any hypothesis overlaps it
       (relative overlap > ``min_overlap``);
    4. a hypothesis event that overlaps no *detected* extended reference is a false
       positive. Everything runs on a 10 Hz grid.

Metrics: sensitivity = TP/ref, precision = TP/(TP+FP), F1 = 2TP/(2TP+FP+FN),
false positives per 24 h. Undefined ratios are NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..config import ScoringConfig
from ..inference.postprocess import runs

Events = List[Tuple[float, float]]


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    ref_true: int = 0
    num_samples: int = 0
    fs: float = 1.0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.tp + other.tp, self.fp + other.fp, self.ref_true + other.ref_true,
                      self.num_samples + other.num_samples, other.fs if other.num_samples else self.fs)

    @property
    def hours(self) -> float:
        return self.num_samples / self.fs / 3600 if self.fs else 0.0

    @property
    def sensitivity(self) -> float:
        return self.tp / self.ref_true if self.ref_true > 0 else float("nan")

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else float("nan")

    @property
    def f1(self) -> float:
        if self.ref_true + self.fp == 0:
            return float("nan")
        return 2 * self.tp / (2 * self.tp + self.fp + (self.ref_true - self.tp))

    @property
    def fp_per_day(self) -> float:
        days = self.num_samples / self.fs / 3600 / 24 if self.fs else 0.0
        return self.fp / days if days > 0 else float("nan")

    def scores(self) -> Dict[str, float]:
        return {"sensitivity": self.sensitivity, "precision": self.precision, "f1": self.f1,
                "fp_per_day": self.fp_per_day}


# ----------------------------------------------------------------------------- helpers
def events_from_mask(mask: np.ndarray, fs: float) -> Events:
    """Runs of a binary mask -> [(start_s, end_s)] (end exclusive), like timescoring.Annotation."""
    s, e = runs(mask)
    return [(int(a) / fs, int(b) / fs) for a, b in zip(s, e)]


def mask_from_events(events: Sequence[Tuple[float, float]], fs: float, n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for start, end in events:
        mask[max(round(start * fs), 0):max(round(end * fs), 0)] = True
    return mask


def resampled_count(n_native: int, fs_native: float, fs: float) -> int:
    return round(n_native / fs_native * fs)


def merge_events(events: Events, min_gap: float) -> Events:
    merged = list(events)
    i = 1
    while i < len(merged):
        if merged[i][0] - merged[i - 1][1] < min_gap:
            merged[i - 1] = (merged[i - 1][0], merged[i][1])
            del merged[i]
            i -= 1
        i += 1
    return merged


def split_events(events: Events, max_duration: float) -> Events:
    out: Events = []
    for start, end in events:
        while end - start > max_duration:
            out.append((start, start + max_duration))
            start = start + max_duration
        out.append((start, end))
    return out


# ----------------------------------------------------------------------------- scoring
def sample_scoring(ref: Events, hyp: Events, n_native: int, fs_native: float, fs: float = 1.0) -> Counts:
    n = resampled_count(n_native, fs_native, fs)
    r = mask_from_events(ref, fs, n)
    h = mask_from_events(hyp, fs, n)
    return Counts(int(np.sum(r & h)), int(np.sum(~r & h)), int(np.sum(r)), n, fs)


def event_scoring(ref: Events, hyp: Events, n_native: int, fs_native: float,
                  params: ScoringConfig) -> Tuple[Counts, List[dict], List[dict]]:
    """Return (counts, per-reference-event details, false-positive events)."""
    fs = params.event_fs
    n = resampled_count(n_native, fs_native, fs)
    ref_ev = split_events(merge_events(list(ref), params.min_duration_between_events), params.max_event_duration)
    hyp_ev = split_events(merge_events(list(hyp), params.min_duration_between_events), params.max_event_duration)
    hyp_mask = mask_from_events(hyp_ev, fs, n)
    duration = n / fs

    tp = 0
    tp_mask = np.zeros(n, dtype=bool)
    details: List[dict] = []
    for start, end in ref_ev:
        es, ee = max(0, start - params.tolerance_start), min(duration, end + params.tolerance_end)
        a, b = round(es * fs), round(ee * fs)
        seg = hyp_mask[a:b]
        rel_overlap = (np.sum(seg) / fs) / (ee - es)
        hit = bool(rel_overlap > params.min_overlap + 1e-6)
        latency = float("nan")
        if hit:
            tp += 1
            tp_mask[a:b] = True
            latency = (a + int(np.argmax(seg))) / fs - start
        ra, rb = round(start * fs), round(end * fs)
        coverage = float(np.mean(hyp_mask[ra:rb])) if rb > ra else float("nan")
        details.append({"ref_start": start, "ref_end": end, "detected": hit,
                        "latency_sec": latency, "coverage": coverage})

    fp = 0
    fp_events: List[dict] = []
    for start, end in hyp_ev:
        if not tp_mask[round(start * fs):round(end * fs)].any():
            fp += 1
            fp_events.append({"start": start, "end": end, "duration": end - start})
    return Counts(tp, fp, len(ref_ev), n, fs), details, fp_events
