"""Probability trace -> binary event mask -> list of events.

Steps (all configurable in ``postprocess.*``):

1. threshold: ``prob > threshold`` (strict, as in the original code);
2. morphological ``opening`` then ``closing`` with a flat element of ``morph_kernel``
   samples. In 1-D these are exact run-length operations: opening removes positive
   runs shorter than k, closing fills interior gaps shorter than k. (Identical to
   ``scipy.ndimage.binary_opening/closing`` away from the array borders; scipy's
   ``border_value=0`` additionally erodes runs touching the borders, which we avoid.)
3. remove events shorter than ``min_duration_sec`` (``int(min_duration_sec * fs)``
   samples, as the original);
4. optionally merge events separated by gaps shorter than ``merge_gap_sec``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..config import PostprocessConfig


def runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Start and end (exclusive) indices of the True runs of a 1-D boolean mask."""
    m = np.asarray(mask, dtype=bool)
    d = np.diff(np.concatenate(([False], m, [False])).astype(np.int8))
    return np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]


def _cover(n: int, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Boolean mask covering the union of [starts, ends) intervals (vectorized)."""
    delta = np.zeros(n + 1, dtype=np.int64)
    np.add.at(delta, starts, 1)
    np.add.at(delta, ends, -1)
    return np.cumsum(delta[:-1]) > 0


def remove_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if min_len <= 1:
        return mask.copy()
    s, e = runs(mask)
    short = (e - s) < min_len
    return mask & ~_cover(len(mask), s[short], e[short])


def fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill False gaps shorter than ``max_gap`` that lie between two True runs."""
    mask = np.asarray(mask, dtype=bool)
    if max_gap <= 1:
        return mask.copy()
    s, e = runs(mask)
    if len(s) < 2:
        return mask.copy()
    gap_start, gap_end = e[:-1], s[1:]
    short = (gap_end - gap_start) < max_gap
    return mask | _cover(len(mask), gap_start[short], gap_end[short])


def postprocess(prob: np.ndarray, fs: float, cfg: PostprocessConfig) -> np.ndarray:
    mask = np.asarray(prob) > cfg.threshold
    for op in cfg.morph_ops:
        if op == "opening":
            mask = remove_short_runs(mask, cfg.morph_kernel)
        elif op == "closing":
            mask = fill_short_gaps(mask, cfg.morph_kernel)
        else:
            raise ValueError(f"unknown morphological op {op!r}")
    if cfg.min_duration_sec > 0:
        mask = remove_short_runs(mask, int(cfg.min_duration_sec * fs))
    if cfg.merge_gap_sec > 0:
        mask = fill_short_gaps(mask, int(round(cfg.merge_gap_sec * fs)))
    return mask


def mask_to_events(mask: np.ndarray, fs: float) -> List[Tuple[float, float]]:
    s, e = runs(mask)
    return [(a / fs, b / fs) for a, b in zip(s.tolist(), e.tolist())]
