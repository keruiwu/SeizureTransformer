"""Threshold-free metrics on a coarse time grid.

Per-sample probabilities are averaged into bins at ``evaluation.auc_fs`` (1 Hz by
default, the SzCORE sample-scoring grid) and compared with the reference rasterized
on the same grid. AUROC and average precision (AUPRC) characterize the probability
trace independently of threshold and post-processing; under extreme class
imbalance AUPRC is the more informative of the two.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def bin_mean(values: np.ndarray, fs: float, out_fs: float) -> np.ndarray:
    n = len(values)
    n_out = round(n / fs * out_fs)
    edges = np.minimum(np.round(np.arange(n_out + 1) * (fs / out_fs)).astype(np.int64), n)
    c = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    width = edges[1:] - edges[:-1]
    out = (c[edges[1:]] - c[edges[:-1]]) / np.maximum(width, 1)
    empty = width == 0
    if empty.any():  # bins past the end (rounding) repeat the last value
        out[empty] = values[-1] if n else 0.0
    return out.astype(np.float32)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(scores, dtype=np.float64))
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Step-wise AP over distinct thresholds (same definition as scikit-learn)."""
    y = np.asarray(labels, dtype=bool)
    if y.sum() == 0:
        return float("nan")
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    idx = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(y)[idx]
    fps = idx + 1 - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
