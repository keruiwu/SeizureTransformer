"""Continuous inference over complete long-term recordings.

A recording of N samples is covered by windows of the training length W placed every
``hop`` samples. Per-sample probabilities from overlapping windows are averaged
(uniformly or with a Hann taper that trusts window centres more than edges, where
the model lacks context). The end of the recording is handled either by

* ``tail: align`` - one extra window ending exactly at N (no padding; default), or
* ``tail: pad``   - zero-padding the last window (the original SeizureTransformer
  behaviour; zeros are out-of-distribution input for the model).

``hop == W`` reproduces the original non-overlapping inference. Memory use is O(N)
for the output and O(batch x W) for the model, independent of recording length.
"""

from __future__ import annotations

import contextlib
from typing import Iterator, List, Optional, Sequence

import numpy as np
import torch

from ..data.datasets import read_window
from ..data.preprocess import Pipeline
from ..models.base import as_output
from ..task import Task, align_logits


def plan_windows(n: int, window: int, hop: int, tail: str = "align") -> List[int]:
    if n <= window:
        return [0]
    starts = list(range(0, n - window + 1, hop))
    if starts[-1] + window < n:
        starts.append(n - window if tail == "align" else starts[-1] + hop)
    return starts


def autocast_context(device: torch.device, amp: str):
    if amp == "none":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _batches(starts: Sequence[int], size: int) -> Iterator[Sequence[int]]:
    for i in range(0, len(starts), size):
        yield starts[i:i + size]


@torch.no_grad()
def predict_recording(
    model: torch.nn.Module,
    signal: np.ndarray,
    task: Task,
    window: int,
    hop: int,
    batch_size: int,
    device: torch.device,
    combine: str = "mean",
    tail: str = "align",
    amp: str = "none",
    window_pipeline: Optional[Pipeline] = None,
) -> np.ndarray:
    """Return per-sample probabilities ``(num_outputs, N)`` (float32) for one recording.

    ``signal`` is the preprocessed ``(C, N)`` array (a memory map is fine)."""
    n = signal.shape[1]
    starts = plan_windows(n, window, hop, tail)
    weights = np.ones(window, dtype=np.float32) if combine == "mean" else \
        (np.hanning(window + 2)[1:-1].astype(np.float32) + 1e-3)
    acc = np.zeros((task.num_outputs, n), dtype=np.float32)
    wsum = np.zeros(n, dtype=np.float32)
    wants_meta = getattr(model, "wants_meta", False)
    for chunk in _batches(starts, batch_size):
        xs, valids = [], []
        for s in chunk:
            x, valid = read_window(signal, s, window)
            if window_pipeline is not None:
                x = window_pipeline(x).astype(np.float32)
            xs.append(x)
            valids.append(valid)
        xb = torch.from_numpy(np.stack(xs)).to(device, non_blocking=True)
        with autocast_context(device, amp):
            if wants_meta:  # same keys as in training; patient/rec are unknown (-1) at inference
                b = len(chunk)
                meta = {"patient": torch.full((b,), -1, device=device), "rec": torch.full((b,), -1, device=device),
                        "start": torch.tensor(list(chunk), device=device)}
                out = model(xb, meta)
            else:
                out = model(xb)
        logits = align_logits(as_output(out).logits.float(), window)
        probs = task.probs(logits).cpu().numpy()
        for p, s, valid in zip(probs, chunk, valids):
            acc[:, s:s + valid] += p[:, :valid] * weights[:valid]
            wsum[s:s + valid] += weights[:valid]
    if np.any(wsum == 0):  # cannot happen with the window plans above; guard anyway
        raise RuntimeError("some samples were not covered by any inference window")
    return acc / wsum
