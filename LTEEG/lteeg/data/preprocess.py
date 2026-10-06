"""Signal preprocessing operators and pipelines.

A pipeline is a list of ``{"op": name, **params}`` dicts (see ``preprocess.*`` in the
config). Two pipelines exist:

* ``preprocess.recording`` runs once on the whole recording; its output is cached on
  disk (``lteeg.data.cache``). Filtering the full recording avoids the start-up
  transient that per-window IIR filtering introduces at every window edge.
* ``preprocess.window`` runs on every window right before the model, identically in
  training and long-range inference (e.g. per-window normalization, or the
  original per-window filtering for exact replication).

Linear filters are designed as second-order sections. Consecutive filter ops with
the same phase mode are fused into one SOS cascade, which is mathematically
identical to applying them one after another (zero initial state) but faster.

Operators work on float64 arrays of shape ``(channels, samples)``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

from ..registry import PREPROCESSORS, call_with_checked_kwargs


class Op:
    """Base operator. Linear time-invariant filters expose ``sos(fs)`` for fusion."""

    zero_phase: bool = False

    def sos(self, fs: float) -> Optional[np.ndarray]:
        return None

    def apply(self, x: np.ndarray, fs: float, info: Dict[str, Any]) -> np.ndarray:
        sos = self.sos(fs)
        if sos is None:
            raise NotImplementedError
        return _run_sos(sos, x, self.zero_phase)

    def describe(self) -> Dict[str, Any]:
        return {"op": self.name, **{k: v for k, v in vars(self).items() if not k.startswith("_")}}

    name = "op"


def _run_sos(sos: np.ndarray, x: np.ndarray, zero_phase: bool) -> np.ndarray:
    if zero_phase:
        return signal.sosfiltfilt(sos, x, axis=-1)
    return signal.sosfilt(sos, x, axis=-1)


def _check_freq(freq: float, fs: float, what: str) -> None:
    if not 0 < freq < fs / 2:
        raise ValueError(f"{what}={freq} Hz must lie in (0, Nyquist={fs / 2} Hz)")


@PREPROCESSORS.register("zscore")
class ZScore(Op):
    """Per-channel standardization. Flat channels (std < eps) become all-zero instead of
    NaN and are reported in ``info['flat_channels']``. ``robust`` uses median and IQR."""

    name = "zscore"

    def __init__(self, eps: float = 1e-8, robust: bool = False):
        self.eps = float(eps)
        self.robust = bool(robust)

    def apply(self, x: np.ndarray, fs: float, info: Dict[str, Any]) -> np.ndarray:
        if self.robust:
            center = np.median(x, axis=-1, keepdims=True)
            q75, q25 = np.percentile(x, [75, 25], axis=-1, keepdims=True)
            scale = (q75 - q25) / 1.349
        else:
            center = x.mean(axis=-1, keepdims=True)
            scale = x.std(axis=-1, keepdims=True)
        flat = scale[..., 0] < self.eps
        if flat.any():
            info.setdefault("flat_channels", []).extend(int(i) for i in np.nonzero(flat)[0])
        scale = np.where(flat[..., None], 1.0, scale)
        out = (x - center) / scale
        if flat.any():
            out[flat] = 0.0
        return out


@PREPROCESSORS.register("bandpass")
class BandPass(Op):
    name = "bandpass"

    def __init__(self, low: float, high: float, order: int = 3, zero_phase: bool = False):
        self.low, self.high, self.order, self.zero_phase = float(low), float(high), int(order), bool(zero_phase)

    def sos(self, fs: float) -> np.ndarray:
        _check_freq(self.low, fs, "bandpass.low")
        _check_freq(self.high, fs, "bandpass.high")
        if self.low >= self.high:
            raise ValueError("bandpass.low must be < bandpass.high")
        return signal.butter(self.order, [self.low, self.high], btype="bandpass", fs=fs, output="sos")


@PREPROCESSORS.register("highpass")
class HighPass(Op):
    name = "highpass"

    def __init__(self, cutoff: float, order: int = 3, zero_phase: bool = False):
        self.cutoff, self.order, self.zero_phase = float(cutoff), int(order), bool(zero_phase)

    def sos(self, fs: float) -> np.ndarray:
        _check_freq(self.cutoff, fs, "highpass.cutoff")
        return signal.butter(self.order, self.cutoff, btype="highpass", fs=fs, output="sos")


@PREPROCESSORS.register("lowpass")
class LowPass(Op):
    name = "lowpass"

    def __init__(self, cutoff: float, order: int = 3, zero_phase: bool = False):
        self.cutoff, self.order, self.zero_phase = float(cutoff), int(order), bool(zero_phase)

    def sos(self, fs: float) -> np.ndarray:
        _check_freq(self.cutoff, fs, "lowpass.cutoff")
        return signal.butter(self.order, self.cutoff, btype="lowpass", fs=fs, output="sos")


@PREPROCESSORS.register("notch")
class Notch(Op):
    """Second-order IIR notch (scipy.signal.iirnotch), quality factor ``q``."""

    name = "notch"

    def __init__(self, freq: float, q: float = 30.0, zero_phase: bool = False):
        self.freq, self.q, self.zero_phase = float(freq), float(q), bool(zero_phase)

    def sos(self, fs: float) -> np.ndarray:
        _check_freq(self.freq, fs, "notch.freq")
        b, a = signal.iirnotch(self.freq, self.q, fs=fs)
        return signal.tf2sos(b, a)


@PREPROCESSORS.register("clip")
class Clip(Op):
    """Clamp amplitudes to ``[-value, value]`` (useful after normalization to tame artifacts)."""

    name = "clip"

    def __init__(self, value: float):
        self.value = float(value)

    def apply(self, x: np.ndarray, fs: float, info: Dict[str, Any]) -> np.ndarray:
        frac = float(np.mean(np.abs(x) > self.value))
        info["clipped_fraction"] = info.get("clipped_fraction", 0.0) + frac
        return np.clip(x, -self.value, self.value)


def build_op(spec: Dict[str, Any]) -> Op:
    if "op" not in spec:
        raise ValueError(f"preprocessing step {spec} has no 'op' key")
    params = {k: v for k, v in spec.items() if k != "op"}
    cls = PREPROCESSORS.get(spec["op"])
    return call_with_checked_kwargs(cls, "preprocessing op", spec["op"], **params)


class Pipeline:
    def __init__(self, specs: Sequence[Dict[str, Any]], fs: float):
        self.specs = [dict(s) for s in specs]
        self.fs = float(fs)
        self.ops: List[Op] = [build_op(s) for s in self.specs]
        self._plan = self._compile()

    def _compile(self) -> List[Tuple[str, Any]]:
        """Group consecutive linear filters with equal phase mode into one SOS cascade."""
        plan: List[Tuple[str, Any]] = []
        for op in self.ops:
            sos = op.sos(self.fs)  # also validates frequencies against fs eagerly
            if sos is not None:
                if plan and plan[-1][0] == "sos" and plan[-1][1][1] == op.zero_phase:
                    prev_sos, zp = plan[-1][1]
                    plan[-1] = ("sos", (np.vstack([prev_sos, sos]), zp))
                else:
                    plan.append(("sos", (sos, op.zero_phase)))
            else:
                plan.append(("op", op))
        return plan

    def __len__(self) -> int:
        return len(self.ops)

    def __call__(self, x: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        info = {} if info is None else info
        x = np.asarray(x, dtype=np.float64)
        for kind, payload in self._plan:
            if kind == "sos":
                sos, zero_phase = payload
                x = _run_sos(sos, x, zero_phase)
            else:
                x = payload.apply(x, self.fs, info)
        return x

    def describe(self) -> List[Dict[str, Any]]:
        return self.specs
