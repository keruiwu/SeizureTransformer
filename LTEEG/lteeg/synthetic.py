"""Synthetic CHB-MIT-like dataset for tests and pipeline smoke runs.

Produces exactly the on-disk format the real data uses::

    <root>/<patient>/<patient>_NN.h5        signals (C, N) float32, attrs fs
    <root>/<patient>/<patient>_annotations.txt   [seizures] section, tab-separated

Background is 1/f-like noise with an alpha rhythm and occasional non-rhythmic
artifacts; seizures are evolving 3-7 Hz rhythmic discharges with amplitude ramps on
a random hemisphere. The task is easy on purpose: it is meant to show that the whole
pipeline learns, not to benchmark models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import CHBMIT_CHANNELS


def _background(rng: np.random.Generator, n_ch: int, n: int, fs: float) -> np.ndarray:
    from scipy.signal import lfilter

    white = rng.standard_normal((n_ch, n))
    leak = np.exp(-2 * np.pi * 1.0 / fs)  # leaky integrator: crude 1/f-like spectrum
    pink = lfilter([1.0], [1.0, -leak], white, axis=1)
    pink = pink / (pink.std(axis=1, keepdims=True) + 1e-9)
    t = np.arange(n) / fs
    alpha = 0.5 * np.sin(2 * np.pi * rng.uniform(8.5, 11.5, size=(n_ch, 1)) * t + rng.uniform(0, 2 * np.pi, (n_ch, 1)))
    return pink + 0.6 * white + alpha


def _seizure(rng: np.random.Generator, n: int, fs: float) -> np.ndarray:
    t = np.arange(n) / fs
    f0, f1 = rng.uniform(5.0, 7.0), rng.uniform(2.5, 3.5)
    freq = np.linspace(f0, f1, n)
    phase = 2 * np.pi * np.cumsum(freq) / fs
    wave = np.sin(phase) + 0.5 * np.sin(2 * phase + 0.3) + 0.25 * np.sin(3 * phase + 0.7)
    ramp = np.minimum(1.0, np.minimum(t, t[-1] - t) / 3.0 + 0.2)
    return 4.0 * wave * ramp


def make_synthetic_dataset(
    root: Path,
    patients: Optional[Sequence[str]] = None,
    records_per_patient: int = 3,
    record_minutes: float = 10.0,
    seizures_per_record: Tuple[int, int] = (0, 2),
    seizure_sec: Tuple[float, float] = (20.0, 90.0),
    fs: float = 256.0,
    channels: Sequence[str] = CHBMIT_CHANNELS,
    seed: int = 0,
    with_channel_names: bool = False,
    min_seizures_per_patient: int = 1,
) -> Dict[str, List[dict]]:
    root = Path(root)
    rng = np.random.default_rng(seed)
    patients = list(patients or [f"chb{i:02d}" for i in range(1, 5)])
    n_ch = len(channels)
    n = int(record_minutes * 60 * fs)
    left = [i for i, c in enumerate(channels) if any(k in c for k in ("1", "3", "7"))] or list(range(n_ch // 2))
    right = [i for i in range(n_ch) if i not in left] or list(range(n_ch // 2, n_ch))
    import h5py

    manifest: Dict[str, List[dict]] = {}
    for patient in patients:
        pdir = root / patient
        pdir.mkdir(parents=True, exist_ok=True)
        rows = []
        manifest[patient] = []
        n_sz_patient = 0
        for r in range(records_per_patient):
            name = f"{patient}_{r + 1:02d}.h5"
            x = _background(rng, n_ch, n, fs)
            # artifacts: short high-amplitude non-rhythmic bursts
            for _ in range(rng.integers(1, 4)):
                a = int(rng.integers(0, n - int(2 * fs)))
                L = int(rng.uniform(0.3, 1.5) * fs)
                x[:, a:a + L] += rng.standard_normal((n_ch, 1)) * 6.0 * np.hanning(L)
            k = int(rng.integers(seizures_per_record[0], seizures_per_record[1] + 1))
            if r == records_per_patient - 1 and n_sz_patient + k < min_seizures_per_patient:
                k = min_seizures_per_patient - n_sz_patient
            events = []
            if k:
                slots = np.linspace(60.0, record_minutes * 60 - 120.0, k + 1)
                for i in range(k):
                    dur = float(np.round(rng.uniform(*seizure_sec)))
                    lo, hi = slots[i], max(slots[i], slots[i + 1] - dur - 5)
                    start = float(np.round(rng.uniform(lo, hi)))
                    a, b = int(start * fs), int((start + dur) * fs)
                    side = left if rng.random() < 0.5 else right
                    sz = _seizure(rng, b - a, fs)
                    for ch in side:
                        x[ch, a:b] += sz * rng.uniform(0.7, 1.3)
                    events.append((start, start + dur))
                    rows.append(f"{name}\t{int(start)}\t{int(start + dur)}")
            n_sz_patient += k
            x = (x * 40.0 + rng.normal(0, 20, size=(n_ch, 1))).astype(np.float32)  # microvolt-ish + DC
            with h5py.File(pdir / name, "w") as f:
                f.create_dataset("signals", data=x)
                f.attrs["fs"] = float(fs)
                if with_channel_names:
                    f.attrs["channels"] = np.array([c.encode() for c in channels])
            manifest[patient].append({"file": name, "events": events, "n_samples": n})
        text = ["# synthetic annotations", "[info]", f"patient\t{patient}", "", "[seizures]",
                "file\tstart_sec\tend_sec"] + rows
        (pdir / f"{patient}_annotations.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    return manifest
