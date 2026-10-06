"""On-disk cache of preprocessed recordings, read through memory maps.

CHB-MIT is ~980 h x 18 ch x 256 Hz: tens of GB, far more than RAM. Each recording
is preprocessed once (``preprocess.recording``) and stored as ``.npy``; training
windows and long-range inference then read slices through ``np.load(mmap_mode='r')``,
so random window access costs one page-cache hit and RAM usage stays flat.

Layout::

    <data.cache_dir>/<fingerprint>/<patient>/<stem>.npy    float32|float16 (C, N)
    <data.cache_dir>/<fingerprint>/<patient>/<stem>.json   sidecar, written last (commit marker)

The fingerprint hashes everything that changes the cached values (channels, fs,
resampling, pipeline, dtype, cache format version). The sidecar stores the source
file size/mtime; a mismatch triggers a rebuild of that recording.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import Config, config_from_dict
from ..utils import atomic_write_json, format_duration, read_json
from ..utils.logging import get_logger
from .preprocess import Pipeline
from .store import H5Store, Recording

log = get_logger("data.cache")

CACHE_FORMAT_VERSION = 1


def cache_fingerprint(cfg: Config) -> str:
    payload = {
        "version": CACHE_FORMAT_VERSION,
        "channels": [c.upper() for c in cfg.data.channels],
        "signal_key": cfg.data.signal_key,
        "fs": cfg.fs,
        "resample_to": cfg.data.resample_to,
        "pipeline": cfg.preprocess.recording,
        "dtype": cfg.data.cache_dtype,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


class SignalCache:
    def __init__(self, cfg: Config, store: Optional[H5Store] = None):
        self.cfg = cfg
        self.store = store or H5Store(cfg)
        self.dir = cfg.resolve_path(cfg.data.cache_dir) / cache_fingerprint(cfg)
        self.dtype = np.dtype(cfg.data.cache_dtype)
        self._maps: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ paths & state
    def array_path(self, rec: Recording) -> Path:
        return self.dir / rec.patient / f"{rec.stem}.npy"

    def meta_path(self, rec: Recording) -> Path:
        return self.dir / rec.patient / f"{rec.stem}.json"

    def is_valid(self, rec: Recording) -> bool:
        meta_p, arr_p = self.meta_path(rec), self.array_path(rec)
        if not (meta_p.is_file() and arr_p.is_file()):
            return False
        try:
            meta = read_json(meta_p)
        except (OSError, ValueError):
            return False
        return (meta.get("source_size") == rec.file_size and meta.get("source_mtime_ns") == rec.file_mtime_ns
                and meta.get("n_samples") == rec.n_samples_at(self.cfg.fs)
                and meta.get("n_channels") == self.cfg.n_channels)

    def meta(self, rec: Recording) -> Dict[str, Any]:
        return read_json(self.meta_path(rec))

    # ------------------------------------------------------------------ build
    def ensure(self, recordings: Sequence[Recording], jobs: int = 1, force: bool = False) -> None:
        todo = [r for r in recordings if force or not self.is_valid(r)]
        if not todo:
            log.info(f"Cache up to date: {len(recordings)} recordings in {self.dir}")
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        _write_fingerprint_info(self.dir, self.cfg)
        hours = sum(r.duration_sec for r in todo) / 3600
        log.info(f"Preprocessing {len(todo)} recording(s) ({hours:.1f} h) into {self.dir} with {jobs} job(s)")
        self.close()
        t0 = time.time()
        cfg_dict = self.cfg.to_dict()
        if jobs <= 1:
            for i, rec in enumerate(todo, 1):
                info = _build_one(cfg_dict, rec, str(self.dir))
                _log_progress(i, len(todo), rec, info, t0)
        else:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futures = {ex.submit(_build_one, cfg_dict, rec, str(self.dir)): rec for rec in todo}
                for i, fut in enumerate(as_completed(futures), 1):
                    _log_progress(i, len(todo), futures[fut], fut.result(), t0)
        log.info(f"Cache built in {format_duration(time.time() - t0)}")

    # ------------------------------------------------------------------ read
    def open(self, rec: Recording) -> np.ndarray:
        """Read-only memory map of shape (C, N). Opened lazily, so a cache object can be
        pickled into DataLoader worker processes (Windows uses spawn)."""
        m = self._maps.get(rec.key)
        if m is None:
            m = np.load(self.array_path(rec), mmap_mode="r")
            self._maps[rec.key] = m
        return m

    def close(self) -> None:
        self._maps.clear()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_maps"] = {}
        return state


def _write_fingerprint_info(cache_dir: Path, cfg: Config) -> None:
    p = cache_dir / "fingerprint.json"
    if not p.exists():
        atomic_write_json(p, {"channels": cfg.data.channels, "fs": cfg.fs, "resample_to": cfg.data.resample_to,
                              "pipeline": cfg.preprocess.recording, "dtype": cfg.data.cache_dtype,
                              "version": CACHE_FORMAT_VERSION})


def _log_progress(i: int, n: int, rec: Recording, info: Dict[str, Any], t0: float) -> None:
    flat = info.get("flat_channels") or []
    extra = f" flat_channels={flat}" if flat else ""
    elapsed = time.time() - t0
    eta = elapsed / i * (n - i)
    log.info(f"[{i}/{n}] {rec.key} ({rec.duration_sec / 3600:.2f} h){extra} eta {format_duration(eta)}")


def _build_one(cfg_dict: Dict[str, Any], rec: Recording, cache_dir: str) -> Dict[str, Any]:
    """Preprocess one recording and store it. Top-level function so it pickles for
    ProcessPoolExecutor on Windows."""
    cfg = config_from_dict(cfg_dict)
    store = H5Store(cfg)
    x = store.load_signal(rec)
    info: Dict[str, Any] = {}
    pipeline = Pipeline(cfg.preprocess.recording, cfg.fs)
    y = pipeline(x, info)
    if not np.isfinite(y).all():
        raise ValueError(f"{rec.key}: preprocessing produced non-finite values")
    dtype = np.dtype(cfg.data.cache_dtype)
    if dtype == np.float16 and np.abs(y).max() > 6e4:
        raise ValueError(f"{rec.key}: values exceed float16 range; use data.cache_dtype=float32 or a clip op")
    out_dir = Path(cache_dir) / rec.patient
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{rec.stem}.npy"
    tmp = out_dir / f"{rec.stem}.npy.tmp"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=y.shape)
    arr[:] = y
    arr.flush()
    del arr
    os.replace(tmp, final)
    meta = {
        "key": rec.key,
        "source": rec.path,
        "source_size": rec.file_size,
        "source_mtime_ns": rec.file_mtime_ns,
        "n_channels": int(y.shape[0]),
        "n_samples": int(y.shape[1]),
        "fs": cfg.fs,
        "flat_channels": [cfg.data.channels[i] for i in sorted(set(info.get("flat_channels", [])))],
        "channel_std_after": [float(v) for v in y.std(axis=1)],
        "clipped_fraction": info.get("clipped_fraction"),
    }
    atomic_write_json(Path(cache_dir) / rec.patient / f"{rec.stem}.json", meta)
    return {"flat_channels": meta["flat_channels"]}


class InMemorySource:
    """Preprocess on the fly without a cache (single-file prediction, small datasets)."""

    def __init__(self, cfg: Config, store: Optional[H5Store] = None):
        self.cfg = cfg
        self.store = store or H5Store(cfg)
        self.pipeline = Pipeline(cfg.preprocess.recording, cfg.fs)
        self._last: Optional[tuple] = None

    def open(self, rec: Recording) -> np.ndarray:
        if self._last is not None and self._last[0] == rec.key:
            return self._last[1]
        x = self.pipeline(self.store.load_signal(rec)).astype(np.float32)
        self._last = (rec.key, x)
        return x

    def close(self) -> None:
        self._last = None


def build_source(cfg: Config, recordings: List[Recording], use_cache: bool = True, jobs: int = 1,
                 store: Optional[H5Store] = None):
    if use_cache:
        cache = SignalCache(cfg, store)
        cache.ensure(recordings, jobs=jobs)
        return cache
    return InMemorySource(cfg, store)
