"""Recording discovery and validated signal access for HDF5 datasets.

Expected layout (configurable through ``data.*``)::

    <root>/<patient>/<record>.h5            dataset ``signals`` (n_channels, n_samples), attr ``fs``
    <root>/<patient>/<patient>_annotations.txt

:class:`H5Store` turns this into a list of :class:`Recording` objects. Scanning only
reads metadata and checks, per recording:

* the signal dataset exists, is 2-D and not transposed;
* the sampling rate attribute exists and matches ``data.fs`` (unless resampling);
* the channel count matches ``data.channels``; if the file stores channel names
  (``data.channel_attr`` or a common attribute name), they are matched by name and
  the rows are reordered to the configured montage;
* every annotation row maps to exactly one file of that patient, every event lies
  inside its recording, events do not overlap.

Any violation raises :class:`DataValidationError` listing all problems found.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config
from ..utils import natural_key
from ..utils.logging import get_logger
from .annotations import (
    AnnotationError,
    Event,
    events_to_intervals,
    group_rows_by_file,
    parse_annotation_file,
    validate_events,
)

log = get_logger("data.store")

_CHANNEL_ATTR_CANDIDATES = ("channels", "channel_names", "ch_names", "labels", "montage")


class DataValidationError(ValueError):
    pass


def resample_ratio(fs_in: float, fs_out: float) -> Tuple[int, int]:
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    if abs(frac.numerator / frac.denominator - fs_out / fs_in) > 1e-9:
        raise ValueError(f"Cannot express resampling {fs_in} -> {fs_out} Hz as a small rational ratio")
    return frac.numerator, frac.denominator


def resampled_length(n: int, fs_in: float, fs_out: float) -> int:
    if fs_in == fs_out:
        return n
    up, down = resample_ratio(fs_in, fs_out)
    return -(-n * up // down)  # ceil, as scipy.signal.resample_poly


@dataclass(frozen=True)
class Recording:
    patient: str
    name: str  # file name on disk, e.g. "chb01_03.h5"
    path: str
    fs: float  # native sampling rate stored in the file
    n_samples: int  # native number of samples
    events: Tuple[Event, ...] = ()
    channel_index: Optional[Tuple[int, ...]] = None  # file rows in configured order (None: identity)
    file_size: int = 0
    file_mtime_ns: int = 0

    @property
    def stem(self) -> str:
        return Path(self.name).stem

    @property
    def key(self) -> str:
        return f"{self.patient}/{self.stem}"

    @property
    def duration_sec(self) -> float:
        return self.n_samples / self.fs

    @property
    def n_events(self) -> int:
        return len(self.events)

    def n_samples_at(self, fs_out: float) -> int:
        return resampled_length(self.n_samples, self.fs, fs_out)

    def intervals(self, fs_out: float) -> np.ndarray:
        """Event intervals ``[start, end, label]`` in samples at ``fs_out``."""
        return events_to_intervals(self.events, fs_out, self.n_samples_at(fs_out))


@dataclass
class ScanReport:
    recordings: List[Recording] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    channel_names_verified: int = 0


def _decode_names(value) -> List[str]:
    arr = np.asarray(value)
    if arr.ndim == 0:
        item = arr.item()
        item = item.decode() if isinstance(item, bytes) else str(item)
        parts = [p for p in item.replace(";", ",").split(",")]
    else:
        parts = [v.decode() if isinstance(v, bytes) else str(v) for v in arr.ravel().tolist()]
    return [p for p in (s.strip() for s in parts) if p]


def normalize_channel_name(name: str) -> str:
    """Case/space-insensitive channel key; a leading 'EEG' prefix (EDF convention) is dropped."""
    s = name.strip().upper().replace(" ", "")
    return s[3:] if s.startswith("EEG") and len(s) > 3 else s


class H5Store:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.d = cfg.data
        self.root = cfg.resolve_path(self.d.root)
        self.channels = list(self.d.channels)
        self.fs_out = cfg.fs

    # ------------------------------------------------------------------ discovery
    def patient_dir(self, patient: str) -> Path:
        return self.root / patient

    def list_patients(self) -> List[str]:
        if not self.root.is_dir():
            raise DataValidationError(f"data.root does not exist or is not a directory: {self.root}")
        return sorted((p.name for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")),
                      key=natural_key)

    def _record_files(self, pdir: Path) -> List[Path]:
        files = [p for p in pdir.iterdir()
                 if p.is_file() and fnmatch.fnmatch(p.name, self.d.file_pattern) and not p.name.startswith((".", "_"))]
        return sorted(files, key=lambda p: natural_key(p.name))

    def _excluded(self, patient: str, name: str) -> bool:
        keys = {f"{patient}/{name}", f"{patient}/{Path(name).stem}"}
        return any(k in keys for k in self.d.exclude_records)

    def scan(self, patients: Sequence[str]) -> ScanReport:
        report = ScanReport()
        errors: List[str] = []
        for patient in patients:
            try:
                self._scan_patient(patient, report, errors)
            except (OSError, AnnotationError, DataValidationError) as e:
                errors.append(str(e))
        if errors:
            raise DataValidationError(
                f"{len(errors)} data validation error(s) under {self.root}:\n  - " + "\n  - ".join(errors))
        for w in report.warnings:
            log.warning(w)
        return report

    def _match_rows(self, patient: str, rows, files: List[Path], ann_path: Path,
                    errors: List[str]) -> Dict[str, List[Event]]:
        by_name = {f.name: f for f in files}
        by_stem: Dict[str, List[Path]] = {}
        for f in files:
            by_stem.setdefault(f.stem, []).append(f)
        out: Dict[str, List[Event]] = {}
        for fname, frows in group_rows_by_file(rows).items():
            target: Optional[Path] = None
            if self.d.annotation_match == "exact":
                target = by_name.get(fname)
                if target is None:
                    hint = ""
                    same_stem = by_stem.get(Path(fname).stem)
                    if same_stem:
                        hint = (f" A file with the same stem exists ({same_stem[0].name}); "
                                f"set data.annotation_match=stem if that is intended.")
                    errors.append(f"{ann_path} (line {frows[0].line}): annotated file {fname!r} "
                                  f"not found among {patient}/{self.d.file_pattern}.{hint}")
                    continue
            else:
                cands = by_stem.get(Path(fname).stem, [])
                if len(cands) != 1:
                    errors.append(f"{ann_path} (line {frows[0].line}): annotated file {fname!r} matches "
                                  f"{len(cands)} files by stem in {patient}")
                    continue
                target = cands[0]
            out.setdefault(target.name, []).extend(r.event for r in frows)
        return out

    def _scan_patient(self, patient: str, report: ScanReport, errors: List[str]) -> None:
        pdir = self.patient_dir(patient)
        if not pdir.is_dir():
            raise DataValidationError(f"patient directory not found: {pdir}")
        files = self._record_files(pdir)
        if not files:
            raise DataValidationError(f"no files matching {self.d.file_pattern!r} in {pdir}")
        ann_path = pdir / self.d.annotation_file.format(patient=patient)
        rows = parse_annotation_file(ann_path, self.d.annotation_section, self.d.label_column,
                                     self.d.label_map, self.d.default_label)
        events_by_file = self._match_rows(patient, rows, files, ann_path, errors)

        for f in files:
            if self._excluded(patient, f.name):
                n = len(events_by_file.get(f.name, []))
                report.warnings.append(f"{patient}/{f.name}: excluded by data.exclude_records"
                                       + (f" (drops {n} annotated event(s))" if n else ""))
                continue
            try:
                rec, verified = self._inspect_file(patient, f, events_by_file.get(f.name, []), report.warnings)
            except (OSError, KeyError, AnnotationError, DataValidationError) as e:
                errors.append(f"{patient}/{f.name}: {e}")
                continue
            report.channel_names_verified += int(verified)
            report.recordings.append(rec)

    def _inspect_file(self, patient: str, path: Path, events: List[Event],
                      warnings: List[str]) -> Tuple[Recording, bool]:
        import h5py

        with h5py.File(path, "r") as f:
            if self.d.signal_key not in f:
                raise DataValidationError(f"dataset {self.d.signal_key!r} not found (keys: {list(f.keys())})")
            ds = f[self.d.signal_key]
            if ds.ndim != 2:
                raise DataValidationError(f"{self.d.signal_key!r} must be 2-D (channels, samples), got shape {ds.shape}")
            if not np.issubdtype(ds.dtype, np.number):
                raise DataValidationError(f"{self.d.signal_key!r} has non-numeric dtype {ds.dtype}")
            n_rows, n_samples = int(ds.shape[0]), int(ds.shape[1])
            fs_raw = f.attrs.get(self.d.fs_attr, ds.attrs.get(self.d.fs_attr))
            if fs_raw is None:
                raise DataValidationError(f"sampling-rate attribute {self.d.fs_attr!r} missing")
            fs = float(np.asarray(fs_raw).ravel()[0])
            names = None
            for key in ([self.d.channel_attr] if self.d.channel_attr else list(_CHANNEL_ATTR_CANDIDATES)):
                for holder in (ds.attrs, f.attrs):
                    if key in holder:
                        names = _decode_names(holder[key])
                        break
                if names is not None:
                    break

        n_ch = len(self.channels)
        if n_rows != n_ch and n_samples == n_ch:
            raise DataValidationError(f"signals shape {(n_rows, n_samples)} looks transposed; expected (channels, samples)")
        if not np.isfinite(fs) or fs <= 0:
            raise DataValidationError(f"invalid sampling rate {fs}")
        if self.d.resample_to is None and abs(fs - self.d.fs) > 1e-6 * self.d.fs:
            raise DataValidationError(f"sampling rate {fs} Hz != data.fs {self.d.fs} Hz "
                                      f"(set data.resample_to to resample)")
        if n_samples <= 0:
            raise DataValidationError("empty recording")

        channel_index: Optional[Tuple[int, ...]] = None
        verified = False
        if names is not None:
            if len(names) != n_rows:
                raise DataValidationError(f"{len(names)} channel names stored for {n_rows} signal rows")
            norm = [normalize_channel_name(n) for n in names]
            idx = []
            missing = []
            for ch in self.channels:
                key = normalize_channel_name(ch)
                if norm.count(key) > 1:
                    raise DataValidationError(f"channel {ch} appears {norm.count(key)} times in the file")
                if key not in norm:
                    missing.append(ch)
                else:
                    idx.append(norm.index(key))
            if missing:
                raise DataValidationError(f"channels {missing} not found in file channels {names}")
            if idx != list(range(n_rows)):
                channel_index = tuple(idx)
            verified = True
        else:
            if self.d.require_channel_names:
                raise DataValidationError("no channel-name attribute found but data.require_channel_names=true")
            if n_rows != n_ch:
                raise DataValidationError(f"{n_rows} signal rows but data.channels lists {n_ch} channels")

        valid, ev_warnings = validate_events(events, n_samples / fs, f"{patient}/{path.name}",
                                             self.d.end_tolerance_sec, self.d.overlapping_events)
        warnings.extend(ev_warnings)
        st = path.stat()
        rec = Recording(patient=patient, name=path.name, path=str(path), fs=fs, n_samples=n_samples,
                        events=tuple(valid), channel_index=channel_index,
                        file_size=int(st.st_size), file_mtime_ns=int(st.st_mtime_ns))
        return rec, verified

    def recording_from_file(self, path: Path, patient: Optional[str] = None) -> Recording:
        """Validated :class:`Recording` for a single file without annotations (prediction)."""
        path = Path(path)
        warnings: List[str] = []
        rec, _ = self._inspect_file(patient or path.parent.name, path, [], warnings)
        return rec

    # ------------------------------------------------------------------ signal access
    def load_signal(self, rec: Recording) -> np.ndarray:
        """Raw signal as float64 ``(n_channels, n_samples_out)`` in configured channel order."""
        import h5py

        with h5py.File(rec.path, "r") as f:
            x = np.asarray(f[self.d.signal_key][()], dtype=np.float64)
        if x.shape[1] != rec.n_samples:
            raise DataValidationError(f"{rec.key}: file changed since scan ({x.shape[1]} != {rec.n_samples} samples)")
        if rec.channel_index is not None:
            x = x[list(rec.channel_index)]
        bad = ~np.isfinite(x)
        if bad.any():
            ch = [self.channels[i] for i in np.unique(np.nonzero(bad)[0])]
            raise DataValidationError(f"{rec.key}: {int(bad.sum())} non-finite samples in channels {ch}")
        if self.fs_out != rec.fs:
            from scipy.signal import resample_poly

            up, down = resample_ratio(rec.fs, self.fs_out)
            x = resample_poly(x, up, down, axis=-1)
        return x


# ---------------------------------------------------------------------- summaries
def summarize(recordings: Iterable[Recording], fs_out: float) -> Dict[str, dict]:
    """Per-patient statistics (hours, seizures, seizure fraction, durations)."""
    out: Dict[str, dict] = {}
    for rec in recordings:
        s = out.setdefault(rec.patient, {"records": 0, "records_with_events": 0, "hours": 0.0,
                                         "events": 0, "event_seconds": 0.0, "event_durations": []})
        s["records"] += 1
        s["hours"] += rec.duration_sec / 3600
        if rec.events:
            s["records_with_events"] += 1
        s["events"] += rec.n_events
        for ev in rec.events:
            s["event_seconds"] += ev.duration
            s["event_durations"].append(ev.duration)
    for s in out.values():
        d = s.pop("event_durations")
        s["event_fraction"] = s["event_seconds"] / (s["hours"] * 3600) if s["hours"] else 0.0
        s["event_duration_min"] = float(min(d)) if d else None
        s["event_duration_median"] = float(np.median(d)) if d else None
        s["event_duration_max"] = float(max(d)) if d else None
    return out


def totals(summary: Dict[str, dict]) -> dict:
    hours = sum(s["hours"] for s in summary.values())
    ev_sec = sum(s["event_seconds"] for s in summary.values())
    return {"patients": len(summary), "records": sum(s["records"] for s in summary.values()),
            "hours": hours, "events": sum(s["events"] for s in summary.values()),
            "event_seconds": ev_sec, "event_fraction": ev_sec / (hours * 3600) if hours else 0.0}
