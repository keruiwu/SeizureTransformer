"""Event annotations: parsing, validation and conversion to per-sample labels.

Annotation file format (one per patient, e.g. ``chb01/chb01_annotations.txt``)::

    [seizures]
    file            start   end
    chb01_03.h5     2996    3036
    ...

* Sections are introduced by ``[name]``; only the configured section is read, other
  sections and text before the first section are ignored.
* Inside the section, the header row starts with ``file`` (case-insensitive); data
  rows are tab-separated ``file, start_sec, end_sec[, label...]``. Blank lines and
  lines starting with ``#`` are skipped.
* Every malformed row is an error that names the file and line number: labels must
  never be silently dropped.

Times are kept in seconds (the source of truth). They are converted to sample
indices with ``round(t * fs)``, the same convention the SzCORE/timescoring scorer
uses, so training targets and evaluation references agree to the sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

IGNORE_INDEX = -1


@dataclass(frozen=True)
class Event:
    start: float  # seconds from recording start
    end: float  # seconds, exclusive
    label: int = 1  # class index, 0 is reserved for background

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class AnnotationRow:
    file: str
    event: Event
    line: int


class AnnotationError(ValueError):
    pass


def parse_annotation_file(
    path: Path,
    section: str = "seizures",
    label_column: Optional[int] = None,
    label_map: Optional[Mapping[str, int]] = None,
    default_label: int = 1,
) -> List[AnnotationRow]:
    """Parse the ``[section]`` block of an annotation file into rows (file order kept)."""
    path = Path(path)
    if not path.is_file():
        raise AnnotationError(f"Annotation file not found: {path}")
    target = section.strip().lower()
    rows: List[AnnotationRow] = []
    in_section = False
    seen_section = False
    # utf-8-sig strips a BOM written by Windows editors; splitlines handles \r\n.
    text = path.read_text(encoding="utf-8-sig")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip().lower()
            if in_section and name != target:
                in_section = False
            elif name == target:
                if seen_section:
                    raise AnnotationError(f"{path}:{lineno}: section [{section}] appears twice")
                in_section = seen_section = True
            continue
        if not in_section:
            continue
        fields = [f.strip() for f in raw.split("\t")]
        fields = [f for f in fields if f != ""]
        if len(fields) < 3:
            # tolerate space-separated rows but only when unambiguous
            fields = line.split()
        if fields and fields[0].lower().startswith("file") and not _is_number(fields[1] if len(fields) > 1 else ""):
            continue  # header row ("file<TAB>start<TAB>end")
        if len(fields) < 3:
            raise AnnotationError(f"{path}:{lineno}: expected 'file<TAB>start<TAB>end', got {raw!r}")
        try:
            start, end = float(fields[1]), float(fields[2])
        except ValueError:
            raise AnnotationError(f"{path}:{lineno}: start/end are not numbers in {raw!r}") from None
        if not (np.isfinite(start) and np.isfinite(end)):
            raise AnnotationError(f"{path}:{lineno}: non-finite time in {raw!r}")
        label = default_label
        if label_column is not None:
            if label_column >= len(fields):
                raise AnnotationError(f"{path}:{lineno}: missing label column {label_column} in {raw!r}")
            name = fields[label_column]
            if not label_map or name not in label_map:
                raise AnnotationError(f"{path}:{lineno}: label {name!r} not in data.label_map {dict(label_map or {})}")
            label = int(label_map[name])
        if label <= 0:
            raise AnnotationError(f"{path}:{lineno}: event label must be >= 1 (0 is background)")
        rows.append(AnnotationRow(fields[0], Event(start, end, label), lineno))
    if not seen_section:
        raise AnnotationError(f"{path}: section [{section}] not found")
    return rows


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def validate_events(
    events: Sequence[Event],
    duration_sec: float,
    where: str,
    end_tolerance_sec: float = 1.0,
    overlapping: str = "error",
) -> Tuple[List[Event], List[str]]:
    """Check one recording's events; return (sorted events, warnings).

    * ``start < end`` and ``start >= 0`` are required.
    * ``end`` may exceed the recording by at most ``end_tolerance_sec`` (rounding in
      annotation tools); such events are clipped with a warning, anything beyond is an error.
    * Overlapping events are an error unless ``overlapping='merge'`` (same label only).
    """
    warnings: List[str] = []
    out: List[Event] = []
    for ev in sorted(events, key=lambda e: (e.start, e.end)):
        if ev.start < 0 or ev.end <= ev.start:
            raise AnnotationError(f"{where}: invalid event [{ev.start}, {ev.end}] s")
        if ev.start >= duration_sec:
            raise AnnotationError(f"{where}: event starts at {ev.start} s but recording lasts {duration_sec:.3f} s")
        if ev.end > duration_sec:
            if ev.end - duration_sec > end_tolerance_sec:
                raise AnnotationError(f"{where}: event ends at {ev.end} s, beyond recording end {duration_sec:.3f} s")
            warnings.append(f"{where}: event end {ev.end} s clipped to recording end {duration_sec:.3f} s")
            ev = Event(ev.start, duration_sec, ev.label)
        if out and ev.start < out[-1].end:
            prev = out[-1]
            if overlapping != "merge" or prev.label != ev.label:
                raise AnnotationError(f"{where}: overlapping events [{prev.start}, {prev.end}] and [{ev.start}, {ev.end}]")
            warnings.append(f"{where}: merged overlapping events [{prev.start}, {prev.end}] and [{ev.start}, {ev.end}]")
            out[-1] = Event(prev.start, max(prev.end, ev.end), prev.label)
            continue
        out.append(ev)
    return out, warnings


def events_to_intervals(events: Sequence[Event], fs: float, n_samples: int) -> np.ndarray:
    """Events -> int64 array of shape (E, 3): [start_idx, end_idx (exclusive), label].

    Uses Python's ``round`` (half-to-even), identical to timescoring's Annotation."""
    rows = []
    for ev in events:
        a = min(max(round(ev.start * fs), 0), n_samples)
        b = min(max(round(ev.end * fs), 0), n_samples)
        if b > a:
            rows.append((a, b, ev.label))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


def intervals_to_labels(intervals: np.ndarray, start: int, length: int, ignore_margin: int = 0) -> np.ndarray:
    """Per-sample class labels for ``[start, start + length)`` (int64, background = 0).

    ``ignore_margin`` > 0 marks samples within that many samples of an event
    boundary as ``IGNORE_INDEX`` (annotation onset/offset uncertainty)."""
    y = np.zeros(length, dtype=np.int64)
    stop = start + length
    for a, b, lab in intervals:
        lo, hi = max(a, start), min(b, stop)
        if hi > lo:
            y[lo - start:hi - start] = lab
    if ignore_margin > 0:
        for a, b, _ in intervals:
            for edge in (a, b):
                lo, hi = max(edge - ignore_margin, start), min(edge + ignore_margin, stop)
                if hi > lo:
                    y[lo - start:hi - start] = IGNORE_INDEX
    return y


def positive_counts(intervals: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """Number of event samples inside each window ``[s, s + length)`` (vectorized)."""
    starts = np.asarray(starts, dtype=np.int64)
    if len(intervals) == 0 or len(starts) == 0:
        return np.zeros(len(starts), dtype=np.int64)
    a = intervals[:, 0][None, :]
    b = intervals[:, 1][None, :]
    s = starts[:, None]
    overlap = np.clip(np.minimum(b, s + length) - np.maximum(a, s), 0, None)
    return overlap.sum(axis=1)


def group_rows_by_file(rows: Sequence[AnnotationRow]) -> Dict[str, List[AnnotationRow]]:
    out: Dict[str, List[AnnotationRow]] = {}
    for r in rows:
        out.setdefault(r.file, []).append(r)
    return out
