"""Patient-level data splits.

Split file (JSON)::

    {
      "train": ["chb01", ...], "dev": ["chb05", ...], "test": [],
      "groups": [["chb01", "chb21"]],          # same subject: must stay in one split
      "expected_seizures": {"train": 159, "dev": 39}   # optional sanity check
    }

or, for patient-level cross-validation, ``{"folds": [{"train": [...], "dev": [...]}, ...],
"test": [...], ...}`` together with ``data.fold``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..utils import read_json

SPLIT_NAMES = ("train", "dev", "test")


@dataclass
class Split:
    train: List[str]
    dev: List[str]
    test: List[str] = field(default_factory=list)
    groups: List[List[str]] = field(default_factory=list)
    expected_seizures: Dict[str, int] = field(default_factory=dict)
    source: str = ""

    def patients(self, name: str) -> List[str]:
        if name not in SPLIT_NAMES:
            raise KeyError(f"Unknown split {name!r}; expected one of {SPLIT_NAMES}")
        return list(getattr(self, name))

    def all_patients(self) -> List[str]:
        return self.train + self.dev + self.test

    def to_dict(self) -> dict:
        return {"train": self.train, "dev": self.dev, "test": self.test, "groups": self.groups,
                "expected_seizures": self.expected_seizures, "source": self.source}


def load_split(path: Path, fold: Optional[int] = None) -> Split:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    raw = read_json(path)
    if "folds" in raw:
        folds = raw["folds"]
        if fold is None:
            raise ValueError(f"{path} defines {len(folds)} folds; set data.fold")
        if not 0 <= fold < len(folds):
            raise ValueError(f"data.fold={fold} out of range for {len(folds)} folds")
        part = dict(folds[fold])
        part.setdefault("test", raw.get("test", []))
        expected = part.get("expected_seizures", {})
    else:
        if fold is not None:
            raise ValueError(f"data.fold={fold} given but {path} has no 'folds'")
        part = raw
        expected = raw.get("expected_seizures", {}) or {}
    unknown = set(part) - {"train", "dev", "test", "groups", "expected_seizures", "description", "folds"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    split = Split(
        train=[str(p) for p in part.get("train", [])],
        dev=[str(p) for p in part.get("dev", [])],
        test=[str(p) for p in part.get("test", [])],
        groups=[[str(p) for p in g] for g in raw.get("groups", [])],
        expected_seizures={k: int(v) for k, v in (expected or {}).items()},
        source=str(path) + (f"#fold{fold}" if fold is not None else ""),
    )
    _check_split(split)
    return split


def _check_split(split: Split) -> None:
    errors = []
    seen: Dict[str, str] = {}
    for name in SPLIT_NAMES:
        members = split.patients(name)
        if len(set(members)) != len(members):
            errors.append(f"duplicate patients inside '{name}'")
        for p in members:
            if p in seen:
                errors.append(f"patient {p} is in both '{seen[p]}' and '{name}'")
            seen[p] = name
    for group in split.groups:
        homes = {seen.get(p) for p in group if p in seen}
        if len(homes) > 1:
            errors.append(f"group {group} (same subject) is spread over splits {sorted(h for h in homes if h)}")
    if not split.train:
        errors.append("train split is empty")
    for k in split.expected_seizures:
        if k not in SPLIT_NAMES:
            errors.append(f"expected_seizures has unknown split {k!r}")
    if errors:
        raise ValueError(f"Invalid split {split.source}:\n  - " + "\n  - ".join(errors))
