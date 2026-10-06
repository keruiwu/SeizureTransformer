"""One entry point that turns a config into validated recordings per split."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ..config import Config
from ..utils.logging import get_logger
from .split import Split, load_split
from .store import DataValidationError, H5Store, Recording, summarize, totals

log = get_logger("data")


@dataclass
class DataBundle:
    split: Split
    store: H5Store
    recordings: Dict[str, List[Recording]] = field(default_factory=dict)
    channel_names_verified: int = 0

    def summary(self) -> Dict[str, dict]:
        out = {}
        for name, recs in self.recordings.items():
            per_patient = summarize(recs, self.store.fs_out)
            out[name] = {"totals": totals(per_patient), "patients": per_patient}
        return out


def load_recordings(cfg: Config, splits: Sequence[str] = ("train", "dev"), check_expected: bool = True) -> DataBundle:
    split = load_split(cfg.split_path, cfg.data.fold)
    store = H5Store(cfg)
    available = set(store.list_patients())
    missing = [p for p in split.all_patients() if p not in available]
    if missing:
        raise DataValidationError(f"patients listed in {split.source} but missing under {store.root}: {missing}")
    unused = sorted(available - set(split.all_patients()))
    if unused:
        log.info(f"patients present on disk but not in any split (ignored): {unused}")
    bundle = DataBundle(split=split, store=store)
    for name in splits:
        patients = split.patients(name)
        if not patients:
            bundle.recordings[name] = []
            continue
        report = store.scan(patients)
        bundle.recordings[name] = report.recordings
        bundle.channel_names_verified += report.channel_names_verified
        n_events = sum(r.n_events for r in report.recordings)
        hours = sum(r.duration_sec for r in report.recordings) / 3600
        log.info(f"[{name}] {len(patients)} patients, {len(report.recordings)} recordings, {hours:.1f} h, "
                 f"{n_events} annotated events")
        expected = split.expected_seizures.get(name)
        if check_expected and expected is not None and expected != n_events:
            per_patient = {p: s["events"] for p, s in summarize(report.recordings, cfg.fs).items()}
            raise DataValidationError(
                f"split '{name}' has {n_events} annotated events but the split file expects {expected}. "
                f"Per patient: {per_patient}. Fix the annotations or update 'expected_seizures' in {split.source}.")
    if bundle.channel_names_verified == 0:
        log.warning("No channel-name attribute found in the HDF5 files: channel ORDER cannot be verified "
                    "and is assumed to match data.channels. Store names in an attribute (e.g. 'channels') "
                    "to have them checked.")
    return bundle
