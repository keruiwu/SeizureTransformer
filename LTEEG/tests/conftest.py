from __future__ import annotations

import json
from pathlib import Path

import pytest

from lteeg.config import load_config
from lteeg.synthetic import make_synthetic_dataset


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    """3 train + 1 dev patients, 2 recordings of 4 minutes each."""
    root = tmp_path_factory.mktemp("synthetic")
    patients = ["chb01", "chb02", "chb03", "chb04"]
    manifest = make_synthetic_dataset(root, patients, records_per_patient=2, record_minutes=4.0,
                                      seizures_per_record=(0, 1), seizure_sec=(20, 50), seed=1)
    n = {p: sum(len(r["events"]) for r in manifest[p]) for p in patients}
    split = {"train": patients[:3], "dev": patients[3:], "test": [],
             "expected_seizures": {"train": n["chb01"] + n["chb02"] + n["chb03"], "dev": n["chb04"]}}
    (root / "split.json").write_text(json.dumps(split), encoding="utf-8")
    return root


@pytest.fixture
def synthetic_cfg(synthetic_root, tmp_path):
    def make(*overrides):
        return load_config(None, [f"data.root={synthetic_root}", f"data.split_file={synthetic_root / 'split.json'}",
                                  f"data.cache_dir={tmp_path / 'cache'}", f"experiment.output_dir={tmp_path / 'runs'}",
                                  *overrides])
    return make
