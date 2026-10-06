import json
import shutil

import h5py
import numpy as np
import pytest

from lteeg.config import CHBMIT_CHANNELS, load_config
from lteeg.data.loading import load_recordings
from lteeg.data.store import DataValidationError, H5Store


def _copy(src, dst):
    shutil.copytree(src, dst)
    return dst


def test_scan_synthetic(synthetic_cfg):
    cfg = synthetic_cfg()
    bundle = load_recordings(cfg, ["train", "dev"])
    assert len(bundle.recordings["train"]) == 6 and len(bundle.recordings["dev"]) == 2
    rec = bundle.recordings["train"][0]
    assert rec.key == "chb01/chb01_01" and rec.fs == 256.0 and rec.n_samples == 4 * 60 * 256
    x = bundle.store.load_signal(rec)
    assert x.shape == (18, rec.n_samples) and x.dtype == np.float64


def test_expected_count_mismatch(synthetic_root, tmp_path):
    root = _copy(synthetic_root, tmp_path / "d")
    split = json.loads((root / "split.json").read_text())
    split["expected_seizures"]["train"] += 1
    (root / "split.json").write_text(json.dumps(split))
    cfg = load_config(None, [f"data.root={root}", f"data.split_file={root / 'split.json'}"])
    with pytest.raises(DataValidationError, match="expects"):
        load_recordings(cfg, ["train"])


def test_annotation_for_missing_file(synthetic_root, tmp_path):
    root = _copy(synthetic_root, tmp_path / "d")
    ann = root / "chb01" / "chb01_annotations.txt"
    ann.write_text(ann.read_text() + "chb01_99.h5\t10\t20\n")
    cfg = load_config(None, [f"data.root={root}"])
    with pytest.raises(DataValidationError, match="chb01_99.h5"):
        H5Store(cfg).scan(["chb01"])


def test_exact_vs_stem_matching(synthetic_root, tmp_path):
    root = _copy(synthetic_root, tmp_path / "d")
    ann = root / "chb02" / "chb02_annotations.txt"
    text = ann.read_text().replace(".h5\t", ".edf\t")
    ann.write_text(text)
    cfg = load_config(None, [f"data.root={root}"])
    if ".edf" in text:
        with pytest.raises(DataValidationError, match="annotation_match=stem"):
            H5Store(cfg).scan(["chb02"])
        cfg = load_config(None, [f"data.root={root}", "data.annotation_match=stem"])
        recs = H5Store(cfg).scan(["chb02"]).recordings
        assert sum(r.n_events for r in recs) == text.count(".edf\t")


def test_event_beyond_recording(synthetic_root, tmp_path):
    root = _copy(synthetic_root, tmp_path / "d")
    ann = root / "chb03" / "chb03_annotations.txt"
    ann.write_text(ann.read_text() + "chb03_01.h5\t230\t400\n")
    cfg = load_config(None, [f"data.root={root}"])
    with pytest.raises(DataValidationError, match="beyond recording end"):
        H5Store(cfg).scan(["chb03"])


def _write_h5(path, x, fs=256.0, names=None):
    with h5py.File(path, "w") as f:
        f.create_dataset("signals", data=x)
        f.attrs["fs"] = fs
        if names is not None:
            f.attrs["channels"] = np.array([n.encode() for n in names])


def _single(tmp_path, x, fs=256.0, names=None, ann=""):
    pdir = tmp_path / "root" / "p01"
    pdir.mkdir(parents=True)
    _write_h5(pdir / "p01_01.h5", x, fs, names)
    (pdir / "p01_annotations.txt").write_text("[seizures]\nfile\tstart\tend\n" + ann)
    return load_config(None, [f"data.root={tmp_path / 'root'}"])


def test_transposed_and_fs_and_count_errors(tmp_path):
    rng = np.random.default_rng(0)
    cfg = _single(tmp_path, rng.standard_normal((1000, 18)).astype(np.float32))
    with pytest.raises(DataValidationError, match="transposed"):
        H5Store(cfg).scan(["p01"])
    cfg = _single(tmp_path / "b", rng.standard_normal((18, 1000)).astype(np.float32), fs=200.0)
    with pytest.raises(DataValidationError, match="sampling rate"):
        H5Store(cfg).scan(["p01"])
    cfg = _single(tmp_path / "c", rng.standard_normal((17, 1000)).astype(np.float32))
    with pytest.raises(DataValidationError, match="17 signal rows"):
        H5Store(cfg).scan(["p01"])


def test_channel_names_reorder_and_missing(tmp_path):
    rng = np.random.default_rng(0)
    names = list(CHBMIT_CHANNELS)[::-1] + ["ECG"]
    x = rng.standard_normal((19, 512)).astype(np.float32)
    cfg = _single(tmp_path, x, names=[f"EEG {n}" for n in names])
    store = H5Store(cfg)
    rep = store.scan(["p01"])
    assert rep.channel_names_verified == 1
    y = store.load_signal(rep.recordings[0])
    # configured channel k must be file row names.index(channel k)
    for k, ch in enumerate(CHBMIT_CHANNELS):
        assert np.allclose(y[k], x[names.index(ch)])
    cfg = _single(tmp_path / "b", x[:18], names=names[:17] + ["XX"])
    with pytest.raises(DataValidationError, match="not found in file channels"):
        H5Store(cfg).scan(["p01"])


def test_nonfinite_signal_rejected(tmp_path):
    x = np.zeros((18, 512), np.float32)
    x[3, 100] = np.nan
    cfg = _single(tmp_path, x)
    store = H5Store(cfg)
    rec = store.scan(["p01"]).recordings[0]
    with pytest.raises(DataValidationError, match="non-finite"):
        store.load_signal(rec)


def test_resampling(tmp_path):
    x = np.random.default_rng(0).standard_normal((18, 5120)).astype(np.float32)
    cfg = _single(tmp_path, x, fs=512.0, ann="p01_01.h5\t2\t4\n")
    cfg = load_config(None, [f"data.root={cfg.data.root}", "data.resample_to=256"])
    store = H5Store(cfg)
    rec = store.scan(["p01"]).recordings[0]
    assert rec.n_samples_at(256.0) == 2560
    assert store.load_signal(rec).shape == (18, 2560)
    assert rec.intervals(256.0).tolist() == [[512, 1024, 1]]
