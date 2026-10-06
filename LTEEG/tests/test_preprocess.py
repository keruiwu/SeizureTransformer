import numpy as np
import pytest
from scipy.signal import butter, iirnotch, lfilter

from lteeg.data.preprocess import Pipeline

FS = 256.0
ORIGINAL = [{"op": "zscore"}, {"op": "bandpass", "low": 0.5, "high": 120.0, "order": 3},
            {"op": "notch", "freq": 1.0, "q": 30.0}, {"op": "notch", "freq": 60.0, "q": 30.0}]


def original_chain(x):
    """The original SeizureTransformer preprocessing (b/a form, lfilter)."""
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    b, a = butter(3, [0.5 / (FS / 2), 120 / (FS / 2)], btype="band")
    x = lfilter(b, a, x)
    for f0 in (1.0, 60.0):
        b, a = iirnotch(f0, Q=30, fs=FS)
        x = lfilter(b, a, x)
    return x


def test_fused_sos_pipeline_matches_original_filter_chain():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 60 * 256)) * 50 + 10
    pipe = Pipeline(ORIGINAL, FS)
    assert sum(1 for kind, _ in pipe._plan if kind == "sos") == 1  # the three filters were fused
    y = pipe(x)
    ref = original_chain(x)
    assert np.max(np.abs(y - ref)) < 1e-6 * np.max(np.abs(ref))


def test_zscore_flat_channel_is_zero_and_reported():
    x = np.random.default_rng(0).standard_normal((3, 1000))
    x[1] = 5.0
    info = {}
    y = Pipeline([{"op": "zscore"}], FS)(x, info)
    assert info["flat_channels"] == [1]
    assert np.all(y[1] == 0) and np.isfinite(y).all()
    assert np.allclose(y[0].std(), 1.0)


def test_invalid_frequency_rejected():
    with pytest.raises(ValueError, match="Nyquist"):
        Pipeline([{"op": "bandpass", "low": 0.5, "high": 120.0}], 200.0)


def test_unknown_param_rejected():
    from lteeg.registry import RegistryError

    with pytest.raises(RegistryError, match="unexpected parameter"):
        Pipeline([{"op": "notch", "freq": 60, "quality": 30}], FS)


def test_zero_phase_has_no_lag():
    t = np.arange(4096) / FS
    x = np.sin(2 * np.pi * 10 * t)[None]
    y = Pipeline([{"op": "bandpass", "low": 5.0, "high": 20.0, "zero_phase": True}], FS)(x)
    mid = slice(1000, 3000)
    assert np.max(np.abs(y[0, mid] - x[0, mid])) < 0.05
