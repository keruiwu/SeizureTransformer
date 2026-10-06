import numpy as np
import pytest
import torch
from scipy.ndimage import binary_closing, binary_opening

from lteeg.config import PostprocessConfig, load_config
from lteeg.inference.longrange import plan_windows, predict_recording
from lteeg.inference.postprocess import fill_short_gaps, postprocess, remove_short_runs
from lteeg.task import Task


class Echo(torch.nn.Module):
    """Logit = first input channel: the stitched output must reproduce sigmoid(signal)."""

    def forward(self, x):
        return x[:, :1] * 1.0


@pytest.mark.parametrize("n,window,hop,tail", [(1000, 100, 100, "align"), (1000, 100, 100, "pad"),
                                               (1037, 100, 100, "align"), (1037, 100, 100, "pad"),
                                               (1037, 100, 30, "align"), (50, 100, 100, "align")])
def test_plan_covers_every_sample(n, window, hop, tail):
    starts = plan_windows(n, window, hop, tail)
    covered = np.zeros(n, bool)
    for s in starts:
        covered[s:s + window] = True
    assert covered.all()
    if tail == "align" and n >= window:
        assert max(starts) + window == n


@pytest.mark.parametrize("hop,combine,tail", [(100, "mean", "align"), (100, "mean", "pad"),
                                              (40, "hann", "align"), (25, "mean", "pad")])
def test_stitching_is_exact_for_position_independent_model(hop, combine, tail):
    cfg = load_config(None)
    sig = np.random.default_rng(0).standard_normal((18, 1037)).astype(np.float32)
    probs = predict_recording(Echo(), sig, Task(cfg), window=100, hop=hop, batch_size=3,
                              device=torch.device("cpu"), combine=combine, tail=tail)
    assert probs.shape == (1, 1037)
    assert np.allclose(probs[0], 1 / (1 + np.exp(-sig[0])), atol=1e-6)


def test_run_length_morphology_matches_scipy_in_interior():
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = rng.random(400) > 0.6
        m[:12] = False
        m[-12:] = False  # keep borders empty: scipy erodes runs touching the border
        for k in (2, 3, 5, 8):
            st = np.ones(k, bool)
            assert np.array_equal(remove_short_runs(m, k), binary_opening(m, structure=st))
            assert np.array_equal(fill_short_gaps(m, k), binary_closing(m, structure=st))


def original_remove_short_events(binary_output, min_length, fs):
    min_samples = int(min_length * fs)
    out = binary_output.copy()
    is_seizure, start_idx = False, 0
    for i in range(len(binary_output)):
        if not is_seizure and out[i] == 1:
            is_seizure, start_idx = True, i
        elif is_seizure and (out[i] == 0 or i == len(binary_output) - 1):
            end_idx = i if out[i] == 0 else i + 1
            if end_idx - start_idx < min_samples:
                out[start_idx:end_idx] = 0
            is_seizure = False
    return out


def test_min_duration_matches_original():
    rng = np.random.default_rng(1)
    for _ in range(20):
        m = (rng.random(3000) > 0.97).astype(int)
        m = np.convolve(m, np.ones(rng.integers(1, 40)), mode="same") > 0
        ours = remove_short_runs(m, int(0.1 * 256))
        ref = original_remove_short_events(m.astype(int), 0.1, 256).astype(bool)
        assert np.array_equal(ours, ref)


def test_postprocess_pipeline():
    prob = np.zeros(2000)
    prob[100:103] = 0.9  # too short for the opening (k=5)
    prob[500:1200] = 0.95
    prob[800:802] = 0.1  # small hole, closed
    prob[1500:1700] = 0.85  # 200 samples < 2 s at 256 Hz -> removed
    mask = postprocess(prob, 256.0, PostprocessConfig())
    assert mask[500:1200].all() and not mask[:500].any() and not mask[1200:].any()
