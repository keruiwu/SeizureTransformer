import numpy as np
import pytest

from lteeg.config import ScoringConfig
from lteeg.evaluation.ranking import average_precision, bin_mean, roc_auc
from lteeg.evaluation.szcore import Counts, event_scoring, events_from_mask, merge_events, sample_scoring, split_events

FS = 256


def _rand_mask(rng, n, k, maxlen):
    m = np.zeros(n, bool)
    for _ in range(k):
        a = int(rng.integers(0, n))
        m[a:a + int(rng.integers(1, maxlen))] = True
    return m


def test_matches_timescoring_reference():
    pytest.importorskip("timescoring")
    from timescoring import scoring
    from timescoring.annotations import Annotation

    rng = np.random.default_rng(0)
    for trial in range(150):
        n = int(rng.integers(FS * 60, FS * 3600))
        ref = _rand_mask(rng, n, int(rng.integers(0, 4)), FS * 400)
        hyp = _rand_mask(rng, n, int(rng.integers(0, 12)), FS * 200)
        if trial % 9 == 0:
            ref[: FS * 5] = True
            hyp[-FS * 3:] = True
        R, H = Annotation(ref, FS), Annotation(hyp, FS)
        s_ref, e_ref = scoring.SampleScoring(R, H), scoring.EventScoring(R, H)
        re, he = events_from_mask(ref, FS), events_from_mask(hyp, FS)
        s = sample_scoring(re, he, n, FS, 1.0)
        e, _, _ = event_scoring(re, he, n, FS, ScoringConfig())
        assert (s.tp, s.fp, s.ref_true, s.num_samples) == (s_ref.tp, s_ref.fp, s_ref.refTrue, s_ref.numSamples)
        assert (e.tp, e.fp, e.ref_true, e.num_samples) == (e_ref.tp, e_ref.fp, e_ref.refTrue, e_ref.numSamples)
        assert np.isclose(e.fp_per_day, e_ref.fpRate)


def test_event_scoring_rules():
    p = ScoringConfig()
    n = 3600 * FS
    ref = [(1000.0, 1060.0)]
    # detection 25 s before onset is inside the 30 s pre-ictal tolerance
    c, det, fps = event_scoring(ref, [(975.0, 980.0)], n, FS, p)
    assert (c.tp, c.fp, c.ref_true) == (1, 0, 1) and det[0]["latency_sec"] == pytest.approx(-25.0)
    # 61 s after offset is outside the 60 s post-ictal tolerance -> miss + false alarm
    c, _, fps = event_scoring(ref, [(1121.0, 1130.0)], n, FS, p)
    assert (c.tp, c.fp) == (0, 1) and fps[0]["start"] == 1121.0
    # two hypotheses 80 s apart are merged into one event (< 90 s)
    c, _, _ = event_scoring([], [(100.0, 110.0), (190.0, 200.0)], n, FS, p)
    assert c.fp == 1
    # a 12-minute reference event is split into 3 events (300 s max)
    assert split_events([(0.0, 720.0)], 300.0) == [(0.0, 300.0), (300.0, 600.0), (600.0, 720.0)]
    assert merge_events([(0.0, 10.0), (50.0, 60.0), (200.0, 210.0)], 90.0) == [(0.0, 60.0), (200.0, 210.0)]


def test_counts_metrics():
    c = Counts(tp=3, fp=2, ref_true=4, num_samples=24 * 3600 * 10, fs=10)
    assert c.sensitivity == 0.75 and c.precision == 0.6 and c.fp_per_day == pytest.approx(2.0)
    assert c.f1 == pytest.approx(2 * 3 / (2 * 3 + 2 + 1))
    assert np.isnan(Counts(num_samples=10, fs=1).f1)
    total = c + Counts(tp=1, fp=0, ref_true=1, num_samples=24 * 3600 * 10, fs=10)
    assert total.fp_per_day == pytest.approx(1.0)


def test_sample_scoring_one_hz_grid():
    n = 100 * FS
    s = sample_scoring([(10.0, 20.0)], [(15.0, 30.0)], n, FS, 1.0)
    assert (s.tp, s.fp, s.ref_true, s.num_samples) == (5, 10, 10, 100)


def test_ranking_metrics_vs_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.05
    s = np.round(rng.random(5000) + y * 0.5, 2)  # ties included
    assert roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s))
    assert average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s))
    assert np.isnan(roc_auc(np.zeros(10, bool), np.arange(10)))


def test_bin_mean():
    v = np.arange(256 * 3 + 200, dtype=np.float32)
    b = bin_mean(v, 256, 1)
    assert len(b) == 4  # round(968 / 256) = 4 bins, the last one partial
    assert b[0] == pytest.approx(v[:256].mean()) and b[3] == pytest.approx(v[768:].mean())
