import numpy as np

from lteeg.data.annotations import Event
from lteeg.data.store import Recording
from lteeg.data.windows import BACKGROUND, BOUNDARY, FULL, WindowSampler, build_window_index, grid_starts
from lteeg.config import load_config

FS = 256.0


def rec(patient, name, minutes, events):
    return Recording(patient, name, f"/x/{name}", FS, int(minutes * 60 * FS), tuple(Event(a, b) for a, b in events))


def recordings():
    return [rec("p1", "p1_01.h5", 30, [(300, 340), (900, 1100)]), rec("p1", "p1_02.h5", 20, []),
            rec("p2", "p2_01.h5", 30, [(100, 130)]), rec("p2", "p2_02.h5", 0.5, [])]


def test_grid_starts():
    assert grid_starts(100, 30, 10).tolist() == [0, 10, 20, 30, 40, 50, 60, 70]
    assert grid_starts(105, 30, 10, include_tail=True).tolist()[-1] == 75
    assert len(grid_starts(20, 30, 10)) == 0


def test_categories_match_bruteforce():
    cfg = load_config(None)
    recs = recordings()
    idx = build_window_index(recs, FS, cfg.window_samples, cfg.train_stride_samples)
    assert 3 not in set(idx.rec.tolist())  # 30 s recording is shorter than the 60 s window
    for i in range(len(idx)):
        r = recs[idx.rec[i]]
        mask = np.zeros(r.n_samples, bool)
        for a, b, _ in r.intervals(FS):
            mask[a:b] = True
        n = mask[idx.start[i]:idx.start[i] + idx.window].sum()
        expected = BACKGROUND if n == 0 else FULL if n == idx.window else BOUNDARY
        assert idx.category[i] == expected and idx.positive[i] == n


def test_balanced_quota_and_determinism():
    cfg = load_config(None)
    recs = recordings()
    idx = build_window_index(recs, FS, cfg.window_samples, cfg.train_stride_samples)
    s1, s2 = WindowSampler(cfg, recs, idx), WindowSampler(cfg, recs, idx)
    counts = idx.counts()
    sel = idx.counts(np.isin(np.arange(len(idx)), s1.selection(0)))
    assert sel["boundary"] == counts["boundary"]
    assert sel["full"] == min(int(counts["boundary"] * 0.7), counts["full"])
    assert sel["background"] == min(int(counts["boundary"] * 3), counts["background"])
    assert s1.epoch_items(0) == s2.epoch_items(0)  # deterministic in the seed
    assert s1.epoch_items(0) != s1.epoch_items(1)  # reshuffled every epoch
    assert sorted(s1.epoch_items(0)) == sorted(s1.epoch_items(1))  # same subset without redraw


def test_redraw_jitter_and_patient_grouping():
    recs = recordings()
    cfg = load_config(None, ["sampling.redraw_every_epoch=true", "sampling.jitter_sec=5",
                             "sampling.background_ratio=0.5"])
    idx = build_window_index(recs, FS, cfg.window_samples, cfg.train_stride_samples)
    s = WindowSampler(cfg, recs, idx)
    a, b = s.epoch_items(0), s.epoch_items(1)
    assert sorted(a) != sorted(b)
    for r, start in a:
        assert 0 <= start <= recs[r].n_samples - cfg.window_samples
    cfg = load_config(None, ["sampling.group_by=patient"])
    s = WindowSampler(cfg, recs, idx)
    sel = s.selection(0)
    for p in (0, 1):
        m = s.patient_of_rec[idx.rec] == p
        n_b = int(((idx.category == BOUNDARY) & m).sum())
        chosen_bg = int(((idx.category == BACKGROUND) & m)[sel].sum())
        assert chosen_bg == min(3 * n_b, int(((idx.category == BACKGROUND) & m).sum()))
