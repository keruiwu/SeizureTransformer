"""Full pipeline on the synthetic dataset: cache -> sampler -> train -> long-range eval -> sweep -> predict."""

import csv
import pickle

import numpy as np
import pytest

from lteeg import workflows
from lteeg.data.cache import SignalCache
from lteeg.data.datasets import TrainWindowDataset
from lteeg.data.loading import load_recordings
from lteeg.utils import read_json

FAST = ["model.name=tcn", "model.params={hidden: 16, levels: 4}", "windows.length_sec=10",
        "windows.train_stride_sec=5", "train.epochs=2", "train.batch_size=8", "train.log_every=0",
        "postprocess.threshold=0.5", "evaluation.sweep_thresholds=[0.3, 0.5]"]


def test_train_evaluate_sweep_predict(synthetic_cfg, tmp_path):
    cfg = synthetic_cfg(*FAST)
    run_dir = workflows.train(cfg)
    for name in ("config.yaml", "env.json", "split.json", "data_summary.json", "sampling_summary.json",
                 "train_log.csv", "result.json", "checkpoints/best.pt", "checkpoints/last.pt",
                 "eval/dev_best/patients.csv", "eval/dev_best/records.csv", "eval/dev_best/summary.json",
                 "eval/dev_best/summary.md", "eval/dev_best/events.csv", "eval/dev_best/sweep.csv",
                 "eval/dev_best/manifest.json"):
        assert (run_dir / name).exists(), name
    with open(run_dir / "train_log.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and "val_event_f1_pooled" in rows[0]
    summary = read_json(run_dir / "eval/dev_best/summary.json")
    for key in ("event_f1_pooled", "event_f1_macro", "event_fp_per_day_pooled", "sample_f1_pooled",
                "auroc_pooled", "auprc_pooled", "nll_pooled"):
        assert key in summary["metrics"]
    probs = list((run_dir / "eval/dev_best/probs").rglob("*.npy"))
    assert len(probs) == 2 and np.load(probs[0]).dtype == np.float16

    # resume for one more epoch
    cfg2 = workflows.load_config(run_dir / "config.yaml", ["train.epochs=3"])
    workflows.train(cfg2, resume=run_dir)
    assert read_json(run_dir / "result.json")["epochs_run"] == 3

    # stand-alone evaluation of last.pt with overrides, then prediction
    workflows.evaluate(run_dir / "checkpoints/last.pt", ["inference.hop_sec=5", "inference.combine=hann"],
                       ["dev"], run_sweep=False)
    assert (run_dir / "eval/dev_last/summary.json").exists()
    out = tmp_path / "pred"
    workflows.predict(run_dir / "checkpoints/last.pt", [cfg.resolve_path(cfg.data.root) / "chb04"], out)
    tsv = sorted(out.glob("*_events.tsv"))
    assert len(tsv) == 2 and tsv[0].read_text().startswith("onset\tduration")


def test_training_options_smoke(synthetic_cfg):
    cfg = synthetic_cfg(*FAST, "train.epochs=1", "train.ema_decay=0.99", "train.grad_clip=1.0",
                        "scheduler.name=cosine", "scheduler.warmup_epochs=0.5", "optimizer.name=adamw",
                        "train.accum_steps=2", "sampling.redraw_every_epoch=true", "sampling.jitter_sec=2",
                        "sampling.group_by=patient", "task.ignore_boundary_sec=1",
                        "loss.terms=[{name: focal}, {name: tmse, weight: 0.15}]",
                        "preprocess.window=[{op: zscore}]", "train.amp=bf16",
                        "train.augment=[{name: amplitude_scale}, {name: sign_flip}, {name: hemisphere_swap},"
                        " {name: gaussian_noise}, {name: channel_dropout}, {name: time_mask}]",
                        "evaluation.save_probs=false")
    run_dir = workflows.train(cfg)
    assert (run_dir / "checkpoints/best.pt").exists()


def test_seizure_transformer_trains(synthetic_cfg):
    cfg = synthetic_cfg("model.params={num_layers: 1, dim_feedforward: 64}", "windows.length_sec=10",
                        "windows.train_stride_sec=5", "train.epochs=1", "train.batch_size=4", "train.log_every=0",
                        "evaluation.splits=[]")
    run_dir = workflows.train(cfg)
    assert (run_dir / "checkpoints/best.pt").exists()


def test_dataset_pickles_for_spawned_workers(synthetic_cfg):
    """Windows DataLoader workers are spawned: the dataset (incl. cache) must pickle."""
    cfg = synthetic_cfg()
    bundle = load_recordings(cfg, ["train"])
    recs = bundle.recordings["train"]
    cache = SignalCache(cfg, bundle.store)
    cache.ensure(recs)
    ds = TrainWindowDataset(cfg, recs, cache, ["chb01", "chb02", "chb03"])
    item = ds[(0, 0)]
    clone = pickle.loads(pickle.dumps(ds))
    assert np.array_equal(clone[(0, 0)]["x"].numpy(), item["x"].numpy())
    assert item["x"].shape == (18, cfg.window_samples) and item["y"].shape == (cfg.window_samples,)


def test_cache_detects_stale_source(synthetic_cfg, synthetic_root):
    cfg = synthetic_cfg()
    bundle = load_recordings(cfg, ["dev"])
    rec = bundle.recordings["dev"][0]
    cache = SignalCache(cfg, bundle.store)
    cache.ensure([rec])
    assert cache.is_valid(rec)
    import dataclasses

    assert not cache.is_valid(dataclasses.replace(rec, file_size=rec.file_size + 1))
    cfg2 = synthetic_cfg("data.cache_dtype=float16")
    assert SignalCache(cfg2).dir != cache.dir  # different preprocessing -> different fingerprint


def test_dry_run_and_existing_dir(synthetic_cfg):
    cfg = synthetic_cfg(*FAST, "experiment.timestamp=false")
    run_dir = workflows.train(cfg, dry_run=True)
    assert (run_dir / "sampling_summary.json").exists()
    with pytest.raises(FileExistsError):
        workflows.train(synthetic_cfg(*FAST, "experiment.timestamp=false"))
