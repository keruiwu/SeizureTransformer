"""High-level workflows behind the CLI commands (usable from Python as well)."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import Config, load_config
from .data.cache import InMemorySource, SignalCache
from .data.loading import DataBundle, load_recordings
from .data.store import H5Store, Recording, summarize, totals
from .data.windows import WindowSampler, build_window_index
from .engine.trainer import Trainer
from .engine.utils import environment_info, load_checkpoint, resolve_device
from .evaluation.report import console_table, short_summary, write_report
from .evaluation.runner import evaluate_recordings
from .evaluation.sweep import sweep, write_manifest
from .inference.longrange import predict_recording
from .inference.postprocess import mask_to_events, postprocess
from .models import build_model, count_parameters, describe_model
from .task import Task
from .utils import natural_key, seed_everything, write_json
from .utils.logging import get_logger, setup_logging

log = get_logger("workflow")


# ---------------------------------------------------------------------- helpers
def absolutize_paths(cfg: Config) -> Config:
    """Store absolute paths so a run directory can be evaluated from any working directory."""
    d = cfg.data
    cfg.data = dataclasses.replace(d, root=str(cfg.resolve_path(d.root)), split_file=str(cfg.split_path),
                                   cache_dir=str(cfg.resolve_path(d.cache_dir)))
    cfg.experiment = dataclasses.replace(cfg.experiment, output_dir=str(cfg.resolve_path(cfg.experiment.output_dir)))
    return cfg


def make_run_dir(cfg: Config) -> Path:
    base = Path(cfg.experiment.output_dir)
    name = cfg.experiment.name
    if cfg.experiment.timestamp:
        name = f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = base / name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory {run_dir} is not empty; use --resume, another experiment.name, "
                              f"or experiment.timestamp=true")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def data_report_text(bundle: DataBundle) -> str:
    lines = []
    cols = ["patient", "records", "records_with_events", "hours", "events", "event_seconds", "event_fraction",
            "event_duration_min", "event_duration_median", "event_duration_max"]
    for name, recs in bundle.recordings.items():
        if not recs:
            continue
        per = summarize(recs, bundle.store.fs_out)
        rows = [{"patient": p, **s} for p, s in sorted(per.items(), key=lambda kv: natural_key(kv[0]))]
        tot = totals(per)
        rows.append({"patient": "TOTAL", "records": tot["records"], "hours": tot["hours"], "events": tot["events"],
                     "event_seconds": tot["event_seconds"], "event_fraction": tot["event_fraction"]})
        lines.append(f"== split '{name}' ==\n" + console_table(rows, cols))
    return "\n\n".join(lines)


def load_eval_model(checkpoint: Path, cfg: Config, device: torch.device, use_ema: bool = True) -> torch.nn.Module:
    state = load_checkpoint(checkpoint)
    model = build_model(cfg, Task(cfg).num_outputs)
    weights = state["ema"]["module"] if (use_ema and state.get("ema")) else state["model"]
    model.load_state_dict(weights)
    return model.to(device).eval()


def config_from_checkpoint(checkpoint: Path, overrides: Sequence[str] = ()) -> Config:
    state = load_checkpoint(checkpoint)
    return load_config(None, list(overrides), base=state["config"])


# ---------------------------------------------------------------------- inspect / prepare
def inspect_data(cfg: Config, splits: Sequence[str] = ("train", "dev", "test"), out: Optional[Path] = None) -> DataBundle:
    bundle = load_recordings(cfg, splits)
    print(data_report_text(bundle))
    train = bundle.recordings.get("train") or []
    if train:
        index = build_window_index(train, cfg.fs, cfg.window_samples, cfg.train_stride_samples,
                                   cfg.windows.include_tail)
        sampler = WindowSampler(cfg, train, index)
        s = sampler.summary()
        print(f"\nTraining windows ({cfg.windows.length_sec:g} s, stride {cfg.windows.train_stride_sec:g} s):")
        print(f"  available: {s['available']}")
        print(f"  selected per epoch: {s['selected_epoch0']} -> {s['windows_per_epoch']} windows, "
              f"event-sample fraction {s['positive_sample_fraction_epoch0']:.3f}")
    print(f"\nChannel names verified from file attributes in {bundle.channel_names_verified} recording(s).")
    print("All validation checks passed.")
    if out:
        write_json(out, {"split": bundle.split.to_dict(), "summary": bundle.summary()})
    return bundle


def prepare_cache(cfg: Config, splits: Sequence[str], jobs: int = 1, force: bool = False) -> None:
    bundle = load_recordings(cfg, splits)
    recs = [r for s in splits for r in bundle.recordings.get(s, [])]
    SignalCache(cfg, bundle.store).ensure(recs, jobs=jobs, force=force)


# ---------------------------------------------------------------------- train
def train(cfg: Config, resume: Optional[Path] = None, jobs: int = 1, dry_run: bool = False) -> Path:
    if resume is not None:
        run_dir = Path(resume)
        ckpt = run_dir / "checkpoints" / "last.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"no checkpoint to resume at {ckpt}")
    else:
        cfg = absolutize_paths(cfg)
        run_dir = make_run_dir(cfg)
    setup_logging(run_dir / "train.log")
    seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)
    log.info(f"Run directory: {run_dir}")
    cfg.dump(run_dir / "config.yaml")
    write_json(run_dir / "env.json", environment_info())

    eval_splits = [s for s in cfg.evaluation.splits if s not in ("train", "dev")]
    bundle = load_recordings(cfg, ["train", "dev"] + eval_splits)
    write_json(run_dir / "split.json", bundle.split.to_dict())
    write_json(run_dir / "data_summary.json", bundle.summary())
    log.info("Data summary:\n" + data_report_text(bundle))
    train_recs, dev_recs = bundle.recordings["train"], bundle.recordings["dev"]
    if not dev_recs:
        log.warning("dev split is empty: no validation, the last epoch is used as 'best'")

    cache = SignalCache(cfg, bundle.store)
    cache.ensure(train_recs + dev_recs, jobs=jobs)

    index = build_window_index(train_recs, cfg.fs, cfg.window_samples, cfg.train_stride_samples,
                               cfg.windows.include_tail)
    sampler = WindowSampler(cfg, train_recs, index)
    summary = sampler.summary()
    write_json(run_dir / "sampling_summary.json", summary)
    log.info(f"Training windows available {summary['available']}, selected {summary['selected_epoch0']} "
             f"({summary['windows_per_epoch']}/epoch, event-sample fraction "
             f"{summary['positive_sample_fraction_epoch0']:.3f})")
    if summary["windows_per_epoch"] == 0:
        raise ValueError("sampler selected no training windows (no annotated events in the training split?)")

    device = resolve_device(cfg.experiment.device)
    patients = sorted({r.patient for r in train_recs}, key=natural_key)
    trainer = Trainer(cfg, run_dir, device, train_recs, dev_recs, cache, sampler, patients)
    if dry_run:
        log.info("Dry run: data, cache, sampler and model are ready; exiting before training.")
        return run_dir
    if resume is not None:
        trainer.resume(run_dir / "checkpoints" / "last.pt")
    result = trainer.fit()
    write_json(run_dir / "result.json", result)

    best = run_dir / "checkpoints" / "best.pt"
    for split in cfg.evaluation.splits:
        recs = bundle.recordings.get(split) or []
        if not recs:
            continue
        cache.ensure(recs, jobs=jobs)
        evaluate_checkpoint(cfg, best, recs, cache, run_dir / "eval" / f"{split}_best", label=f"{split} (best)",
                            device=device, run_sweep=True)
    return run_dir


# ---------------------------------------------------------------------- evaluate
def evaluate_checkpoint(cfg: Config, checkpoint: Path, recordings: List[Recording], source, out_dir: Path,
                        label: str, device: Optional[torch.device] = None, run_sweep: bool = False,
                        use_ema: bool = True) -> Dict[str, float]:
    device = device or resolve_device(cfg.experiment.device)
    model = load_eval_model(checkpoint, cfg, device, use_ema)
    task = Task(cfg)
    out_dir = Path(out_dir)
    probs_dir = out_dir / "probs" if cfg.evaluation.save_probs else None
    result = evaluate_recordings(model, recordings, source, cfg, task, device, probs_dir=probs_dir, label=label,
                                 log_each=True)
    context = {"checkpoint": str(checkpoint), "threshold": cfg.postprocess.threshold,
               "morph_kernel": cfg.postprocess.morph_kernel, "min_duration_sec": cfg.postprocess.min_duration_sec,
               "inference_hop_sec": cfg.inference.hop_sec or cfg.windows.length_sec,
               "note": "dev is also used for checkpoint selection: dev numbers are optimistic" if "dev" in label else ""}
    metrics = write_report(result, out_dir, context)
    log.info(f"[{label}] per-patient results:\n" + short_summary(result))
    if probs_dir is not None:
        write_manifest(out_dir / "manifest.json", [
            {"patient": r.patient, "record": r.record, "fs": r.fs, "n_samples": r.n_samples,
             "ref_events": r.ref_events, "mode": task.mode} for r in result.records])
        if run_sweep:
            sweep(out_dir, cfg, cfg.evaluation.sweep_thresholds)
    return metrics


def evaluate(checkpoint: Path, overrides: Sequence[str], splits: Sequence[str], patients: Sequence[str] = (),
             out: Optional[Path] = None, jobs: int = 1, run_sweep: bool = True, use_cache: bool = True,
             use_ema: bool = True) -> None:
    cfg = config_from_checkpoint(checkpoint, overrides)
    run_dir = Path(checkpoint).resolve().parent.parent
    setup_logging(run_dir / "evaluate.log")
    if patients:
        store = H5Store(cfg)
        recs = store.scan(list(patients)).recordings
        groups = {"patients": recs}
    else:
        bundle = load_recordings(cfg, splits)
        groups = {s: bundle.recordings[s] for s in splits}
    for name, recs in groups.items():
        if not recs:
            log.warning(f"split '{name}' is empty; skipped")
            continue
        source = SignalCache(cfg) if use_cache else InMemorySource(cfg)
        if use_cache:
            source.ensure(recs, jobs=jobs)
        out_dir = Path(out) / name if out else run_dir / "eval" / f"{name}_{Path(checkpoint).stem}"
        evaluate_checkpoint(cfg, Path(checkpoint), recs, source, out_dir, label=name, run_sweep=run_sweep,
                            use_ema=use_ema)
        log.info(f"Reports written to {out_dir}")


# ---------------------------------------------------------------------- predict
def predict(checkpoint: Path, inputs: Sequence[Path], out_dir: Path, overrides: Sequence[str] = (),
            save_probs: bool = False) -> None:
    """Annotate arbitrary recordings (no labels needed); writes one events TSV per file."""
    cfg = config_from_checkpoint(checkpoint, overrides)
    setup_logging(Path(out_dir) / "predict.log")
    device = resolve_device(cfg.experiment.device)
    model = load_eval_model(checkpoint, cfg, device)
    task = Task(cfg)
    store = H5Store(cfg)
    source = InMemorySource(cfg, store)
    files: List[Path] = []
    for p in inputs:
        p = Path(p)
        files.extend(sorted(p.rglob(cfg.data.file_pattern), key=lambda q: natural_key(q.name)) if p.is_dir() else [p])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        rec = store.recording_from_file(f)
        probs = predict_recording(model, source.open(rec), task, cfg.window_samples, cfg.inference_hop_samples,
                                  cfg.inference.batch_size, device, cfg.inference.combine, cfg.inference.tail,
                                  cfg.inference.amp)
        ev_prob = task.event_prob(probs)
        mask = postprocess(ev_prob, cfg.fs, cfg.postprocess)
        events = mask_to_events(mask, cfg.fs)
        lines = ["onset\tduration\teventType\tconfidence"]
        for a, b in events:
            conf = float(ev_prob[int(round(a * cfg.fs)):int(round(b * cfg.fs))].mean())
            lines.append(f"{a:.3f}\t{b - a:.3f}\tsz\t{conf:.4f}")
        target = out_dir / f"{rec.patient}_{rec.stem}_events.tsv"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if save_probs:
            np.save(out_dir / f"{rec.patient}_{rec.stem}_probs.npy", probs.astype(np.float16))
        log.info(f"{rec.key}: {len(events)} event(s) -> {target}")


# ---------------------------------------------------------------------- check-model
def check_model(cfg: Config, batch_size: int = 2, overfit_steps: int = 0) -> Dict[str, Any]:
    """Build the configured model, verify the I/O contract, time forward/backward and
    optionally overfit one synthetic batch (a quick convergence sanity check)."""
    from .losses import build_loss
    from .models import as_output
    from .task import align_logits

    device = resolve_device(cfg.experiment.device)
    seed_everything(cfg.experiment.seed)
    task = Task(cfg)
    model = build_model(cfg, task.num_outputs).to(device)
    print(describe_model(model))
    T = cfg.window_samples
    x = torch.randn(batch_size, cfg.n_channels, T, device=device)
    y = torch.zeros(batch_size, T, dtype=torch.long, device=device)
    y[:, T // 3: T // 2] = 1
    loss_fn = build_loss(cfg)
    model.train()
    t0 = time.time()
    out = as_output(model(x))
    logits = out.logits
    if logits.ndim != 3 or logits.shape[0] != batch_size or logits.shape[1] != task.num_outputs:
        raise ValueError(f"model output {tuple(logits.shape)} violates the contract (B, {task.num_outputs}, T')")
    loss, _ = loss_fn(align_logits(logits.float(), T), y)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    info = {"params": count_parameters(model), "output_shape": tuple(logits.shape), "input_samples": T,
            "fwd_bwd_sec": time.time() - t0, "initial_loss": float(loss.detach())}
    if device.type == "cuda":
        info["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    print(f"forward/backward OK: output {tuple(logits.shape)} for input {(batch_size, cfg.n_channels, T)}, "
          f"loss {float(loss):.4f}, {info['fwd_bwd_sec']:.2f}s")
    if overfit_steps:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for step in range(overfit_steps):
            opt.zero_grad()
            l, _ = loss_fn(align_logits(as_output(model(x)).logits.float(), T), y)
            l.backward()
            opt.step()
        info["overfit_final_loss"] = float(l)
        print(f"overfit {overfit_steps} steps on one batch: loss {info['initial_loss']:.4f} -> {float(l):.4f}")
    return info
