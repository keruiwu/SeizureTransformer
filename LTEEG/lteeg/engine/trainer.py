"""Training loop with long-range validation.

One epoch = one pass over the windows chosen by the sampler. Validation is not done
on windows but by running the model over every complete dev recording and scoring
the stitched predictions exactly like the final evaluation, so the checkpoint is
selected on the quantity that is eventually reported.

Engineering features (all configurable): mixed precision (bf16/fp16 + GradScaler),
gradient accumulation and clipping, EMA weights, non-finite-loss guard, atomic
checkpoints with full resume (model/optimizer/scheduler/scaler/EMA/RNG), early
stopping, background prefetch, ``torch.compile``, per-epoch CSV/JSON logs.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..data.augment import Augmenter
from ..data.datasets import TrainWindowDataset
from ..data.windows import EpochSampler
from ..evaluation.report import short_summary
from ..evaluation.runner import EvalResult, evaluate_recordings
from ..inference.longrange import autocast_context
from ..losses import build_loss
from ..models import as_output, build_model, describe_model
from ..task import Task, align_logits
from ..utils import format_duration
from ..utils.logging import get_logger
from .optim import build_optimizer, build_scheduler
from .utils import (ModelEMA, ThreadPrefetcher, cuda_memory_gb, load_checkpoint, make_grad_scaler, rng_state,
                    save_checkpoint, set_rng_state, unwrap)

log = get_logger("train")


class NonFiniteLossError(RuntimeError):
    pass


class Trainer:
    def __init__(self, cfg: Config, run_dir: Path, device: torch.device, train_recs, dev_recs, source,
                 sampler, patients: List[str]):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.device = device
        self.train_recs, self.dev_recs = list(train_recs), list(dev_recs)
        self.source = source
        self.task = Task(cfg)
        tr = cfg.train

        dataset = TrainWindowDataset(cfg, self.train_recs, source, patients)
        self.epoch_sampler = EpochSampler(sampler)
        n_windows = len(self.epoch_sampler)
        loader = DataLoader(dataset, batch_size=tr.batch_size, sampler=self.epoch_sampler,
                            num_workers=tr.num_workers, pin_memory=tr.pin_memory and device.type == "cuda",
                            drop_last=n_windows > tr.batch_size,  # avoid a size-1 BatchNorm batch
                            persistent_workers=tr.num_workers > 0)
        self.loader = loader
        self.batches_per_epoch = len(loader)
        self.steps_per_epoch = max(1, math.ceil(self.batches_per_epoch / tr.accum_steps))

        model = build_model(cfg, self.task.num_outputs).to(device)
        log.info(describe_model(model))
        self.model = model
        self.optimizer = build_optimizer(cfg, model)
        self.scheduler = build_scheduler(cfg, self.optimizer, self.steps_per_epoch)
        self.scaler = make_grad_scaler(tr.amp == "fp16" and device.type == "cuda")
        self.ema = ModelEMA(model, tr.ema_decay) if tr.ema_decay else None
        if tr.compile:
            try:
                self.model = torch.compile(model)
            except Exception as e:  # pragma: no cover - platform dependent (e.g. no Triton on Windows)
                log.warning(f"torch.compile unavailable ({e}); continuing without it")
        self.loss_fn = build_loss(cfg)
        self.augment = Augmenter(cfg, tr.augment)
        self.wants_meta = getattr(unwrap(self.model), "wants_meta", False)

        self.epoch = 0
        self.global_step = 0
        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.bad_validations = 0
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ checkpointing
    def _state(self) -> Dict[str, Any]:
        return {
            "model": unwrap(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "ema": self.ema.state_dict() if self.ema else None,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "bad_validations": self.bad_validations,
            "history": self.history,
            "config": self.cfg.to_dict(),
            "rng": rng_state(),
        }

    def _weights_state(self) -> Dict[str, Any]:
        """Evaluation checkpoint: weights (+EMA), config and metadata, no optimizer state."""
        return {"model": unwrap(self.model).state_dict(), "ema": self.ema.state_dict() if self.ema else None,
                "epoch": self.epoch, "best_value": self.best_value,
                "selection_metric": self.cfg.train.selection_metric, "config": self.cfg.to_dict()}

    def resume(self, path: Path) -> None:
        state = load_checkpoint(path)
        unwrap(self.model).load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        if self.ema and state.get("ema"):
            self.ema.load_state_dict(state["ema"])
        self.epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.best_value, self.best_epoch = state["best_value"], state["best_epoch"]
        self.bad_validations = state.get("bad_validations", 0)
        self.history = state.get("history", [])
        set_rng_state(state["rng"])
        log.info(f"Resumed from {path} at epoch {self.epoch} (best {self.best_value} @ {self.best_epoch})")

    def eval_model(self) -> torch.nn.Module:
        return self.ema.module if self.ema else self.model

    # ------------------------------------------------------------------ training
    def _forward_loss(self, x: torch.Tensor, y: torch.Tensor, batch: Dict[str, Any]):
        with autocast_context(self.device, self.cfg.train.amp):
            if self.wants_meta:
                meta = {k: torch.as_tensor(batch[k]).to(self.device) for k in ("patient", "rec", "start")}
                out = as_output(self.model(x, meta))
            else:
                out = as_output(self.model(x))
        T = y.shape[-1]
        loss, parts = self.loss_fn(align_logits(out.logits.float(), T), y)
        for i, aux in enumerate(out.aux_logits):
            aux_loss, _ = self.loss_fn(align_logits(aux.float(), T), y)
            loss = loss + aux_loss
            parts[f"aux{i}"] = float(aux_loss.detach())
        for name, value in out.losses.items():
            loss = loss + value.float()
            parts[name] = float(value.detach())
        return loss, parts

    def train_epoch(self) -> Dict[str, float]:
        tr = self.cfg.train
        self.model.train()
        self.epoch_sampler.set_epoch(self.epoch)
        iterable = ThreadPrefetcher(self.loader, tr.prefetch_batches) \
            if (tr.num_workers == 0 and tr.prefetch_batches > 0) else self.loader
        sums: Dict[str, float] = {}
        n_batches, n_skipped, consecutive_bad = 0, 0, 0
        grad_norms: List[float] = []
        t0 = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(iterable):
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            if self.augment:
                x, y = self.augment(x, y)
            loss, parts = self._forward_loss(x, y, batch)
            if not torch.isfinite(loss):
                n_skipped += 1
                consecutive_bad += 1
                self.optimizer.zero_grad(set_to_none=True)
                log.warning(f"non-finite loss at epoch {self.epoch} batch {i}: {parts}; batch skipped")
                if consecutive_bad >= tr.max_nonfinite_steps:
                    raise NonFiniteLossError(f"{consecutive_bad} consecutive non-finite losses; "
                                             f"lower the learning rate, enable train.grad_clip or disable AMP")
                continue
            consecutive_bad = 0
            self.scaler.scale(loss / tr.accum_steps).backward()
            last_in_epoch = (i + 1) == self.batches_per_epoch
            if (i + 1) % tr.accum_steps == 0 or last_in_epoch:
                self.scaler.unscale_(self.optimizer)
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                    tr.grad_clip if tr.grad_clip else float("inf"))
                grad_norms.append(float(gn))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1
                if self.ema:
                    self.ema.update(self.model)
            n_batches += 1
            sums["loss"] = sums.get("loss", 0.0) + float(loss.detach())
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            if tr.log_every and (i + 1) % tr.log_every == 0:
                rate = (i + 1) * tr.batch_size / (time.time() - t0)
                log.info(f"epoch {self.epoch} [{i + 1}/{self.batches_per_epoch}] loss {sums['loss'] / n_batches:.4f} "
                         f"lr {self.optimizer.param_groups[0]['lr']:.2e} "
                         f"grad_norm {grad_norms[-1] if grad_norms else float('nan'):.3f} {rate:.0f} win/s")
        out = {k: v / max(1, n_batches) for k, v in sums.items()}
        out.update({"batches": n_batches, "skipped": n_skipped, "time_sec": time.time() - t0,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
                    "grad_norm_max": float(np.max(grad_norms)) if grad_norms else float("nan")})
        return out

    # ------------------------------------------------------------------ validation
    def validate(self) -> EvalResult:
        return evaluate_recordings(self.eval_model(), self.dev_recs, self.source, self.cfg, self.task,
                                   self.device, label=f"dev@{self.epoch}")

    def _is_better(self, value: float) -> bool:
        if value is None or not np.isfinite(value):
            return False
        if self.best_value is None:
            return True
        delta = self.cfg.train.min_delta
        return value > self.best_value + delta if self.cfg.train.selection_mode == "max" \
            else value < self.best_value - delta

    # ------------------------------------------------------------------ main loop
    def fit(self) -> Dict[str, Any]:
        tr = self.cfg.train
        log.info(f"Training {tr.epochs} epochs x {self.batches_per_epoch} batches "
                 f"({len(self.epoch_sampler)} windows/epoch, batch {tr.batch_size}, accum {tr.accum_steps}) "
                 f"on {self.device}")
        log_csv = self.run_dir / "train_log.csv"
        start_epoch = self.epoch + 1 if self.history else 1
        t_start = time.time()
        for epoch in range(start_epoch, tr.epochs + 1):
            self.epoch = epoch
            stats = self.train_epoch()
            row: Dict[str, Any] = {"epoch": epoch, **{f"train_{k}": v for k, v in stats.items()}}
            msg = (f"epoch {epoch}/{tr.epochs} train loss {stats.get('loss', float('nan')):.4f} "
                   f"({format_duration(stats['time_sec'])})")
            mem = cuda_memory_gb(self.device)
            if mem is not None:
                row["max_mem_gb"] = mem
            validated = bool(self.dev_recs) and (epoch % tr.val_every == 0 or epoch == tr.epochs)
            improved = False
            if validated:
                result = self.validate()
                metrics = result.metrics()
                if tr.selection_metric not in metrics:
                    raise KeyError(f"train.selection_metric '{tr.selection_metric}' not among validation "
                                   f"metrics: {sorted(metrics)}")
                value = metrics[tr.selection_metric]
                row.update({f"val_{k}": v for k, v in metrics.items()})
                improved = self._is_better(value)
                if improved:
                    self.best_value, self.best_epoch, self.bad_validations = value, epoch, 0
                else:
                    self.bad_validations += 1
                msg += (f" | dev {tr.selection_metric} {value:.4f} (best {self.best_value} @ {self.best_epoch})"
                        f" event F1 pooled {metrics['event_f1_pooled']:.3f} macro {metrics['event_f1_macro']:.3f}"
                        f" FA/day {metrics['event_fp_per_day_pooled']:.2f} AUPRC {metrics['auprc_pooled']:.3f}")
                log.info("per-patient dev results (epoch %d):\n%s", epoch, short_summary(result))
            self.history.append(row)
            _append_csv(log_csv, row)
            log.info(msg)
            if improved:
                save_checkpoint(self.ckpt_dir / "best.pt", self._weights_state())
                with open(self.run_dir / "best.json", "w", encoding="utf-8") as f:
                    json.dump({"epoch": epoch, "metric": tr.selection_metric, "value": self.best_value}, f, indent=2)
            save_checkpoint(self.ckpt_dir / "last.pt", self._state())
            if validated and tr.early_stopping and self.bad_validations >= tr.patience:
                log.info(f"Early stopping: no improvement in {tr.patience} validations")
                break
            elapsed = time.time() - t_start
            done = epoch - start_epoch + 1
            log.info(f"elapsed {format_duration(elapsed)}, eta {format_duration(elapsed / done * (tr.epochs - epoch))}")
        if not self.dev_recs:
            save_checkpoint(self.ckpt_dir / "best.pt", self._weights_state())
        return {"best_epoch": self.best_epoch, "best_value": self.best_value, "epochs_run": self.epoch}


def _append_csv(path: Path, row: Dict[str, Any]) -> None:
    """Append a row; if new columns appear (first validation), the file is rewritten."""
    rows: List[Dict[str, Any]] = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows.append({k: v for k, v in row.items()})
    columns: List[str] = []
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)
