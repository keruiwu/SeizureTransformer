"""Optimizers and step-wise learning-rate schedules."""

from __future__ import annotations

import math
from typing import List

import torch

from ..config import Config


def param_groups(model: torch.nn.Module, weight_decay: float, exclude_norm_bias: bool) -> List[dict]:
    if not exclude_norm_bias or weight_decay == 0:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": weight_decay}]
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(cfg: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    o = cfg.optimizer
    groups = param_groups(model, o.weight_decay, o.exclude_norm_bias_from_decay)
    betas = tuple(o.betas)
    if o.name == "adamw":
        return torch.optim.AdamW(groups, lr=o.lr, betas=betas, eps=o.eps)
    if o.name == "adam":
        return torch.optim.Adam(groups, lr=o.lr, betas=betas, eps=o.eps)
    if o.name == "radam":
        kwargs = {"decoupled_weight_decay": True} if o.decoupled_weight_decay else {}
        return torch.optim.RAdam(groups, lr=o.lr, betas=betas, eps=o.eps, **kwargs)
    if o.name == "sgd":
        return torch.optim.SGD(groups, lr=o.lr, momentum=o.momentum, nesterov=o.momentum > 0)
    raise ValueError(f"unknown optimizer {o.name}")


def build_scheduler(cfg: Config, optimizer: torch.optim.Optimizer, steps_per_epoch: int):
    """LambdaLR stepped once per optimizer step: optional linear warm-up, then
    constant (``none``), cosine decay to ``min_lr_ratio``, or step decay."""
    s = cfg.scheduler
    total = max(1, cfg.train.epochs * steps_per_epoch)
    warmup = int(round(s.warmup_epochs * steps_per_epoch))

    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if s.name == "cosine":
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            return s.min_lr_ratio + (1 - s.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        if s.name == "step":
            return s.gamma ** ((step // max(1, steps_per_epoch)) // max(1, s.step_epochs))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
