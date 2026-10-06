"""Point-wise losses for dense (per-sample) event detection.

The loss is a weighted sum of terms configured as::

    loss:
      terms:
        - {name: bce, weight: 1.0}                    # optional: pos_weight, label_smoothing
        - {name: tmse, weight: 0.15, tau: 4.0}        # temporal smoothing (MS-TCN)

Every term receives logits ``(B, K, T)`` (already aligned to the sample rate), integer
targets ``(B, T)`` with ``-1`` = ignore, and returns a scalar. Ignored samples
(padding, optional onset/offset margins) never contribute.

Built-in terms
--------------
``bce``    binary cross-entropy with logits (binary task); ``pos_weight`` re-weights
           event samples, ``label_smoothing`` softens targets.
``ce``     cross-entropy (multiclass task), optional ``class_weights``.
``focal``  sigmoid/softmax focal loss (Lin et al. 2017), ``gamma``, ``alpha``.
``dice``   soft Dice on the event probability, computed over the whole batch so that
           windows without events are well-defined (V-Net / U-Time style).
``tmse``   truncated MSE between log-probabilities of neighbouring samples
           (Farha & Gall, MS-TCN, CVPR 2019): penalizes flicker, i.e. the
           over-segmentation that turns into false-alarm events after thresholding.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .config import Config
from .registry import LOSSES, call_with_checked_kwargs

Tensor = torch.Tensor


def _valid(target: Tensor) -> Tensor:
    return target >= 0


def _binary_logits(logits: Tensor, mode: str) -> Tensor:
    """(B, K, T) -> (B, T) logit of 'any event' for the binary formulation."""
    if mode == "binary":
        return logits[:, 0]
    # multiclass: log-odds of not-background
    logp = F.log_softmax(logits, dim=1)
    return torch.logsumexp(logp[:, 1:], dim=1) - logp[:, 0]


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    m = mask.to(values.dtype)
    return (values * m).sum() / m.sum().clamp_min(1.0)


@LOSSES.register("bce")
class BCELoss:
    def __init__(self, mode: str, pos_weight: Optional[float] = None, label_smoothing: float = 0.0):
        if mode != "binary":
            raise ValueError("'bce' is for task.mode=binary; use 'ce' or 'focal' for multiclass")
        self.pos_weight = pos_weight
        self.smooth = float(label_smoothing)

    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        z = logits[:, 0].float()
        valid = _valid(target)
        y = (target > 0).float()
        if self.smooth:
            y = y * (1 - self.smooth) + 0.5 * self.smooth
        pw = None if self.pos_weight is None else torch.tensor(self.pos_weight, device=z.device)
        loss = F.binary_cross_entropy_with_logits(z, y, reduction="none", pos_weight=pw)
        return _masked_mean(loss, valid)


@LOSSES.register("ce")
class CELoss:
    def __init__(self, mode: str, class_weights: Optional[Sequence[float]] = None, label_smoothing: float = 0.0):
        if mode != "multiclass":
            raise ValueError("'ce' is for task.mode=multiclass; use 'bce' for binary")
        self.w = class_weights
        self.smooth = float(label_smoothing)

    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        w = None if self.w is None else torch.tensor(self.w, device=logits.device, dtype=torch.float32)
        return F.cross_entropy(logits.float(), target, weight=w, ignore_index=-1, label_smoothing=self.smooth)


@LOSSES.register("focal")
class FocalLoss:
    def __init__(self, mode: str, gamma: float = 2.0, alpha: Optional[float] = 0.25):
        self.mode, self.gamma, self.alpha = mode, float(gamma), alpha

    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        valid = _valid(target)
        if self.mode == "binary":
            z = logits[:, 0].float()
            y = (target > 0).float()
            ce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
            p = torch.sigmoid(z)
            p_t = p * y + (1 - p) * (1 - y)
            loss = ce * (1 - p_t) ** self.gamma
            if self.alpha is not None:
                loss = loss * (self.alpha * y + (1 - self.alpha) * (1 - y))
            return _masked_mean(loss, valid)
        logp = F.log_softmax(logits.float(), dim=1)
        t = target.clamp_min(0)
        logp_t = logp.gather(1, t.unsqueeze(1)).squeeze(1)
        loss = -((1 - logp_t.exp()) ** self.gamma) * logp_t
        return _masked_mean(loss, valid)


@LOSSES.register("dice")
class DiceLoss:
    def __init__(self, mode: str, smooth: float = 1.0):
        self.mode, self.smooth = mode, float(smooth)

    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        valid = _valid(target).float()
        p = torch.sigmoid(_binary_logits(logits.float(), self.mode)) * valid
        y = (target > 0).float() * valid
        inter = (p * y).sum()
        return 1 - (2 * inter + self.smooth) / (p.sum() + y.sum() + self.smooth)


@LOSSES.register("tmse")
class TruncatedMSE:
    def __init__(self, mode: str, tau: float = 4.0, stride: int = 1):
        self.mode, self.tau, self.stride = mode, float(tau), int(stride)

    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        z = logits.float()
        if self.mode == "binary":
            z = torch.cat([F.logsigmoid(-z[:, :1]), F.logsigmoid(z[:, :1])], dim=1)
        else:
            z = F.log_softmax(z, dim=1)
        s = self.stride
        diff = (z[:, :, s:] - z[:, :, :-s].detach()) ** 2
        diff = diff.clamp(max=self.tau ** 2).mean(dim=1)
        valid = _valid(target)
        return _masked_mean(diff, valid[:, s:] & valid[:, :-s])


class CompositeLoss:
    def __init__(self, terms: List[Tuple[str, float, Any]]):
        self.terms = terms

    def __call__(self, logits: Tensor, target: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        total = logits.new_zeros((), dtype=torch.float32)
        parts: Dict[str, float] = {}
        for name, weight, fn in self.terms:
            value = fn(logits, target)
            total = total + weight * value
            parts[name] = float(value.detach())
        return total, parts


def build_loss(cfg: Config) -> CompositeLoss:
    terms = []
    seen: Dict[str, int] = {}
    for spec in cfg.loss.terms:
        spec = dict(spec)
        name = spec.pop("name")
        weight = float(spec.pop("weight", 1.0))
        fn = call_with_checked_kwargs(LOSSES.get(name), "loss", name, mode=cfg.task.mode, **spec)
        key = name if name not in seen else f"{name}{seen[name]}"
        seen[name] = seen.get(name, 0) + 1
        terms.append((key, weight, fn))
    if not terms:
        raise ValueError("loss.terms is empty")
    return CompositeLoss(terms)
