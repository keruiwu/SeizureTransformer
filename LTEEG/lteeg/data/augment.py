"""Batch-level augmentations applied on the training device.

Configured as ``train.augment: [{name: ..., p: ..., ...}, ...]``. Each op receives
``x`` (B, C, T) and ``y`` (B, T) and returns both; label-preserving ops leave ``y``
untouched. Dataset-specific knowledge (left/right channel pairs) comes from the
config (``data.symmetric_pairs``), never from the op itself.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch

from ..config import Config
from ..registry import AUGMENTATIONS, RegistryError

Tensor = torch.Tensor


class AugContext:
    def __init__(self, cfg: Config):
        self.channels = [c.upper() for c in cfg.data.channels]
        self.fs = cfg.fs
        self.pairs = [(self.channels.index(a.upper()), self.channels.index(b.upper()))
                      for a, b in cfg.data.symmetric_pairs]


def _sample_mask(b: int, p: float, device: torch.device) -> Tensor:
    return torch.rand(b, device=device) < p


@AUGMENTATIONS.register("amplitude_scale")
def amplitude_scale(ctx: AugContext, low: float = 0.8, high: float = 1.2, per_channel: bool = False,
                    p: float = 1.0) -> Callable:
    def op(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        shape = (x.shape[0], x.shape[1] if per_channel else 1, 1)
        g = torch.empty(shape, device=x.device, dtype=x.dtype).uniform_(low, high)
        apply = _sample_mask(x.shape[0], p, x.device).view(-1, 1, 1)
        return torch.where(apply, x * g, x), y
    return op


@AUGMENTATIONS.register("sign_flip")
def sign_flip(ctx: AugContext, p: float = 0.5) -> Callable:
    def op(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        apply = _sample_mask(x.shape[0], p, x.device).view(-1, 1, 1)
        return torch.where(apply, -x, x), y
    return op


@AUGMENTATIONS.register("hemisphere_swap")
def hemisphere_swap(ctx: AugContext, p: float = 0.5) -> Callable:
    """Swap homologous left/right derivations (``data.symmetric_pairs``)."""
    if not ctx.pairs:
        raise ValueError("hemisphere_swap needs data.symmetric_pairs")
    perm = list(range(len(ctx.channels)))
    for a, b in ctx.pairs:
        perm[a], perm[b] = b, a

    def op(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        apply = _sample_mask(x.shape[0], p, x.device).view(-1, 1, 1)
        return torch.where(apply, x[:, perm], x), y
    return op


@AUGMENTATIONS.register("gaussian_noise")
def gaussian_noise(ctx: AugContext, std: float = 0.05, p: float = 0.5) -> Callable:
    def op(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        apply = _sample_mask(x.shape[0], p, x.device).view(-1, 1, 1)
        return torch.where(apply, x + std * torch.randn_like(x), x), y
    return op


@AUGMENTATIONS.register("channel_dropout")
def channel_dropout(ctx: AugContext, drop_prob: float = 0.1, p: float = 0.5) -> Callable:
    """Zero random channels (simulates detached electrodes / flat derivations)."""
    def op(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        apply = _sample_mask(x.shape[0], p, x.device).view(-1, 1, 1)
        keep = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) >= drop_prob).to(x.dtype)
        return torch.where(apply, x * keep, x), y
    return op


@AUGMENTATIONS.register("time_mask")
def time_mask(ctx: AugContext, max_sec: float = 1.0, n_masks: int = 1, p: float = 0.5) -> Callable:
    """Zero random time spans in all channels (labels unchanged)."""
    max_len = max(1, int(max_sec * ctx.fs))

    def op(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        b, _, t = x.shape
        pos = torch.arange(t, device=x.device).view(1, -1)
        keep = torch.ones(b, t, device=x.device, dtype=torch.bool)
        for _ in range(n_masks):
            length = torch.randint(1, max_len + 1, (b, 1), device=x.device)
            start = (torch.rand(b, 1, device=x.device) * (t - length + 1).clamp(min=1)).long()
            keep &= ~((pos >= start) & (pos < start + length))
        apply = _sample_mask(b, p, x.device).view(-1, 1)
        keep = keep | ~apply
        return x * keep.unsqueeze(1).to(x.dtype), y
    return op


class Augmenter:
    def __init__(self, cfg: Config, specs: Sequence[Dict[str, Any]]):
        ctx = AugContext(cfg)
        self.ops: List[Callable] = []
        self.names: List[str] = []
        for spec in specs:
            spec = dict(spec)
            name = spec.pop("name")
            self.ops.append(_build(AUGMENTATIONS.get(name), ctx, name, spec))
            self.names.append(name)

    def __bool__(self) -> bool:
        return bool(self.ops)

    def __call__(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        for op in self.ops:
            x, y = op(x, y)
        return x, y


def _build(factory: Callable, ctx: AugContext, name: str, params: Dict[str, Any]) -> Callable:
    import inspect

    accepted = set(inspect.signature(factory).parameters) - {"ctx"}
    unknown = set(params) - accepted
    if unknown:
        raise RegistryError(f"augmentation '{name}' got unexpected parameter(s) {sorted(unknown)}; accepted: {sorted(accepted)}")
    return factory(ctx, **params)
