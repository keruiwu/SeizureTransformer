"""Model contract shared by every network in the framework.

A model is any ``torch.nn.Module`` constructed as::

    Model(in_channels: int, in_samples: int, num_outputs: int, **params)

whose ``forward(x)`` takes ``x`` of shape ``(B, C, T)`` and returns either

* a tensor of logits ``(B, num_outputs, T')``, or
* a :class:`ModelOutput` with the main logits plus optional auxiliary logits (deep
  supervision / multi-stage refinement, each scored with the main loss) and
  ready-to-add auxiliary loss terms (e.g. a reconstruction loss for
  anomaly-detection-style objectives).

``T'`` may be smaller than ``T`` (token-level models); the framework interpolates
logits to ``T`` before the loss and during inference. ``in_samples`` is the training
window length; length-agnostic models may ignore it.

Optional class attributes:

* ``wants_meta = True`` -> ``forward(x, meta)`` receives a dict with ``patient``,
  ``rec`` and ``start`` tensors (patient-conditioned or adaptive models).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Union

import torch


@dataclass
class ModelOutput:
    logits: torch.Tensor
    aux_logits: List[torch.Tensor] = field(default_factory=list)
    losses: Dict[str, torch.Tensor] = field(default_factory=dict)


def as_output(out: Union[torch.Tensor, ModelOutput, dict]) -> ModelOutput:
    if isinstance(out, ModelOutput):
        return out
    if isinstance(out, torch.Tensor):
        return ModelOutput(out)
    if isinstance(out, dict):
        return ModelOutput(out["logits"], list(out.get("aux_logits", [])), dict(out.get("losses", {})))
    raise TypeError(f"Model returned unsupported type {type(out).__name__}")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
