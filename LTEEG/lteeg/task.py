"""Task definition: how logits map to probabilities and targets.

* ``binary``: the model emits 1 logit channel per time step; sigmoid gives the event
  (seizure) probability. Any annotated class collapses to 1.
* ``multiclass``: K = len(class_names) channels (index 0 = background); softmax gives
  per-class probabilities and the event probability is ``1 - p(background)``.

Evaluation (post-processing, SzCORE scoring) always runs on the event probability,
so a multi-class model is scored on "any seizure" with no extra code.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config


class Task:
    def __init__(self, cfg: Config):
        self.mode = cfg.task.mode
        self.class_names = list(cfg.task.class_names)
        self.num_classes = len(self.class_names)
        self.num_outputs = 1 if self.mode == "binary" else self.num_classes

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, num_outputs, T) logits -> probabilities of the same shape (float32)."""
        logits = logits.float()
        return torch.sigmoid(logits) if self.mode == "binary" else torch.softmax(logits, dim=1)

    def event_prob(self, probs: np.ndarray) -> np.ndarray:
        """(num_outputs, N) probabilities -> (N,) probability of any event class."""
        return probs[0] if self.mode == "binary" else 1.0 - probs[0]

    def event_mask(self, labels: np.ndarray) -> np.ndarray:
        return labels > 0


def align_logits(logits: torch.Tensor, length: int) -> torch.Tensor:
    """Bring logits produced at a coarser temporal resolution (``model.output_stride`` > 1,
    e.g. patch/token-level models) to one value per input sample by linear interpolation."""
    if logits.shape[-1] == length:
        return logits
    return F.interpolate(logits, size=length, mode="linear", align_corners=False)
