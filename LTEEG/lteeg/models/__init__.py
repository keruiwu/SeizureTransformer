"""Model registry. Importing this package registers the built-in models.

Add a model by decorating its class with ``@MODELS.register("name")`` in a module
that gets imported (e.g. add an import below), or reference it from the config as
``model.name: "my_package.my_module:MyModel"`` without touching the framework.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..config import Config
from ..registry import MODELS, call_with_checked_kwargs
from ..utils.logging import get_logger
from .base import ModelOutput, as_output, count_parameters  # noqa: F401
from . import seizure_transformer, tcn  # noqa: F401  (registration side effect)

log = get_logger("models")


def build_model(cfg: Config, num_outputs: int) -> torch.nn.Module:
    factory = MODELS.get(cfg.model.name)
    model = call_with_checked_kwargs(factory, "model", cfg.model.name, in_channels=cfg.n_channels,
                                     in_samples=cfg.window_samples, num_outputs=num_outputs,
                                     **dict(cfg.model.params))
    if cfg.model.init_checkpoint:
        load_weights(model, cfg.resolve_path(cfg.model.init_checkpoint))
    return model


def load_weights(model: torch.nn.Module, path: Path, strict: bool = True) -> None:
    """Load weights from an LTEEG checkpoint (``{'model': state_dict, ...}``) or a bare
    state dict (e.g. one saved by the original SeizureTransformer repository)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if isinstance(model, seizure_transformer.SeizureTransformer):
        state = seizure_transformer.load_original_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    log.info(f"Loaded weights from {path}"
             + (f" (missing={list(missing)}, unexpected={list(unexpected)})" if (missing or unexpected) else ""))


def describe_model(model: torch.nn.Module) -> str:
    return f"{type(model).__name__}: {count_parameters(model) / 1e6:.2f}M trainable parameters"
