"""Minimal name -> factory registries.

Components (models, losses, preprocessing operators, augmentations) register
themselves with a decorator. A name of the form ``"package.module:Attribute"`` is
imported on demand, so research code living outside this package can be plugged in
from the config without editing the framework.
"""

from __future__ import annotations

import difflib
import importlib
import inspect
from typing import Any, Callable, Dict, Iterable, Optional


class RegistryError(ValueError):
    """Unknown component name or invalid component parameters (a configuration error)."""


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Any] = {}

    def register(self, name: Optional[str] = None) -> Callable[[Any], Any]:
        def deco(obj: Any) -> Any:
            key = name or getattr(obj, "__name__")
            if key in self._items and self._items[key] is not obj:
                raise KeyError(f"{self.kind} '{key}' is already registered")
            self._items[key] = obj
            return obj

        return deco

    def get(self, name: str) -> Any:
        if name in self._items:
            return self._items[name]
        if ":" in name:  # dynamic import "pkg.module:Attr"
            module_name, attr = name.split(":", 1)
            module = importlib.import_module(module_name)
            try:
                return getattr(module, attr)
            except AttributeError as e:
                raise RegistryError(f"{module_name} has no attribute {attr!r}") from e
        close = difflib.get_close_matches(name, list(self._items), n=1)
        hint = f" (did you mean '{close[0]}'?)" if close else ""
        raise RegistryError(f"Unknown {self.kind} '{name}'{hint}. Available: {sorted(self._items)}; "
                       f"or use 'package.module:Attribute'.")

    def names(self) -> Iterable[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


def call_with_checked_kwargs(factory: Callable[..., Any], kind: str, name: str, **kwargs: Any) -> Any:
    """Call ``factory(**kwargs)`` but fail with a readable message on unexpected keywords
    (a misspelled hyper-parameter should never be silently ignored)."""
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(**kwargs)
    params = sig.parameters
    if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
        unknown = [k for k in kwargs if k not in params]
        if unknown:
            accepted = [k for k, p in params.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
            raise RegistryError(f"{kind} '{name}' got unexpected parameter(s) {unknown}; accepted: {accepted}")
    return factory(**kwargs)


MODELS = Registry("model")
LOSSES = Registry("loss")
PREPROCESSORS = Registry("preprocessing op")
AUGMENTATIONS = Registry("augmentation")
