"""Training utilities: EMA, checkpoints, prefetching, device/environment helpers."""

from __future__ import annotations

import copy
import os
import platform
import queue
import random
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import numpy as np
import torch


# ------------------------------------------------------------------------- device
def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {name!r} requested but CUDA is not available")
    return dev


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)  # torch.compile wrapper


# ------------------------------------------------------------------------- EMA
class ModelEMA:
    """Exponential moving average of weights (buffers are copied). The effective decay
    ramps up as ``min(decay, (1 + n) / (10 + n))`` so early averages are not dominated
    by the random initialization."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.module = copy.deepcopy(unwrap(model)).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        src = unwrap(model)
        for e, p in zip(self.module.parameters(), src.parameters()):
            e.lerp_(p.detach(), 1.0 - d)
        for e, b in zip(self.module.buffers(), src.buffers()):
            e.copy_(b)

    def state_dict(self) -> Dict[str, Any]:
        return {"module": self.module.state_dict(), "updates": self.updates, "decay": self.decay}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.module.load_state_dict(state["module"])
        self.updates = state["updates"]


# ------------------------------------------------------------------------- checkpoints
def save_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    """Atomic save: a crash mid-write never leaves a truncated checkpoint behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def rng_state() -> Dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ------------------------------------------------------------------------- prefetch
class ThreadPrefetcher:
    """Fetch batches from ``iterable`` in a background thread.

    With ``num_workers=0`` (the safe default on Windows) a DataLoader reads and
    collates in the training thread; this overlaps that work with GPU compute.
    Memory-map reads and tensor collation release the GIL, so a thread suffices."""

    _END = object()

    def __init__(self, iterable: Iterable, depth: int = 2):
        self.iterable = iterable
        self.depth = max(1, depth)

    def __len__(self) -> int:
        return len(self.iterable)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator:
        q: "queue.Queue" = queue.Queue(maxsize=self.depth)
        stop = threading.Event()

        def worker() -> None:
            try:
                for item in self.iterable:
                    while not stop.is_set():
                        try:
                            q.put(item, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                    if stop.is_set():
                        return
                q.put(self._END)
            except BaseException as e:  # propagate to the consumer
                q.put(e)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        try:
            while True:
                item = q.get()
                if item is self._END:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()


# ------------------------------------------------------------------------- environment
def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "argv": sys.argv,
    }
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    try:
        import scipy

        info["scipy"] = scipy.__version__
    except ImportError:
        pass
    try:
        here = Path(__file__).resolve().parent
        info["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                                            text=True, timeout=5).stdout.strip() or None
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True,
                               text=True, timeout=5).stdout.strip()
        info["git_dirty"] = bool(dirty)
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


def cuda_memory_gb(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024 ** 3
