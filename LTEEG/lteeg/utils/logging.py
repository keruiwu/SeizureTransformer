"""Logging setup: concise console output plus a full log file inside the run directory.

Messages are kept ASCII so they render on any Windows console code page.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

_FMT = "%(asctime)s %(levelname).1s %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(log_file: Optional[Union[str, Path]] = None, level: int = logging.INFO) -> logging.Logger:
    root = logging.getLogger("lteeg")
    root.setLevel(logging.DEBUG)
    root.propagate = False
    if not any(getattr(h, "_lteeg_console", False) for h in root.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        h.setLevel(level)
        h._lteeg_console = True  # type: ignore[attr-defined]
        root.addHandler(h)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        for h in list(root.handlers):
            target = getattr(h, "_lteeg_file", None)
            if target == str(log_file):
                return root
            if target is not None:  # one run log at a time (several runs in one process)
                root.removeHandler(h)
                h.close()
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
        fh.setLevel(logging.DEBUG)
        fh._lteeg_file = str(log_file)  # type: ignore[attr-defined]
        root.addHandler(fh)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"lteeg.{name}")
