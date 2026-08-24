"""Configuration for hierarchical Bayesian inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union
import inspect
import math
import os
import time


def _default_fname() -> str:
    """Create a progress filename near the user script when possible."""
    stamp = time.time()
    cbm_dir = os.path.dirname(os.path.abspath(__file__))
    frame = inspect.currentframe()

    try:
        current = frame
        while current is not None:
            frame_file = current.f_globals.get("__file__")
            if frame_file:
                frame_dir = os.path.dirname(os.path.abspath(frame_file))
                if not frame_dir.startswith(cbm_dir):
                    return os.path.join(
                        frame_dir,
                        f"cbm_hbi_{stamp:0.4f}.pkl",
                    )
            current = current.f_back
    finally:
        del frame

    return f"cbm_hbi_{stamp:0.4f}.pkl"


def _valid_fname(value: Optional[str]) -> bool:
    if value is None or value == "":
        return True

    directory, filename = os.path.split(value)
    directory = directory or "."
    _, extension = os.path.splitext(filename)

    return os.path.isdir(directory) and extension == ".pkl"


def _valid_flog(value: Optional[Union[str, int]]) -> bool:
    """Validate log destination.

    Accepted values:
    - None / "": no explicit log file
    - -1: disable file logging
    - 1: retained for compatibility
    - string path whose parent directory exists
    """
    if value is None or value == "":
        return True
    if isinstance(value, int) and value in (-1, 1):
        return True
    if isinstance(value, str):
        directory, _ = os.path.split(value)
        return os.path.isdir(directory or ".")
    return False


@dataclass
class HBIConfig:
    """Options controlling the HBI iteration, not the MAP optimizer."""

    verbose: int = 1
    fname_prog: Optional[str] = field(default_factory=_default_fname)
    flog: Optional[Union[str, int]] = None
    save_prog: int = 0
    initialize: str = "all_r_1"
    maxiter: int = 50
    tolx: float = 0.01
    tolL: float = -math.log(0.5)

    def __post_init__(self):
        if not isinstance(self.verbose, int):
            raise ValueError("verbose must be an integer")

        self.save_prog = int(bool(self.save_prog))

        if self.initialize not in ("all_r_1", "cluster_r"):
            raise ValueError(
                "initialize must be 'all_r_1' or 'cluster_r'"
            )

        if not isinstance(self.maxiter, int) or self.maxiter < 1:
            raise ValueError("maxiter must be a positive integer")

        if not isinstance(self.tolx, (int, float)) or self.tolx < 0:
            raise ValueError("tolx must be a non-negative scalar")

        if not isinstance(self.tolL, (int, float)):
            raise ValueError("tolL must be a scalar number")

        if not _valid_fname(self.fname_prog):
            raise ValueError(
                f"Invalid fname_prog: {self.fname_prog}"
            )

        if not _valid_flog(self.flog):
            raise ValueError(f"Invalid flog: {self.flog}")

        if self.save_prog == 0:
            self.fname_prog = None
