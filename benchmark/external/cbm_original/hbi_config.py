import os
import time
import math
import inspect
from dataclasses import dataclass, field
from typing import Optional, Union


def _default_fname() -> str:
    
    t = time.time()
    # Get the directory of the calling script (not cwd)
    frame = inspect.currentframe()
    cbm_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        # Walk up the stack to find the first frame outside the cbm package
        current = frame
        while current is not None:
            frame_file = current.f_globals.get('__file__')
            if frame_file:
                frame_dir = os.path.dirname(os.path.abspath(frame_file))
                # Found a frame outside the cbm package
                if not frame_dir.startswith(cbm_dir):
                    return os.path.join(frame_dir, f"cbm_hbi_{t:0.4f}.pkl")
            current = current.f_back
    finally:
        del frame
    # Fallback to current directory if we can't determine caller
    return f"cbm_hbi_{t:0.4f}.pkl"


def _valid_fname(arg: Optional[str]) -> bool:
    """
    valid_fname:
      - empty or None → valid
      - otherwise directory must exist and extension must be '.pkl'
    """
    if arg is None or arg == "":
        return True
    try:
        fdir, fname = os.path.split(arg)
        _, fext = os.path.splitext(fname)
        if fdir == "":
            fdir = "."
        return os.path.isdir(fdir) and fext == ".pkl"
    except Exception:
        return False


def _valid_flog(arg: Optional[Union[str, int]]) -> bool:

    if arg is None or arg == "":
        return True
    if isinstance(arg, int) and arg == -1:
        return True
    if isinstance(arg, int) and arg == 1:
        return True
    if isinstance(arg, str):
        fdir, _ = os.path.split(arg)
        if fdir == "":
            fdir = "."
        return os.path.isdir(fdir)
    return False


@dataclass
class HBIConfig:

    verbose: int = 1
    fname_prog: Optional[str] = field(default_factory=_default_fname)
    flog: Optional[Union[str, int]] = None
    save_prog: int = 0
    initialize: str = "all_r_1"
    maxiter: int = 50
    tolx: float = 0.01
    tolL: float = -math.log(0.5)

    def __post_init__(self):
        # -----------------------
        # verbose
        # -----------------------
        if not isinstance(self.verbose, int):
            raise ValueError("verbose must be an integer")

        # -----------------------
        # save_prog (logical)
        # -----------------------
        self.save_prog = int(bool(self.save_prog))
        if self.save_prog not in (0, 1):
            raise ValueError("save_prog must be 0 or 1")

        # -----------------------
        # initialize ∈ {'all_r_1','cluster_r'}
        # -----------------------
        if self.initialize not in ("all_r_1", "cluster_r"):
            raise ValueError("initialize must be 'all_r_1' or 'cluster_r'")

        # -----------------------
        # maxiter integer
        # -----------------------
        if not isinstance(self.maxiter, int):
            raise ValueError("maxiter must be an integer")

        # -----------------------
        # tolx, tolL scalar numerics
        # -----------------------
        if not isinstance(self.tolx, (int, float)):
            raise ValueError("tolx must be a scalar number")
        if not isinstance(self.tolL, (int, float)):
            raise ValueError("tolL must be a scalar number")

        # -----------------------
        # fname_prog validity
        # -----------------------
        if not _valid_fname(self.fname_prog):
            raise ValueError(f"Invalid fname_prog: {self.fname_prog}")

        # -----------------------
        # flog validity
        # -----------------------
        if not _valid_flog(self.flog):
            raise ValueError(f"Invalid flog: {self.flog}")

        # -----------------------
        if self.save_prog == 0:
            self.fname_prog = None