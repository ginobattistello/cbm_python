from __future__ import annotations
import itertools
import numpy as np

DEFAULT_GRIDS = {
    "binary": {
        "param1_name": "alpha",
        "param2_name": "beta",
        "param1_values": np.array([0.05, 0.25, 0.50, 0.75, 0.95]),
        "param2_values": np.array([0.25, 0.50, 1.0, 3.0, 8.0]),
    },
    "categorical": {
        "param1_name": "alpha",
        "param2_name": "beta",
        "param1_values": np.array([0.05, 0.25, 0.50, 0.75, 0.95]),
        "param2_values": np.array([0.25, 0.50, 1.0, 3.0, 8.0]),
    },
    "ces": {
        "param1_name": "alpha",
        "param2_name": "rho",
        "param1_values": np.array([0.05, 0.25, 0.50, 0.75, 0.95]),
        "param2_values": np.array([-0.70, -0.30, -0.10, 0.30, 0.70]),
    },
}

def iter_grid(model):
    spec = DEFAULT_GRIDS[model]
    for p1, p2 in itertools.product(spec["param1_values"], spec["param2_values"]):
        yield {
            "param1_name": spec["param1_name"],
            "param2_name": spec["param2_name"],
            "param1": float(p1),
            "param2": float(p2),
            "theta_true": np.array([p1, p2], dtype=float),
        }
