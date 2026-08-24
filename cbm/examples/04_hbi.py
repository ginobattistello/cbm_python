"""Minimal hierarchical Bayesian inference example."""

from pathlib import Path

import numpy as np

from cbm.hbi import hbi_main
from cbm.hbi_config import HBIConfig
from cbm.individual_fit import individual_fit
from cbm.optimization import Config

from models import binary_model
from simulate import binary_subject


rng = np.random.default_rng(11)
data = [binary_subject(rng, theta=(0.35, 3.0)) for _ in range(12)]

output_dir = Path("cbm/examples/output")
output_dir.mkdir(parents=True, exist_ok=True)

config = Config(
    d=2,
    range_bounds=np.array([[0.02, 0.10], [0.98, 8.00]]),
    hard_bounds=np.array([[0.001, 0.01], [0.999, 20.0]]),
    num_init=5,
    verbose=False,
    display=False,
)

map_free = output_dir / "hbi_map_free.pkl"
map_fixed = output_dir / "hbi_map_fixed_alpha.pkl"

individual_fit(
    data=data,
    model=binary_model,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([1.0, 16.0]),
    fname=str(map_free),
    config=config,
)

individual_fit(
    data=data,
    model=binary_model,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([0.0, 16.0]),
    fname=str(map_fixed),
    config=config,
)

hbi = hbi_main(
    data=data,
    models=[binary_model, binary_model],
    fcbm_maps=[str(map_free), str(map_fixed)],
    config=HBIConfig(verbose=True, maxiter=20),
)

print("\nmodel frequency:")
print(hbi.output.model_frequency)

print("\nexceedance probability:")
print(hbi.output.exceedance_prob)

print("\nprotected exceedance probability:")
print(hbi.output.protected_exceedance_prob)

print("\ngroup means:")
for k, mean in enumerate(hbi.output.group_mean, start=1):
    print(f"model {k}: {mean}")
