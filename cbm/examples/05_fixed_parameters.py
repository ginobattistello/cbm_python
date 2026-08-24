"""Fix a model parameter by setting its prior variance to zero."""

import numpy as np

from cbm.individual_fit import individual_fit
from cbm.optimization import Config


def observation(theta, data):
    """Predicted continuous outcome."""
    mu, bias = theta
    return np.full(len(data["y"]), mu + bias, dtype=float)


def model(theta, data):
    """Per-observation Gaussian log-likelihood."""
    sigma = 0.5
    y = np.asarray(data["y"], dtype=float)
    prediction = observation(theta, data)

    return (
        -0.5 * ((y - prediction) / sigma) ** 2
        - np.log(sigma * np.sqrt(2.0 * np.pi))
    )


data = [
    {"y": np.array([2.8, 3.1, 3.0, 2.9]), "X": {}},
    {"y": np.array([3.2, 2.9, 3.0, 3.1]), "X": {}},
]

fit = individual_fit(
    data=data,
    model=model,
    observation=observation,
    prior_mean=np.array([1.0, 2.0]),
    prior_variance=np.array([4.0, 0.0]),
    config=Config(
        d=2,
        verbose=True,
        display=True,
    ),
)

print("\nFull parameter vectors:")
print(fit.output.parameters)

print("\nFree mask:")
print(fit.input.free_mask)

print("\nFixed mask:")
print(fit.input.fixed_mask)

print("\nPosterior standard errors:")
print(fit.se)
