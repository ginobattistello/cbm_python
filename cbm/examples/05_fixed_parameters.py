"""Fix a model parameter by setting its prior variance to zero."""

import numpy as np

from cbm.individual_fit import individual_fit
from cbm.optimization import Config


def model(theta, data):
    """Simple two-parameter Gaussian log-likelihood."""
    mu, bias = theta
    prediction = mu + bias
    sigma = 0.5

    residual = np.asarray(data) - prediction
    return float(
        np.sum(
            -0.5 * (residual / sigma) ** 2
            - np.log(sigma * np.sqrt(2.0 * np.pi))
        )
    )


data = [
    np.array([2.8, 3.1, 3.0, 2.9]),
    np.array([3.2, 2.9, 3.0, 3.1]),
]

# mu is estimated.
# bias is fixed exactly at 2.0 because its prior variance is zero.
fit = individual_fit(
    data=data,
    model=model,
    prior_mean=np.array([1.0, 2.0]),
    prior_variance=np.array([4.0, 0.0]),
    config=Config(
        d=2,
        verbose=True,
        display=False,
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

# The second column is exactly 2.0 for every subject and its SE is zero.
