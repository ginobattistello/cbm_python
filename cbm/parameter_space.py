"""Free/fixed parameter bookkeeping.

A zero prior variance fixes a parameter exactly at its prior mean, following
the VBA-style convention:

    prior_variance[i] > 0  -> estimate theta[i]
    prior_variance[i] == 0 -> fix theta[i] = prior_mean[i]

Fixed parameters are removed from optimization and Laplace integration rather
than represented by an artificial infinite precision. Cognitive models still
receive the complete parameter vector.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np


@dataclass
class ParameterSpace:
    """Mapping between the model's full vector and the inferred free vector."""

    full_mean: np.ndarray
    covariance: np.ndarray
    free_mask: np.ndarray
    fixed_mask: np.ndarray
    free_covariance: np.ndarray
    free_precision: np.ndarray

    @classmethod
    def from_prior(cls, prior_mean, prior_variance) -> "ParameterSpace":
        mean = np.asarray(prior_mean, dtype=float).reshape(-1)
        d = mean.size

        variance = np.asarray(prior_variance, dtype=float)

        if variance.ndim == 0:
            value = float(variance)
            if value < 0:
                raise ValueError("prior_variance must be non-negative")
            covariance = value * np.eye(d)

        elif variance.ndim == 1:
            if variance.size != d:
                raise ValueError(
                    f"prior_variance has length {variance.size}; expected {d}."
                )
            if np.any(variance < 0):
                raise ValueError("prior_variance entries must be non-negative")
            covariance = np.diag(variance)

        elif variance.shape == (d, d):
            covariance = 0.5 * (variance + variance.T)
            diag = np.diag(covariance)

            if np.any(diag < 0):
                raise ValueError(
                    "prior covariance diagonal entries must be non-negative"
                )

        else:
            raise ValueError(
                "prior_variance must be a scalar, a length-d vector, or a "
                f"({d}, {d}) covariance matrix."
            )

        diag = np.diag(covariance)
        fixed_mask = diag == 0.0
        free_mask = ~fixed_mask

        # In a covariance matrix, zero variance implies zero covariance with
        # every other parameter. Enforce this explicitly for clear errors.
        for i in np.where(fixed_mask)[0]:
            row = covariance[i, :].copy()
            col = covariance[:, i].copy()
            row[i] = 0.0
            col[i] = 0.0
            if not (
                np.allclose(row, 0.0, atol=1e-12, rtol=0.0)
                and np.allclose(col, 0.0, atol=1e-12, rtol=0.0)
            ):
                raise ValueError(
                    f"parameter {i} has zero prior variance but non-zero "
                    "prior covariance. A fixed parameter must have a zero "
                    "row and column in the prior covariance."
                )

        free_covariance = covariance[np.ix_(free_mask, free_mask)]

        if free_covariance.size:
            try:
                np.linalg.cholesky(free_covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "The covariance of the free parameters must be positive "
                    "definite."
                ) from exc
            free_precision = np.linalg.inv(free_covariance)
        else:
            free_covariance = np.empty((0, 0), dtype=float)
            free_precision = np.empty((0, 0), dtype=float)

        return cls(
            full_mean=mean,
            covariance=covariance,
            free_mask=free_mask,
            fixed_mask=fixed_mask,
            free_covariance=free_covariance,
            free_precision=free_precision,
        )

    @classmethod
    def all_free(cls, prior_mean, prior_precision) -> "ParameterSpace":
        """Create an all-free space from an already defined precision."""
        mean = np.asarray(prior_mean, dtype=float).reshape(-1)
        precision = np.asarray(prior_precision, dtype=float)
        d = mean.size

        if precision.shape != (d, d):
            raise ValueError(
                f"prior_precision must have shape ({d}, {d}), "
                f"got {precision.shape}"
            )

        precision = 0.5 * (precision + precision.T)
        try:
            np.linalg.cholesky(precision)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "prior_precision must be positive definite"
            ) from exc

        covariance = np.linalg.inv(precision)

        return cls(
            full_mean=mean,
            covariance=covariance,
            free_mask=np.ones(d, dtype=bool),
            fixed_mask=np.zeros(d, dtype=bool),
            free_covariance=covariance,
            free_precision=precision,
        )

    @property
    def d_full(self) -> int:
        return int(self.full_mean.size)

    @property
    def d_free(self) -> int:
        return int(np.sum(self.free_mask))

    @property
    def free_indices(self) -> np.ndarray:
        return np.flatnonzero(self.free_mask)

    @property
    def fixed_indices(self) -> np.ndarray:
        return np.flatnonzero(self.fixed_mask)

    @property
    def free_mean(self) -> np.ndarray:
        return self.full_mean[self.free_mask]

    @property
    def fixed_values(self) -> np.ndarray:
        return self.full_mean[self.fixed_mask]

    @property
    def full_precision(self) -> np.ndarray:
        """Free-parameter precision embedded in full model coordinates."""
        P = np.zeros((self.d_full, self.d_full), dtype=float)
        if self.d_free:
            P[np.ix_(self.free_mask, self.free_mask)] = self.free_precision
        return P

    def reduce(self, theta_full) -> np.ndarray:
        theta_full = np.asarray(theta_full, dtype=float).reshape(-1)
        if theta_full.size != self.d_full:
            raise ValueError(
                f"full parameter vector has length {theta_full.size}; "
                f"expected {self.d_full}"
            )
        return theta_full[self.free_mask]

    def expand(self, theta_free) -> np.ndarray:
        theta_free = np.asarray(theta_free, dtype=float).reshape(-1)
        if theta_free.size != self.d_free:
            raise ValueError(
                f"free parameter vector has length {theta_free.size}; "
                f"expected {self.d_free}"
            )

        theta = self.full_mean.copy()
        theta[self.free_mask] = theta_free
        return theta

    def expand_free_vector(
        self,
        values_free,
        fixed_value: float = 0.0,
    ) -> np.ndarray:
        """Embed a free-space vector in full model coordinates."""
        values_free = np.asarray(values_free, dtype=float).reshape(-1)
        if values_free.size != self.d_free:
            raise ValueError(
                f"free vector has length {values_free.size}; "
                f"expected {self.d_free}"
            )

        out = np.full(self.d_full, fixed_value, dtype=float)
        out[self.free_mask] = values_free
        return out

    def expand_jax(self, theta_free):
        """JAX-compatible expansion used by the optional AD Hessian."""
        import jax.numpy as jnp

        theta = jnp.asarray(self.full_mean)
        indices = jnp.asarray(self.free_indices, dtype=jnp.int32)
        return theta.at[indices].set(theta_free)

    def reduce_config(self, config):
        """Return an optimizer Config containing only free dimensions.

        Bounds and user initializations are subset to the free coordinates.
        Fixed values are constants, so their range/hard bounds do not enter
        the optimizer.
        """
        if self.d_free == self.d_full:
            return config

        from .optimization import Config

        valid_names = {f.name for f in fields(Config)}
        values = {
            key: value
            for key, value in config.__dict__.items()
            if key in valid_names
        }
        values["d"] = self.d_free

        for name in ("range_bounds", "hard_bounds"):
            bounds = np.asarray(getattr(config, name), dtype=float)
            if bounds.shape == (2, self.d_full):
                values[name] = bounds[:, self.free_mask]

        inits = getattr(config, "inits", None)
        if inits is not None:
            arr = np.asarray(inits, dtype=float)
            if arr.ndim == 1:
                if arr.size == self.d_full:
                    arr = arr[self.free_mask]
                elif arr.size != self.d_free:
                    raise ValueError(
                        "config.inits must use either full-model or free "
                        "parameter dimensionality."
                    )
            elif arr.ndim == 2:
                if arr.shape[1] == self.d_full:
                    arr = arr[:, self.free_mask]
                elif arr.shape[1] != self.d_free:
                    raise ValueError(
                        "config.inits must use either full-model or free "
                        "parameter dimensionality."
                    )
            else:
                raise ValueError("config.inits must be one- or two-dimensional")
            values["inits"] = arr

        return Config(**values)
