"""Shared pytest fixtures for CBM."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def quadratic_matrix():
    """Well-conditioned positive-definite matrix with off-diagonal terms."""
    return np.array([
        [4.0, 1.0],
        [1.0, 2.0],
    ])


@pytest.fixture
def quadratic_objective(quadratic_matrix):
    A = quadratic_matrix

    def objective(theta):
        theta = np.asarray(theta, dtype=float)
        return 0.5 * theta @ A @ theta

    return objective


@pytest.fixture
def simple_continuous_data():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    theta = np.array([0.75, -0.4])
    y = theta[0] + theta[1] * x
    return [{"y": y, "X": {"x": x, "sigma": 1.0}}]


@pytest.fixture
def simple_continuous_model():
    def model(theta, data):
        y = np.asarray(data["y"], dtype=float)
        x = np.asarray(data["X"]["x"], dtype=float)
        sigma = float(data["X"].get("sigma", 1.0))
        mu = theta[0] + theta[1] * x
        return (
            -0.5 * ((y - mu) / sigma) ** 2
            - np.log(sigma * np.sqrt(2.0 * np.pi))
        )
    return model


@pytest.fixture
def simple_continuous_observation():
    def observation(theta, data):
        x = np.asarray(data["X"]["x"], dtype=float)
        return theta[0] + theta[1] * x
    return observation


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: slower scientific regression tests",
    )
    config.addinivalue_line(
        "markers",
        "jax: tests requiring JAX",
    )
