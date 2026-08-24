"""Minimal model definitions for the unified CBM API.

Every model accepts the same standardized subject data:

    data = {
        "y": observed outcomes,
        "X": model inputs,
    }

A model may return:
- a scalar summed log-likelihood, or
- a vector of per-trial log-likelihoods.

Returning a vector automatically enables the GN polish.
"""

import numpy as np


def _softmax(x):
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


# ---------------------------------------------------------------------
# Binary Rescorla-Wagner + softmax
# ---------------------------------------------------------------------

def binary_observation(theta, data):
    """Return P(choice=1) on every trial."""
    alpha, beta = theta
    y = np.asarray(data["y"], dtype=int)
    rewards = np.asarray(data["X"]["reward"], dtype=float)

    q = np.zeros(2)
    p1 = np.zeros(len(y))

    for t, (choice, reward) in enumerate(zip(y, rewards)):
        p = _softmax(beta * q)
        p1[t] = p[1]
        q[choice] += alpha * (reward - q[choice])

    return p1


def binary_model(theta, data):
    """Per-trial binary log-likelihood; GN is therefore available."""
    y = np.asarray(data["y"], dtype=int)
    p1 = np.clip(
        binary_observation(theta, data),
        1e-12,
        1.0 - 1e-12,
    )

    return (
        y * np.log(p1)
        + (1 - y) * np.log(1.0 - p1)
    )


# ---------------------------------------------------------------------
# Categorical Rescorla-Wagner + softmax
# ---------------------------------------------------------------------

def categorical_observation(theta, data):
    """Return T x K choice-probability matrix."""
    alpha, beta = theta
    y = np.asarray(data["y"], dtype=int)
    rewards = np.asarray(data["X"]["reward"], dtype=float)
    n_options = int(data["X"]["n_options"])

    q = np.zeros(n_options)
    probs = np.zeros((len(y), n_options))

    for t, (choice, reward) in enumerate(zip(y, rewards)):
        p = _softmax(beta * q)
        probs[t] = p
        q[choice] += alpha * (reward - q[choice])

    return probs


def categorical_model(theta, data):
    """Per-trial categorical log-likelihood."""
    y = np.asarray(data["y"], dtype=int)
    probs = np.clip(
        categorical_observation(theta, data),
        1e-12,
        1.0,
    )
    return np.log(probs[np.arange(len(y)), y])


# ---------------------------------------------------------------------
# Continuous Gaussian linear model
# ---------------------------------------------------------------------

def continuous_observation(theta, data):
    """Return the predicted conditional mean."""
    intercept, slope = theta
    x = np.asarray(data["X"]["x"], dtype=float)
    return intercept + slope * x


def continuous_model(theta, data):
    """Per-observation Gaussian log-likelihood."""
    y = np.asarray(data["y"], dtype=float)
    sigma = float(data["X"]["sigma"])
    mu = continuous_observation(theta, data)

    return (
        -0.5 * ((y - mu) / sigma) ** 2
        - np.log(sigma * np.sqrt(2.0 * np.pi))
    )


# ---------------------------------------------------------------------
# Scalar example
# ---------------------------------------------------------------------

def continuous_model_scalar(theta, data):
    """Same model, but summed manually: GN is unavailable."""
    return float(np.sum(continuous_model(theta, data)))
