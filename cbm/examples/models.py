"""Minimal model definitions for the unified CBM API.

Dynamic RL models expose an optional ``evolution(theta, data)`` function.
It returns deterministic trialwise latent variables. CBM evaluates this
function once at the final MAP for latent tracking.

The likelihood API itself is unchanged:
- scalar return -> L-BFGS-B only;
- per-trial vector -> L-BFGS-B + automatic GN polish.
"""

import numpy as np


def _softmax(x):
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


# ---------------------------------------------------------------------
# Binary Rescorla-Wagner + softmax
# ---------------------------------------------------------------------

def binary_evolution(theta, data):
    """Trialwise RW latent states evaluated before each observed choice."""
    alpha, _ = theta

    y = np.asarray(data["y"], dtype=int)
    rewards = np.asarray(data["X"]["reward"], dtype=float)

    q = np.zeros(2)
    q_history = np.zeros((len(y), 2))
    prediction_error = np.zeros(len(y))

    for t, (choice, reward) in enumerate(zip(y, rewards)):
        q_history[t] = q

        pe = reward - q[choice]
        prediction_error[t] = pe

        q[choice] += alpha * pe

    return {
        "Q": q_history,
        "prediction_error": prediction_error,
    }


def binary_observation(theta, data, latent=None):
    """Return P(choice=1) on every trial."""
    _, beta = theta

    if latent is None:
        latent = binary_evolution(theta, data)

    q = np.asarray(latent["Q"], dtype=float)
    p1 = np.zeros(q.shape[0])

    for t in range(q.shape[0]):
        p1[t] = _softmax(beta * q[t])[1]

    return p1


def binary_model(theta, data):
    """Per-trial binary log-likelihood."""
    y = np.asarray(data["y"], dtype=int)

    latent = binary_evolution(theta, data)
    p1 = np.clip(
        binary_observation(theta, data, latent=latent),
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

def categorical_evolution(theta, data):
    """Trialwise categorical RW values and prediction errors."""
    alpha, _ = theta

    y = np.asarray(data["y"], dtype=int)
    rewards = np.asarray(data["X"]["reward"], dtype=float)
    n_options = int(data["X"]["n_options"])

    q = np.zeros(n_options)
    q_history = np.zeros((len(y), n_options))
    prediction_error = np.zeros(len(y))

    for t, (choice, reward) in enumerate(zip(y, rewards)):
        q_history[t] = q

        pe = reward - q[choice]
        prediction_error[t] = pe

        q[choice] += alpha * pe

    return {
        "Q": q_history,
        "prediction_error": prediction_error,
    }


def categorical_observation(theta, data, latent=None):
    """Return T x K choice-probability matrix."""
    _, beta = theta

    if latent is None:
        latent = categorical_evolution(theta, data)

    q = np.asarray(latent["Q"], dtype=float)
    probs = np.zeros_like(q)

    for t in range(q.shape[0]):
        probs[t] = _softmax(beta * q[t])

    return probs


def categorical_model(theta, data):
    """Per-trial categorical log-likelihood."""
    y = np.asarray(data["y"], dtype=int)

    latent = categorical_evolution(theta, data)
    probs = np.clip(
        categorical_observation(theta, data, latent=latent),
        1e-12,
        1.0,
    )

    return np.log(
        probs[np.arange(len(y)), y]
    )


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


def continuous_model_scalar(theta, data):
    """Same continuous model summed manually: GN is unavailable."""
    return float(np.sum(continuous_model(theta, data)))
