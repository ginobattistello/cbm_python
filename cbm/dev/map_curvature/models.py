"""Minimal NumPy/JAX models for the MAP curvature experiment.

Three models:
1. Binary Rescorla-Wagner + logistic/softmax choice.
2. Three-choice Rescorla-Wagner + categorical softmax.
3. CES value function + Gaussian continuous observations.

The NumPy and JAX implementations are deliberately kept mathematically
parallel so that the AD implementation can be verified against NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None


EPS = 1e-12


@dataclass
class Dataset:
    model: str
    data: dict
    theta_true: np.ndarray
    bounds: list[tuple[float, float]]
    prior_mean: np.ndarray
    prior_precision: np.ndarray


# ---------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------

def _softmax_np(x):
    z = x - np.max(x)
    ez = np.exp(z)
    return ez / np.sum(ez)


def _softmax_jax(x):
    z = x - jnp.max(x)
    ez = jnp.exp(z)
    return ez / jnp.sum(ez)


def _log_normal_np(y, mu, sigma):
    return -0.5 * ((y - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2.0 * np.pi))


def _log_normal_jax(y, mu, sigma):
    return -0.5 * ((y - mu) / sigma) ** 2 - jnp.log(
        sigma * jnp.sqrt(2.0 * jnp.pi)
    )


# ---------------------------------------------------------------------
# Binary RW + softmax
# theta = [alpha, beta]
# ---------------------------------------------------------------------

def binary_rw_trials_np(theta, data):
    alpha, beta = theta
    choices = np.asarray(data["choices"], dtype=int)
    rewards = np.asarray(data["rewards"], dtype=float)

    q = np.zeros(2, dtype=float)
    ll = np.empty(len(choices), dtype=float)

    for t, a in enumerate(choices):
        p = _softmax_np(beta * q)
        ll[t] = np.log(np.clip(p[a], EPS, 1.0))

        pe = rewards[t] - q[a]
        q[a] = q[a] + alpha * pe

    return ll


def binary_rw_trials_jax(theta, data):
    choices = jnp.asarray(data["choices"], dtype=jnp.int32)
    rewards = jnp.asarray(data["rewards"], dtype=jnp.float32)
    alpha, beta = theta

    def step(q, inputs):
        a, r = inputs
        p = _softmax_jax(beta * q)
        ll = jnp.log(jnp.clip(p[a], EPS, 1.0))

        pe = r - q[a]
        q_new = q.at[a].add(alpha * pe)
        return q_new, ll

    import jax
    q0 = jnp.zeros(2)
    _, ll = jax.lax.scan(step, q0, (choices, rewards))
    return ll


# ---------------------------------------------------------------------
# Three-choice RW + categorical softmax
# theta = [alpha, beta]
# ---------------------------------------------------------------------

def categorical_rw_trials_np(theta, data):
    alpha, beta = theta
    choices = np.asarray(data["choices"], dtype=int)
    rewards = np.asarray(data["rewards"], dtype=float)

    q = np.zeros(3, dtype=float)
    ll = np.empty(len(choices), dtype=float)

    for t, a in enumerate(choices):
        p = _softmax_np(beta * q)
        ll[t] = np.log(np.clip(p[a], EPS, 1.0))

        pe = rewards[t] - q[a]
        q[a] = q[a] + alpha * pe

    return ll


def categorical_rw_trials_jax(theta, data):
    choices = jnp.asarray(data["choices"], dtype=jnp.int32)
    rewards = jnp.asarray(data["rewards"], dtype=jnp.float32)
    alpha, beta = theta

    def step(q, inputs):
        a, r = inputs
        p = _softmax_jax(beta * q)
        ll = jnp.log(jnp.clip(p[a], EPS, 1.0))

        pe = r - q[a]
        q_new = q.at[a].add(alpha * pe)
        return q_new, ll

    import jax
    q0 = jnp.zeros(3)
    _, ll = jax.lax.scan(step, q0, (choices, rewards))
    return ll


# ---------------------------------------------------------------------
# CES + Gaussian continuous output
# theta = [alpha, rho]
# V = [alpha*x1^rho + (1-alpha)*x2^rho]^(1/rho)
# sigma is fixed by the dataset.
# ---------------------------------------------------------------------

def ces_values_np(theta, data):
    alpha, rho = theta
    x1 = np.asarray(data["x1"], dtype=float)
    x2 = np.asarray(data["x2"], dtype=float)

    inner = alpha * x1**rho + (1.0 - alpha) * x2**rho
    return inner ** (1.0 / rho)


def ces_trials_np(theta, data):
    sigma = float(data["sigma"])
    y = np.asarray(data["y"], dtype=float)
    v = ces_values_np(theta, data)
    return _log_normal_np(y, v, sigma)


def ces_values_jax(theta, data):
    alpha, rho = theta
    x1 = jnp.asarray(data["x1"])
    x2 = jnp.asarray(data["x2"])

    inner = alpha * x1**rho + (1.0 - alpha) * x2**rho
    return inner ** (1.0 / rho)


def ces_trials_jax(theta, data):
    sigma = float(data["sigma"])
    y = jnp.asarray(data["y"])
    v = ces_values_jax(theta, data)
    return _log_normal_jax(y, v, sigma)


# ---------------------------------------------------------------------
# Unified model interface
# ---------------------------------------------------------------------

MODEL_SPECS = {
    "binary": {
        "theta_true": np.array([0.35, 3.0]),
        "bounds": [(0.02, 0.98), (0.1, 8.0)],
        "prior_mean": np.array([0.5, 2.0]),
        "prior_sd": np.array([0.25, 2.0]),
    },
    "categorical": {
        "theta_true": np.array([0.35, 3.0]),
        "bounds": [(0.02, 0.98), (0.1, 8.0)],
        "prior_mean": np.array([0.5, 2.0]),
        "prior_sd": np.array([0.25, 2.0]),
    },
    "ces": {
        "theta_true": np.array([0.60, 0.30]),
        "bounds": [(0.02, 0.98), (-0.8, 0.8)],
        "prior_mean": np.array([0.5, 0.0]),
        "prior_sd": np.array([0.25, 0.5]),
    },
}


def make_prior(spec):
    mean = np.asarray(spec["prior_mean"], dtype=float)
    precision = np.diag(1.0 / np.asarray(spec["prior_sd"], dtype=float) ** 2)
    return mean, precision


def generate_dataset(model: str, rng: np.random.Generator, n_trials: int = 250) -> Dataset:
    """Generate one synthetic dataset and its fixed Gaussian prior."""
    if model not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {model}")

    spec = MODEL_SPECS[model]
    theta_true = spec["theta_true"].copy()
    prior_mean, prior_precision = make_prior(spec)

    if model == "binary":
        alpha, beta = theta_true
        q = np.zeros(2)
        choices = np.zeros(n_trials, dtype=int)
        rewards = np.zeros(n_trials)

        # Simple stochastic reward schedule.
        reward_probs = np.array([0.75, 0.25])

        for t in range(n_trials):
            p = _softmax_np(beta * q)
            a = rng.choice(2, p=p)
            r = float(rng.random() < reward_probs[a])

            choices[t] = a
            rewards[t] = r
            q[a] += alpha * (r - q[a])

        data = {"choices": choices, "rewards": rewards}

    elif model == "categorical":
        alpha, beta = theta_true
        q = np.zeros(3)
        choices = np.zeros(n_trials, dtype=int)
        rewards = np.zeros(n_trials)

        reward_probs = np.array([0.75, 0.50, 0.25])

        for t in range(n_trials):
            p = _softmax_np(beta * q)
            a = rng.choice(3, p=p)
            r = float(rng.random() < reward_probs[a])

            choices[t] = a
            rewards[t] = r
            q[a] += alpha * (r - q[a])

        data = {"choices": choices, "rewards": rewards}

    elif model == "ces":
        alpha, rho = theta_true
        x1 = rng.uniform(0.5, 2.0, size=n_trials)
        x2 = rng.uniform(0.5, 2.0, size=n_trials)

        inner = alpha * x1**rho + (1.0 - alpha) * x2**rho
        values = inner ** (1.0 / rho)

        sigma = 0.10
        y = values + rng.normal(0.0, sigma, size=n_trials)

        data = {"x1": x1, "x2": x2, "y": y, "sigma": sigma}

    else:
        raise AssertionError("unreachable")

    return Dataset(
        model=model,
        data=data,
        theta_true=theta_true,
        bounds=spec["bounds"],
        prior_mean=prior_mean,
        prior_precision=prior_precision,
    )


def trial_loglik_np(model, theta, data):
    if model == "binary":
        return binary_rw_trials_np(theta, data)
    if model == "categorical":
        return categorical_rw_trials_np(theta, data)
    if model == "ces":
        return ces_trials_np(theta, data)
    raise ValueError(model)


def trial_loglik_jax(model, theta, data):
    if jnp is None:
        raise ImportError("JAX is required for autodiff experiments.")
    if model == "binary":
        return binary_rw_trials_jax(theta, data)
    if model == "categorical":
        return categorical_rw_trials_jax(theta, data)
    if model == "ces":
        return ces_trials_jax(theta, data)
    raise ValueError(model)
