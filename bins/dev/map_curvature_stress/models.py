from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
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

def _softmax_np(x):
    z = x - np.max(x)
    ez = np.exp(z)
    return ez / np.sum(ez)

def _softmax_jax(x):
    z = x - jnp.max(x)
    ez = jnp.exp(z)
    return ez / jnp.sum(ez)

def binary_trials_np(theta, data):
    alpha, beta = theta
    choices = np.asarray(data["choices"], int)
    rewards = np.asarray(data["rewards"], float)
    q = np.zeros(2)
    ll = np.empty(len(choices))
    for t, a in enumerate(choices):
        p = _softmax_np(beta * q)
        ll[t] = np.log(np.clip(p[a], EPS, 1.0))
        q[a] += alpha * (rewards[t] - q[a])
    return ll

def binary_trials_jax(theta, data):
    alpha, beta = theta
    choices = jnp.asarray(data["choices"], dtype=jnp.int32)
    rewards = jnp.asarray(data["rewards"])
    def step(q, xs):
        a, r = xs
        p = _softmax_jax(beta * q)
        ll = jnp.log(jnp.clip(p[a], EPS, 1.0))
        q = q.at[a].add(alpha * (r - q[a]))
        return q, ll
    _, ll = jax.lax.scan(step, jnp.zeros(2), (choices, rewards))
    return ll

def categorical_trials_np(theta, data):
    alpha, beta = theta
    choices = np.asarray(data["choices"], int)
    rewards = np.asarray(data["rewards"], float)
    q = np.zeros(3)
    ll = np.empty(len(choices))
    for t, a in enumerate(choices):
        p = _softmax_np(beta * q)
        ll[t] = np.log(np.clip(p[a], EPS, 1.0))
        q[a] += alpha * (rewards[t] - q[a])
    return ll

def categorical_trials_jax(theta, data):
    alpha, beta = theta
    choices = jnp.asarray(data["choices"], dtype=jnp.int32)
    rewards = jnp.asarray(data["rewards"])
    def step(q, xs):
        a, r = xs
        p = _softmax_jax(beta * q)
        ll = jnp.log(jnp.clip(p[a], EPS, 1.0))
        q = q.at[a].add(alpha * (r - q[a]))
        return q, ll
    _, ll = jax.lax.scan(step, jnp.zeros(3), (choices, rewards))
    return ll

def ces_values_np(theta, data):
    alpha, rho = theta
    x1 = np.asarray(data["x1"], float)
    x2 = np.asarray(data["x2"], float)
    inner = alpha * x1**rho + (1.0-alpha) * x2**rho
    return inner ** (1.0/rho)

def ces_values_jax(theta, data):
    alpha, rho = theta
    x1 = jnp.asarray(data["x1"])
    x2 = jnp.asarray(data["x2"])
    inner = alpha * x1**rho + (1.0-alpha) * x2**rho
    return inner ** (1.0/rho)

def ces_trials_np(theta, data):
    y = np.asarray(data["y"], float)
    sigma = float(data["sigma"])
    mu = ces_values_np(theta, data)
    return -0.5*((y-mu)/sigma)**2 - np.log(sigma*np.sqrt(2*np.pi))

def ces_trials_jax(theta, data):
    y = jnp.asarray(data["y"])
    sigma = float(data["sigma"])
    mu = ces_values_jax(theta, data)
    return -0.5*((y-mu)/sigma)**2 - jnp.log(sigma*jnp.sqrt(2*jnp.pi))

def trial_loglik_np(model, theta, data):
    return {
        "binary": binary_trials_np,
        "categorical": categorical_trials_np,
        "ces": ces_trials_np,
    }[model](theta, data)

def trial_loglik_jax(model, theta, data):
    return {
        "binary": binary_trials_jax,
        "categorical": categorical_trials_jax,
        "ces": ces_trials_jax,
    }[model](theta, data)

DEFAULTS = {
    "binary": {
        "bounds": [(0.02,0.98),(0.1,8.0)],
        "prior_mean": np.array([0.5,2.0]),
        "prior_sd": np.array([0.25,2.0]),
    },
    "categorical": {
        "bounds": [(0.02,0.98),(0.1,8.0)],
        "prior_mean": np.array([0.5,2.0]),
        "prior_sd": np.array([0.25,2.0]),
    },
    "ces": {
        "bounds": [(0.02,0.98),(-0.8,0.8)],
        "prior_mean": np.array([0.5,0.0]),
        "prior_sd": np.array([0.25,0.5]),
    },
}

def generate_dataset(model, theta_true, rng, n_trials=250, sigma=0.1, prior_scale=1.0):
    theta_true = np.asarray(theta_true, float)
    spec = DEFAULTS[model]
    prior_mean = spec["prior_mean"].copy()
    prior_precision = prior_scale * np.diag(1.0/spec["prior_sd"]**2)

    if model == "binary":
        alpha, beta = theta_true
        q = np.zeros(2)
        choices = np.zeros(n_trials, int)
        rewards = np.zeros(n_trials)
        reward_probs = np.array([0.75,0.25])
        for t in range(n_trials):
            p = _softmax_np(beta*q)
            a = rng.choice(2, p=p)
            r = float(rng.random() < reward_probs[a])
            choices[t] = a
            rewards[t] = r
            q[a] += alpha*(r-q[a])
        data = {"choices":choices, "rewards":rewards}

    elif model == "categorical":
        alpha, beta = theta_true
        q = np.zeros(3)
        choices = np.zeros(n_trials, int)
        rewards = np.zeros(n_trials)
        reward_probs = np.array([0.75,0.50,0.25])
        for t in range(n_trials):
            p = _softmax_np(beta*q)
            a = rng.choice(3, p=p)
            r = float(rng.random() < reward_probs[a])
            choices[t] = a
            rewards[t] = r
            q[a] += alpha*(r-q[a])
        data = {"choices":choices, "rewards":rewards}

    elif model == "ces":
        alpha, rho = theta_true
        x1 = rng.uniform(0.5,2.0,n_trials)
        x2 = rng.uniform(0.5,2.0,n_trials)
        inner = alpha*x1**rho + (1-alpha)*x2**rho
        values = inner**(1.0/rho)
        y = values + rng.normal(0.0, sigma, n_trials)
        data = {"x1":x1, "x2":x2, "y":y, "sigma":float(sigma)}
    else:
        raise ValueError(model)

    return Dataset(
        model=model,
        data=data,
        theta_true=theta_true,
        bounds=list(spec["bounds"]),
        prior_mean=prior_mean,
        prior_precision=prior_precision,
    )
