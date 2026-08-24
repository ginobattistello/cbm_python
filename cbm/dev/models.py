"""Three small model families shared by the validation experiment."""
from __future__ import annotations
import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None


def _softmax_np(x):
    z = x - np.max(x)
    ez = np.exp(z)
    return ez / ez.sum()


def _softmax_jax(x):
    z = x - jnp.max(x)
    ez = jnp.exp(z)
    return ez / ez.sum()


def binary_trials_np(theta, data):
    alpha, beta = theta
    q = np.zeros(2)
    out = []
    for a, r in zip(data["choice"], data["reward"]):
        p = _softmax_np(beta * q)
        out.append(np.log(np.clip(p[a], 1e-12, 1)))
        q[a] += alpha * (r - q[a])
    return np.asarray(out)


def binary_trials_jax(theta, data):
    alpha, beta = theta
    choice = jnp.asarray(data["choice"], dtype=jnp.int32)
    reward = jnp.asarray(data["reward"])
    def step(q, xs):
        a, r = xs
        p = _softmax_jax(beta * q)
        ll = jnp.log(jnp.clip(p[a], 1e-12, 1))
        q = q.at[a].add(alpha * (r - q[a]))
        return q, ll
    _, ll = jax.lax.scan(step, jnp.zeros(2), (choice, reward))
    return ll


def categorical_trials_np(theta, data):
    alpha, beta = theta
    q = np.zeros(3)
    out = []
    for a, r in zip(data["choice"], data["reward"]):
        p = _softmax_np(beta * q)
        out.append(np.log(np.clip(p[a], 1e-12, 1)))
        q[a] += alpha * (r - q[a])
    return np.asarray(out)


def categorical_trials_jax(theta, data):
    alpha, beta = theta
    choice = jnp.asarray(data["choice"], dtype=jnp.int32)
    reward = jnp.asarray(data["reward"])
    def step(q, xs):
        a, r = xs
        p = _softmax_jax(beta * q)
        ll = jnp.log(jnp.clip(p[a], 1e-12, 1))
        q = q.at[a].add(alpha * (r - q[a]))
        return q, ll
    _, ll = jax.lax.scan(step, jnp.zeros(3), (choice, reward))
    return ll


def ces_trials_np(theta, data):
    alpha, rho = theta
    x1, x2, y, sigma = data["x1"], data["x2"], data["y"], data["sigma"]
    v = (alpha * x1**rho + (1-alpha) * x2**rho) ** (1/rho)
    return -0.5*((y-v)/sigma)**2 - np.log(sigma*np.sqrt(2*np.pi))


def ces_trials_jax(theta, data):
    alpha, rho = theta
    x1, x2 = jnp.asarray(data["x1"]), jnp.asarray(data["x2"])
    y, sigma = jnp.asarray(data["y"]), data["sigma"]
    v = (alpha * x1**rho + (1-alpha) * x2**rho) ** (1/rho)
    return -0.5*((y-v)/sigma)**2 - jnp.log(sigma*jnp.sqrt(2*jnp.pi))


def simulate_binary(rng, theta=(0.35, 3.0), n=250):
    alpha, beta = theta
    q = np.zeros(2)
    choice, reward = [], []
    probs = np.array([0.75, 0.25])
    for _ in range(n):
        p = _softmax_np(beta*q)
        a = rng.choice(2, p=p)
        r = float(rng.random() < probs[a])
        choice.append(a); reward.append(r)
        q[a] += alpha*(r-q[a])
    return {"choice": np.asarray(choice), "reward": np.asarray(reward)}


def simulate_categorical(rng, theta=(0.35, 3.0), n=250):
    alpha, beta = theta
    q = np.zeros(3)
    choice, reward = [], []
    probs = np.array([0.75, 0.50, 0.25])
    for _ in range(n):
        p = _softmax_np(beta*q)
        a = rng.choice(3, p=p)
        r = float(rng.random() < probs[a])
        choice.append(a); reward.append(r)
        q[a] += alpha*(r-q[a])
    return {"choice": np.asarray(choice), "reward": np.asarray(reward)}


def simulate_ces(rng, theta=(0.60, 0.30), n=250, sigma=0.10):
    alpha, rho = theta
    x1 = rng.uniform(0.5, 2.0, n)
    x2 = rng.uniform(0.5, 2.0, n)
    v = (alpha*x1**rho + (1-alpha)*x2**rho) ** (1/rho)
    y = v + rng.normal(0, sigma, n)
    return {"x1": x1, "x2": x2, "y": y, "sigma": sigma}
