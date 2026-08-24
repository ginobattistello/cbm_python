"""Small simulators using the standardized y/X data format."""

import numpy as np


def _softmax(x):
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def binary_subject(rng, theta=(0.35, 3.0), n_trials=150):
    alpha, beta = theta
    reward_prob = np.array([0.75, 0.25])

    q = np.zeros(2)
    y = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials)

    for t in range(n_trials):
        p = _softmax(beta * q)
        choice = rng.choice(2, p=p)
        reward = float(rng.random() < reward_prob[choice])

        y[t] = choice
        rewards[t] = reward
        q[choice] += alpha * (reward - q[choice])

    return {
        "y": y,
        "X": {"reward": rewards},
    }


def categorical_subject(
    rng,
    theta=(0.35, 3.0),
    n_trials=150,
    n_options=3,
):
    alpha, beta = theta
    reward_prob = np.linspace(0.75, 0.25, n_options)

    q = np.zeros(n_options)
    y = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials)

    for t in range(n_trials):
        p = _softmax(beta * q)
        choice = rng.choice(n_options, p=p)
        reward = float(rng.random() < reward_prob[choice])

        y[t] = choice
        rewards[t] = reward
        q[choice] += alpha * (reward - q[choice])

    return {
        "y": y,
        "X": {
            "reward": rewards,
            "n_options": n_options,
        },
    }


def continuous_subject(
    rng,
    theta=(1.0, 2.0),
    n_trials=150,
    sigma=0.5,
):
    intercept, slope = theta
    x = rng.uniform(-2.0, 2.0, n_trials)
    mu = intercept + slope * x
    y = mu + rng.normal(0.0, sigma, n_trials)

    return {
        "y": y,
        "X": {
            "x": x,
            "sigma": sigma,
        },
    }
