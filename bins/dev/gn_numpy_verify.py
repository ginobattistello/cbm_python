"""
Verification harness: NumPy Gauss-Newton curvature (optimization.py Mod 5)
vs the Mod 2 finite-difference/eigenvalue-clip fallback.

Fits the same subject (RL2 model, example_RL.py's generator/prior) both
ways and compares parameters, Hessian eigenvalues, and log-evidence.
"""
import sys
import time
from pathlib import Path

import numpy as np

# Make the repo root importable so this runs without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cbm.optimization import BFGSOptimizer, Config


def RL2_model(parameters, data):
    choices, rewards = data
    alpha_pos = 1 / (1 + np.exp(-parameters[0]))
    alpha_neg = 1 / (1 + np.exp(-parameters[1]))
    beta = np.exp(parameters[2])
    Q = np.zeros(2)
    log_lik = 0.0
    for t in range(len(choices)):
        action = int(choices[t])
        arg = beta * Q
        e = np.exp(arg - np.max(arg))
        p = e / np.sum(e)
        log_lik += np.log(p[action] + 1e-10)
        delta = rewards[t] - Q[action]
        if delta >= 0:
            Q[action] = Q[action] + alpha_pos * delta
        else:
            Q[action] = Q[action] + alpha_neg * delta
    return log_lik


def RL2_model_trials(parameters, data):
    choices, rewards = data
    alpha_pos = 1 / (1 + np.exp(-parameters[0]))
    alpha_neg = 1 / (1 + np.exp(-parameters[1]))
    beta = np.exp(parameters[2])
    Q = np.zeros(2)
    lls = np.zeros(len(choices))
    for t in range(len(choices)):
        action = int(choices[t])
        arg = beta * Q
        e = np.exp(arg - np.max(arg))
        p = e / np.sum(e)
        lls[t] = np.log(p[action] + 1e-10)
        delta = rewards[t] - Q[action]
        if delta >= 0:
            Q[action] = Q[action] + alpha_pos * delta
        else:
            Q[action] = Q[action] + alpha_neg * delta
    return lls


def generate(n_trials, a_pos, a_neg, beta, rp, seed):
    rng = np.random.default_rng(seed)
    Q = np.zeros(len(rp))
    ch = np.zeros(n_trials, int)
    rw = np.zeros(n_trials)
    for t in range(n_trials):
        p = np.exp(beta * Q) / np.sum(np.exp(beta * Q))
        a = rng.choice(len(rp), p=p)
        ch[t] = a
        r = rng.binomial(1, rp[a])
        rw[t] = r
        pe = r - Q[a]
        Q[a] += (a_pos if pe > 0 else a_neg) * pe
    return ch, rw


if __name__ == "__main__":
    data = generate(100, 0.8, 0.4, 3.0, [0.7, 0.3], seed=0)
    prior_mean = np.zeros(3)
    prior_precision = np.eye(3) / 10.0  # variance=10 -> precision=1/10

    def objective(theta):
        diff = theta - prior_mean
        log_prior = -0.5 * diff @ prior_precision @ diff
        return -(RL2_model(theta, data) + log_prior)

    def trial_func(theta):
        return RL2_model_trials(theta, data)

    config = Config(d=3, num_init=1, verbose=False)

    print("=" * 70)
    print("MOD 2 fallback (finite-difference Hessian, eigenvalue-clipped)")
    print("=" * 70)
    opt1 = BFGSOptimizer(3, config=config)
    t0 = time.perf_counter()
    r1 = opt1.optimize(objective, x_init=prior_mean.copy())
    t1 = time.perf_counter() - t0
    eig1 = np.linalg.eigvalsh(r1.hess)
    print(f"x = {r1.x}")
    print(f"f = {r1.f:.10f}")
    print(f"hess_method = {r1.hess_method}")
    print(f"hess eigenvalues = {eig1}")
    print(f"time = {t1:.4f}s")

    print("\n" + "=" * 70)
    print("MOD 5 (Gauss-Newton curvature, VBA-style)")
    print("=" * 70)
    opt2 = BFGSOptimizer(3, config=config)
    t0 = time.perf_counter()
    r2 = opt2.optimize(objective, x_init=prior_mean.copy(),
                        trial_func=trial_func, prior_precision=prior_precision)
    t2 = time.perf_counter() - t0
    eig2 = np.linalg.eigvalsh(r2.hess)
    print(f"x = {r2.x}")
    print(f"f = {r2.f:.10f}")
    print(f"hess_method = {r2.hess_method}")
    print(f"hess eigenvalues = {eig2}")
    print(f"time = {t2:.4f}s")

    print("\n" + "=" * 70)
    print("Diff")
    print("=" * 70)
    print(f"max|x1 - x2| = {np.max(np.abs(r1.x - r2.x)):.3e}")
    print(f"|f1 - f2|    = {abs(r1.f - r2.f):.3e}")
    print(f"min eig: Mod2={eig1.min():.4f}  Mod5={eig2.min():.4f}")

    logdet1 = np.linalg.slogdet(r1.hess)[1]
    logdet2 = np.linalg.slogdet(r2.hess)[1]
    loglik1 = RL2_model(r1.x, data)
    loglik2 = RL2_model(r2.x, data)
    lme1 = loglik1 + 0.5 * 3 * np.log(2 * np.pi) - 0.5 * logdet1
    lme2 = loglik2 + 0.5 * 3 * np.log(2 * np.pi) - 0.5 * logdet2
    print(f"log-evidence: Mod2={lme1:.6f}  Mod5={lme2:.6f}  diff={lme1 - lme2:.3e}")
