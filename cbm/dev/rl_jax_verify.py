"""
Verification harness: JAX autodiff port of example_RL.py models.

Checks, on synthetic data:
  1. JAX log-likelihood == original NumPy log-likelihood
  2. jax.grad == central finite-difference gradient (what the toolbox does now)
  3. exact jax.hessian is available (replaces (n+1)^2 finite differencing)
  4. Gauss-Newton outer-product curvature is PSD by construction
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, hessian, jacrev
from functools import partial

jax.config.update("jax_enable_x64", True)  # match float64; evidence needs it


# ---------------------------------------------------------------------------
# ORIGINAL NumPy models (verbatim from example_RL.py) — reference truth
# ---------------------------------------------------------------------------
def RL_model_np(parameters, data):
    choices, rewards = data
    alpha = 1 / (1 + np.exp(-parameters[0]))
    beta = np.exp(parameters[1])
    Q = np.zeros(2)
    log_lik = 0.0
    for t in range(len(choices)):
        action = int(choices[t])
        arg = beta * Q
        e = np.exp(arg - np.max(arg))
        p = e / np.sum(e)
        log_lik += np.log(p[action] + 1e-10)
        delta = rewards[t] - Q[action]
        Q[action] = Q[action] + alpha * delta
    return log_lik


def RL2_model_np(parameters, data):
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


# ---------------------------------------------------------------------------
# JAX ports — same math, differentiable, compiled with lax.scan
# ---------------------------------------------------------------------------
# Key changes vs NumPy:
#   * Python for-loop over trials  -> jax.lax.scan (Q is the carry)
#   * Q[action] = ...              -> Q.at[action].set(...)   (functional)
#   * if delta >= 0: ... else: ... -> jnp.where(...)          (no host branch)
#   * trailing sum split out       -> per-trial log-liks returned, enables Gauss-Newton

@partial(jax.jit, static_argnums=())
def rl_trial_logliks(parameters, choices, rewards):
    alpha = jax.nn.sigmoid(parameters[0])
    beta = jnp.exp(parameters[1])

    def step(Q, obs):
        action, reward = obs
        arg = beta * Q
        p = jax.nn.softmax(arg)                 # numerically stable softmax
        ll = jnp.log(p[action] + 1e-10)
        delta = reward - Q[action]
        Q = Q.at[action].set(Q[action] + alpha * delta)
        return Q, ll

    Q0 = jnp.zeros(2)
    _, lls = jax.lax.scan(step, Q0, (choices, rewards))
    return lls                                   # per-trial, shape (T,)


@jax.jit
def rl2_trial_logliks(parameters, choices, rewards):
    alpha_pos = jax.nn.sigmoid(parameters[0])
    alpha_neg = jax.nn.sigmoid(parameters[1])
    beta = jnp.exp(parameters[2])

    def step(Q, obs):
        action, reward = obs
        arg = beta * Q
        p = jax.nn.softmax(arg)
        ll = jnp.log(p[action] + 1e-10)
        delta = reward - Q[action]
        alpha_eff = jnp.where(delta >= 0, alpha_pos, alpha_neg)   # the branch, vectorized
        Q = Q.at[action].set(Q[action] + alpha_eff * delta)
        return Q, ll

    Q0 = jnp.zeros(2)
    _, lls = jax.lax.scan(step, Q0, (choices, rewards))
    return lls


def rl_loglik(parameters, choices, rewards):
    return jnp.sum(rl_trial_logliks(parameters, choices, rewards))


def rl2_loglik(parameters, choices, rewards):
    return jnp.sum(rl2_trial_logliks(parameters, choices, rewards))



# ---------------------------------------------------------------------------
# Gauss-Newton / outer-product-of-gradients curvature (solution #2).
# For objective = sum_t ll_t, GN precision = sum_t grad(ll_t) grad(ll_t)^T.
# PSD by construction -> no eigenvalue clipping needed.
# ---------------------------------------------------------------------------
def gauss_newton_precision(trial_logliks_fn, parameters, choices, rewards):
    # jacrev gives d ll_t / d theta as (T, d); GN = J^T J
    J = jacrev(trial_logliks_fn)(parameters, choices, rewards)   # (T, d)
    return J.T @ J


def central_fd_grad(f, x, eps=1e-5):
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps; xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2 * eps)
    return g


# ---------------------------------------------------------------------------
# Synthetic data (same generator shape as example_RL.py)
# ---------------------------------------------------------------------------
def generate(n_trials, a_pos, a_neg, beta, rp, seed):
    rng = np.random.default_rng(seed)
    Q = np.zeros(len(rp)); ch = np.zeros(n_trials, int); rw = np.zeros(n_trials)
    for t in range(n_trials):
        p = np.exp(beta * Q) / np.sum(np.exp(beta * Q))
        a = rng.choice(len(rp), p=p); ch[t] = a
        r = rng.binomial(1, rp[a]); rw[t] = r
        pe = r - Q[a]
        Q[a] += (a_pos if pe > 0 else a_neg) * pe
    return ch, rw


if __name__ == "__main__":
    ch_np, rw_np = generate(100, 0.8, 0.4, 3.0, [0.7, 0.3], seed=0)
    ch = jnp.asarray(ch_np); rw = jnp.asarray(rw_np)

    print("=" * 64)
    print("1) LOG-LIKELIHOOD EQUIVALENCE (JAX vs original NumPy)")
    for name, np_fn, jx_fn, th in [
        ("RL ", RL_model_np, rl_loglik, np.array([0.3, 0.7])),
        ("RL2", RL2_model_np, rl2_loglik, np.array([0.5, -0.2, 0.7])),
    ]:
        ll_np = np_fn(th, (ch_np, rw_np))
        ll_jx = float(jx_fn(jnp.asarray(th), ch, rw))
        print(f"   {name}: numpy={ll_np:+.10f}  jax={ll_jx:+.10f}  "
              f"|diff|={abs(ll_np - ll_jx):.2e}")

    print("=" * 64)
    print("2) GRADIENT: exact autodiff vs central finite differences")
    for name, jx_ll, jx_tr, th in [
        ("RL ", rl_loglik, rl_trial_logliks, np.array([0.3, 0.7])),
        ("RL2", rl2_loglik, rl2_trial_logliks, np.array([0.5, -0.2, 0.7])),
    ]:
        g_ad = np.asarray(grad(jx_ll)(jnp.asarray(th), ch, rw))
        g_fd = central_fd_grad(lambda x: float(jx_ll(jnp.asarray(x), ch, rw)), th)
        print(f"   {name}: max|grad_ad - grad_fd| = {np.max(np.abs(g_ad - g_fd)):.2e}")

        print("=" * 64 if name == "RL2" else "", end="")
        # 3) exact Hessian + 4) Gauss-Newton PSD check
        H = np.asarray(hessian(jx_ll)(jnp.asarray(th), ch, rw))
        GN = np.asarray(gauss_newton_precision(jx_tr, jnp.asarray(th), ch, rw))
        H_eig = np.linalg.eigvalsh(H)
        GN_eig = np.linalg.eigvalsh(GN)
        print(f"\n   {name} exact Hessian eigenvalues:        "
              f"{np.array2string(H_eig, precision=3)}")
        print(f"   {name} Gauss-Newton eigenvalues (>=0?):  "
              f"{np.array2string(GN_eig, precision=3)}  min={GN_eig.min():.3e}")
    print("=" * 64)
    print("Autodiff gradient cost is O(1) model passes, independent of d.")
