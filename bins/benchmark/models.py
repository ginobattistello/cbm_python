"""
RL model likelihoods shared by both Python arms of the benchmark.

Defined ONCE here so the fork and the pristine-CBM arm fit literally the
same objective — any difference in results is then attributable to the
optimizer/curvature, which is the question being asked. Verbatim (up to
formatting) from examples/example_RL.py.

Each model has two forms:
    <model>          -> scalar summed log-likelihood (what CBM needs)
    <model>_trials   -> per-trial log-likelihood vector, enabling the
                        fork's Gauss-Newton curvature (Mod 5). The
                        pristine arm cannot use these — its optimize()
                        has no trial_func parameter — which is precisely
                        the capability difference under test.
"""

import numpy as np


def RL_model(parameters, data):
    """Single learning rate. theta = [alpha*, beta*] (unconstrained)."""
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


def RL_model_trials(parameters, data):
    choices, rewards = data
    alpha = 1 / (1 + np.exp(-parameters[0]))
    beta = np.exp(parameters[1])
    Q = np.zeros(2)
    lls = np.zeros(len(choices))
    for t in range(len(choices)):
        action = int(choices[t])
        arg = beta * Q
        e = np.exp(arg - np.max(arg))
        p = e / np.sum(e)
        lls[t] = np.log(p[action] + 1e-10)
        delta = rewards[t] - Q[action]
        Q[action] = Q[action] + alpha * delta
    return lls


def RL2_model(parameters, data):
    """Dual learning rate. theta = [a_pos*, a_neg*, beta*]."""
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


# ══════════════════════════════════════════════════════════════════
# Value-function models — neuroeconomic risky choice
# ──────────────────────────────────────────────────────────────────
# Task: on each trial the subject chooses between a SURE amount s and a
# GAMBLE (amount g with probability p, else 0). Expected-utility choice
# with a softmax:
#
#     U(sure)   = v(s)
#     U(gamble) = p * v(g)
#     P(gamble) = 1 / (1 + exp(-beta * (U(gamble) - U(sure))))
#
# LIN : v(x) = x                     theta = [beta*]              (1 param)
# POW : v(x) = x^rho                 theta = [rho*, beta*]        (2 params)
#
# The pair is NESTED: POW reduces exactly to LIN at rho = 1, which is
# what makes it a clean model-recovery test (LIN-generated data is a
# point in POW's parameter space, so any preference for POW is a genuine
# complexity penalty question, not a misspecification artifact).
#
# rho < 1 is concave utility = risk aversion — the classical
# neuroeconomic value function (Kahneman & Tversky 1979's value function
# for gains; Bernoulli 1738 for the log limit). rho > 1 is convex = risk
# seeking.
#
# Parameterization, matching the RL models' convention (unconstrained
# space, transformed inside the likelihood):
#     rho  = exp(theta)   -> (0, inf), rho=1 at theta=0, the LIN point
#     beta = exp(theta)   -> (0, inf)
# Amounts are scaled to ~[0,1] by the generator so that x^rho stays
# numerically tame and beta has a comparable meaning across models.
# ══════════════════════════════════════════════════════════════════
def _choice_loglik(dU, beta, chose_gamble):
    """Shared softmax log-likelihood for the value-function models.
    Uses the numerically stable -log(1+exp(-z)) form."""
    z = beta * dU
    z = np.where(chose_gamble > 0.5, z, -z)
    # log sigmoid(z), stable for large |z|
    return -np.logaddexp(0.0, -z)


def LIN_model_trials(parameters, data):
    """Linear (risk-neutral) utility. theta = [log beta]."""
    sure, gamble, prob, chose = data
    beta = np.exp(parameters[0])
    dU = prob * gamble - sure
    return _choice_loglik(dU, beta, chose)


def LIN_model(parameters, data):
    return float(np.sum(LIN_model_trials(parameters, data)))


def POW_model_trials(parameters, data):
    """Power (CRRA) utility v(x)=x^rho. theta = [log rho, log beta].
    rho<1 concave = risk averse; rho=1 recovers LIN exactly."""
    sure, gamble, prob, chose = data
    rho = np.exp(parameters[0])
    beta = np.exp(parameters[1])
    dU = prob * np.power(gamble, rho) - np.power(sure, rho)
    return _choice_loglik(dU, beta, chose)


def POW_model(parameters, data):
    return float(np.sum(POW_model_trials(parameters, data)))


# name -> (scalar model, per-trial model, n_params, prior mean)
MODELS = {
    "RL": (RL_model, RL_model_trials, 2, np.zeros(2)),
    "RL2": (RL2_model, RL2_model_trials, 3, np.zeros(3)),
    "LIN": (LIN_model, LIN_model_trials, 1, np.zeros(1)),
    "POW": (POW_model, POW_model_trials, 2, np.zeros(2)),
}

# Which candidate models compete, per task family. Model recovery is
# scored within a family (fitting an RL model to gamble data is
# meaningless — the data format differs).
FAMILIES = {"rl": ("RL", "RL2"), "value": ("LIN", "POW")}

PRIOR_VARIANCE = 10.0   # matches examples/example_RL.py


def to_native(theta, model_name):
    """Unconstrained fit space -> interpretable parameters.

    Returns a dict keyed by native parameter name, so the value-function
    models (rho) and the RL models (alphas) can share one results table.
    """
    theta = np.asarray(theta, dtype=float)
    if model_name == "RL":
        a = 1 / (1 + np.exp(-theta[0]))
        return dict(alpha_pos=a, alpha_neg=a, beta=np.exp(theta[1]))
    if model_name == "RL2":
        return dict(alpha_pos=1 / (1 + np.exp(-theta[0])),
                    alpha_neg=1 / (1 + np.exp(-theta[1])),
                    beta=np.exp(theta[2]))
    if model_name == "LIN":
        # rho is fixed at 1 by construction — report it so LIN and POW
        # rows line up in the same table.
        return dict(rho=1.0, beta=np.exp(theta[0]))
    if model_name == "POW":
        return dict(rho=np.exp(theta[0]), beta=np.exp(theta[1]))
    raise ValueError(f"unknown model {model_name!r}")
