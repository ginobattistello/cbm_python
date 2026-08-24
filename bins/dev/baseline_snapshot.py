"""
Baseline snapshot harness (DEV.md §7 steps 2 and 4).

Records the numbers every later MODIFICATION block is judged against:
per-subject MAP parameters, log-evidence, Hessian eigenvalues, convergence
flags and timings, for both RL models on the example_RL.py data.

Usage
-----
    python cbm/dev/baseline_snapshot.py --save cbm/dev/baseline.json
        Write a snapshot (do this BEFORE touching anything).

    python cbm/dev/baseline_snapshot.py --compare cbm/dev/baseline.json
        Re-run and diff against a saved snapshot. Exit code 1 if any
        recorded quantity moved by more than --tol.

The data generator is copied verbatim from examples/example_RL.py (same
np.random.seed(42), same call order), so the subjects are bit-identical to
the ones the example fits. HBI is deliberately NOT included: it is slow and
its exceedance probabilities are Monte-Carlo, so it is a poor regression
signal. The individual fits are what the optimization work actually changes.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cbm.individual_fit import individual_fit  # noqa: E402


# ---------------------------------------------------------------------------
# Models — verbatim from examples/example_RL.py
# ---------------------------------------------------------------------------
def RL_model(parameters, data):
    choices, rewards = data
    alpha = 1 / (1 + np.exp(-parameters[0]))
    beta = np.exp(parameters[1])
    Q = np.zeros(2)
    log_lik = 0.0
    for t in range(len(choices)):
        action = int(choices[t])
        arg_exp = beta * Q
        exp_values = np.exp(arg_exp - np.max(arg_exp))
        p = exp_values / np.sum(exp_values)
        log_lik += np.log(p[action] + 1e-10)
        delta = rewards[t] - Q[action]
        Q[action] = Q[action] + alpha * delta
    return log_lik


def RL_model_trials(parameters, data):
    choices, rewards = data
    alpha = 1 / (1 + np.exp(-parameters[0]))
    beta = np.exp(parameters[1])
    Q = np.zeros(2)
    log_liks = np.zeros(len(choices))
    for t in range(len(choices)):
        action = int(choices[t])
        arg_exp = beta * Q
        exp_values = np.exp(arg_exp - np.max(arg_exp))
        p = exp_values / np.sum(exp_values)
        log_liks[t] = np.log(p[action] + 1e-10)
        delta = rewards[t] - Q[action]
        Q[action] = Q[action] + alpha * delta
    return log_liks


def RL2_model(parameters, data):
    choices, rewards = data
    alpha_pos = 1 / (1 + np.exp(-parameters[0]))
    alpha_neg = 1 / (1 + np.exp(-parameters[1]))
    beta = np.exp(parameters[2])
    Q = np.zeros(2)
    log_lik = 0.0
    for t in range(len(choices)):
        action = int(choices[t])
        arg_exp = beta * Q
        exp_values = np.exp(arg_exp - np.max(arg_exp))
        p = exp_values / np.sum(exp_values)
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
    log_liks = np.zeros(len(choices))
    for t in range(len(choices)):
        action = int(choices[t])
        arg_exp = beta * Q
        exp_values = np.exp(arg_exp - np.max(arg_exp))
        p = exp_values / np.sum(exp_values)
        log_liks[t] = np.log(p[action] + 1e-10)
        delta = rewards[t] - Q[action]
        if delta >= 0:
            Q[action] = Q[action] + alpha_pos * delta
        else:
            Q[action] = Q[action] + alpha_neg * delta
    return log_liks


def generate_data(n_trials, alpha_pos, alpha_neg, beta, reward_probs):
    """Verbatim from examples/example_RL.py."""
    n_options = len(reward_probs)
    Q = np.zeros(n_options)
    choices = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials)
    for t in range(n_trials):
        p = np.exp(beta * Q) / np.sum(np.exp(beta * Q))
        action = np.random.choice(n_options, p=p)
        choices[t] = action
        reward = np.random.binomial(1, reward_probs[action])
        rewards[t] = reward
        prediction_error = reward - Q[action]
        if prediction_error > 0:
            Q[action] = Q[action] + alpha_pos * prediction_error
        else:
            Q[action] = Q[action] + alpha_neg * prediction_error
    return choices, rewards


def build_data():
    """Reproduce example_RL.py's 40 subjects exactly (seed 42, same order)."""
    np.random.seed(42)
    n_trials = 100
    reward_probs = [0.7, 0.3]
    all_data = []
    for _ in range(30):
        alpha_pos = 0.8 + np.random.rand() * 0.05
        alpha_neg = 0.4 + np.random.rand() * 0.05
        beta = 3.0 + np.random.rand() * 0.5
        all_data.append(generate_data(n_trials, alpha_pos, alpha_neg, beta, reward_probs))
    for _ in range(10):
        alpha = 0.1 + np.random.rand() * 0.05
        beta = 1.0 + np.random.rand() * 0.5
        all_data.append(generate_data(n_trials, alpha, alpha, beta, reward_probs))
    return all_data


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------
def _fit_block(all_data, model, model_trials, prior_mean, use_trials):
    """Fit all subjects and record the quantities we regress on."""
    np.random.seed(0)  # pin the optimizer's random initializations
    t0 = time.perf_counter()
    fit = individual_fit(
        all_data, model, prior_mean, 10,
        config={"num_init": 1, "verbose": False},
        model_trials=model_trials if use_trials else None,
    )
    elapsed = time.perf_counter() - t0

    eigs = [np.linalg.eigvalsh(H).tolist() for H in fit.math.hessian]
    return {
        "parameters": fit.output.parameters.tolist(),
        "log_evidence": fit.output.log_evidence.tolist(),
        "loglik": fit.math.loglik.tolist(),
        "log_det_hessian": fit.math.log_det_hessian.tolist(),
        "hessian_eigenvalues": eigs,
        "min_eigenvalue": [min(e) for e in eigs],
        "flag": fit.math.flag.tolist(),
        "seconds": elapsed,
    }


def take_snapshot():
    all_data = build_data()
    snap = {"n_subjects": len(all_data), "blocks": {}}

    for name, model, trials, pm in [
        ("RL", RL_model, RL_model_trials, np.array([0, 0])),
        ("RL2", RL2_model, RL2_model_trials, np.array([0, 0, 0])),
    ]:
        # Both curvature paths: Mod 5 (Gauss-Newton) and the Mod 2 fallback.
        snap["blocks"][f"{name}_gauss_newton"] = _fit_block(
            all_data, model, trials, pm, use_trials=True)
        snap["blocks"][f"{name}_finite_diff_clipped"] = _fit_block(
            all_data, model, trials, pm, use_trials=False)

    return snap


# Fields compared numerically; "seconds" is recorded but never regressed on.
_NUMERIC_FIELDS = [
    "parameters", "log_evidence", "loglik", "log_det_hessian",
    "hessian_eigenvalues", "min_eigenvalue", "flag",
]


def compare(old, new, tol):
    """Return a list of human-readable differences."""
    diffs = []
    for block in sorted(set(old["blocks"]) | set(new["blocks"])):
        if block not in old["blocks"]:
            diffs.append(f"{block}: present now, absent in baseline")
            continue
        if block not in new["blocks"]:
            diffs.append(f"{block}: in baseline, absent now")
            continue
        o, n = old["blocks"][block], new["blocks"][block]
        for field in _NUMERIC_FIELDS:
            a = np.asarray(o[field], dtype=float)
            b = np.asarray(n[field], dtype=float)
            if a.shape != b.shape:
                diffs.append(f"{block}.{field}: shape {a.shape} -> {b.shape}")
                continue
            delta = np.nanmax(np.abs(a - b)) if a.size else 0.0
            if delta > tol:
                diffs.append(f"{block}.{field}: max|delta| = {delta:.3e}")
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", metavar="PATH", help="write a snapshot to PATH")
    ap.add_argument("--compare", metavar="PATH", help="diff against snapshot at PATH")
    ap.add_argument("--tol", type=float, default=1e-8,
                    help="max allowed absolute change (default 1e-8)")
    args = ap.parse_args()

    if not args.save and not args.compare:
        ap.error("give --save or --compare")

    print("Running fits (4 blocks x 40 subjects)...", flush=True)
    snap = take_snapshot()
    for name, blk in snap["blocks"].items():
        print(f"  {name:32s} {blk['seconds']:7.2f}s  "
              f"sum(lme) = {np.sum(blk['log_evidence']):.6f}")

    if args.save:
        Path(args.save).write_text(json.dumps(snap, indent=1))
        print(f"\nSnapshot written to {args.save}")

    if args.compare:
        old = json.loads(Path(args.compare).read_text())
        diffs = compare(old, snap, args.tol)
        print()
        if diffs:
            print(f"CHANGED vs {args.compare} (tol={args.tol:g}):")
            for d in diffs:
                print(f"  {d}")
            print("\nA change is not necessarily wrong — but per DEV.md §7 step 4 "
                  "it must be explained, not just observed.")
            return 1
        print(f"UNCHANGED vs {args.compare} (tol={args.tol:g}) — all fields match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
