"""
Shared data generator for the three-arm robustness benchmark
(this fork / pristine CBM / MATLAB VBA).

Generates ground-truth RL datasets across a stress grid and writes them
ONCE, in two formats reading the same bytes:
    benchmark/data/<cell>.npz   — for the Python arms
    benchmark/data/<cell>.mat   — for the MATLAB VBA arm
Both arms must fit IDENTICAL data; that is the whole point of generating
here rather than in each arm.

Stress axes (DEV.md workstream 5)
---------------------------------
n_trials    30 / 100 / 300     — how much data per subject
beta        0.5 / 1 / 3 / 8    — choice stochasticity. Low beta means
                                near-random choices: the likelihood
                                barely constrains alpha, which is the
                                flat-direction regime §2.1's
                                Gauss-Newton curvature targets.
generator   RL / RL2           — single vs dual learning rate, for the
                                model-recovery confusion matrix.
degenerate  perseverative / near-deterministic / extreme-alpha cells,
            run separately (see DEGENERATE_CELLS).

Models
------
RL  : Q(a) += alpha * (r - Q(a))                       theta = [alpha*, beta*]
RL2 : Q(a) += alpha_pos * delta  if delta >= 0
      Q(a) += alpha_neg * delta  otherwise             theta = [a_pos*, a_neg*, beta*]
where * denotes the unconstrained space the toolboxes fit in:
alpha = sigmoid(theta), beta = exp(theta). Identical to examples/example_RL.py.

Usage
-----
    python benchmark/simulate.py            # write the full grid
    python benchmark/simulate.py --quick    # small grid, for smoke tests
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR / "data"

# ── Stress grid ────────────────────────────────────────────────────
N_TRIALS = (30, 100, 300)
BETAS = (0.5, 1.0, 3.0, 8.0)
GENERATORS = ("RL", "RL2")
N_SUBJECTS = 60          # per cell; enough that a 10-point accuracy
                         # difference between arms is not pure noise
REWARD_PROBS = (0.7, 0.3)
MASTER_SEED = 20260812
MIN_ASYMMETRY = 0.35     # minimum |alpha_pos - alpha_neg| for RL2 truth;
                         # see draw_true_params for why this is required

QUICK = dict(n_trials=(100,), betas=(0.5, 3.0), n_subjects=10)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ── Generative process ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════
# Observation noise for CHOICE models — the lapse rate
# ──────────────────────────────────────────────────────────────────
# These are binary-outcome models, so there is no residual to perturb
# the way there is in regression. The standard contamination for choice
# data is a LAPSE: on a fraction of trials the subject ignores the model
# entirely and responds at random (attention lapse, misclick, guess).
#
# This is the right noise for a robustness test because the fitted model
# CANNOT represent it — neither RL nor POW has a lapse parameter — so the
# fits are genuinely misspecified, which is what we want to stress. (By
# contrast, jittering the true parameters keeps the model perfectly
# specified and only widens the spread.)
#
# Note the lapse enters the CHOICE only. For RL the agent still learns
# from whatever outcome the lapse choice produced, exactly as a real
# subject would — so a lapse also perturbs the downstream Q-trajectory,
# not just the single trial.
LAPSE_RATE = 0.10


def _apply_lapse(action, n_options, lapse, rng):
    """With probability `lapse`, replace the model's choice by a uniform
    random one. Returns the (possibly replaced) action."""
    if lapse > 0 and rng.random() < lapse:
        return int(rng.integers(n_options))
    return action


def simulate_subject(n_trials, alpha_pos, alpha_neg, beta, reward_probs, rng,
                     lapse=0.0):
    """One subject's choices/rewards from the (possibly asymmetric) RL agent.

    Returns (choices, rewards) with choices in {0, 1}. Note the update
    uses `delta >= 0` for the positive branch, matching RL2_model in
    examples/example_RL.py exactly (a subtle but real difference from
    `> 0`: a zero prediction error takes the positive rate).
    """
    n_opt = len(reward_probs)
    Q = np.zeros(n_opt)
    choices = np.zeros(n_trials, dtype=np.int64)
    rewards = np.zeros(n_trials)
    for t in range(n_trials):
        arg = beta * Q
        e = np.exp(arg - np.max(arg))
        p = e / e.sum()
        a = _apply_lapse(int(rng.choice(n_opt, p=p)), n_opt, lapse, rng)
        r = float(rng.binomial(1, reward_probs[a]))
        choices[t] = a
        rewards[t] = r
        delta = r - Q[a]
        Q[a] += (alpha_pos if delta >= 0 else alpha_neg) * delta
    return choices, rewards


def draw_true_params(generator, beta, rng):
    """Ground-truth parameters for one subject.

    Learning rates are drawn in a mid range where they are identifiable
    in principle (0.2-0.8); beta is fixed per cell (it IS the stress
    axis) with mild jitter so subjects are not clones.

    RL2 asymmetry — calibrated 2026-08-12, do not weaken.
      The first draft used a_pos ~ U(0.5,0.85), a_neg ~ U(0.1,0.4)
      (mean gap 0.42). At T=100 that yields a mean evidence gap of only
      -0.05 nats: RL2's likelihood gain is almost exactly cancelled by
      the Laplace complexity penalty for its third parameter, so model
      recovery sat near chance for ALL THREE ARMS. That measures the
      simulation, not the toolboxes.
      Direct check with a strong asymmetry (0.85 / 0.10) confirmed the
      evidence machinery is fine: mean gap +2.6 nats and 92% correct at
      T=100, +20.3 nats and 100% at T=1000.
      So the asymmetry is now drawn to be genuinely detectable —
      |a_pos - a_neg| >= MIN_ASYMMETRY — and the residual difficulty
      comes from the stress axes (trials, beta), which is the intent.
    """
    beta_i = float(beta * np.exp(rng.normal(0, 0.10)))
    if generator == "RL":
        a = float(rng.uniform(0.2, 0.8))
        return dict(alpha_pos=a, alpha_neg=a, beta=beta_i)
    for _ in range(100):
        a_pos = float(rng.uniform(0.55, 0.95))
        a_neg = float(rng.uniform(0.05, 0.35))
        if a_pos - a_neg >= MIN_ASYMMETRY:
            break
    return dict(alpha_pos=a_pos, alpha_neg=a_neg, beta=beta_i)


def make_cell(name, generator, n_trials, beta, n_subjects, seed,
              reward_probs=REWARD_PROBS, param_override=None):
    """Build one grid cell: n_subjects datasets sharing a condition."""
    rng = np.random.default_rng(seed)
    choices = np.zeros((n_subjects, n_trials), dtype=np.int64)
    rewards = np.zeros((n_subjects, n_trials))
    truth = []
    for i in range(n_subjects):
        p = draw_true_params(generator, beta, rng)
        if param_override is not None:
            p = param_override(p, rng)
        choices[i], rewards[i] = simulate_subject(
            n_trials, p["alpha_pos"], p["alpha_neg"], p["beta"],
            reward_probs, rng)
        truth.append(p)

    # Choice balance: a subject who always picks one option carries no
    # information about alpha. Recorded so the analysis can separate
    # "the fitter failed" from "the data had nothing to fit".
    frac_opt0 = choices.mean(axis=1)
    return dict(
        name=name, generator=generator, n_trials=int(n_trials),
        beta=float(beta), n_subjects=int(n_subjects), seed=int(seed),
        choices=choices, rewards=rewards,
        true_alpha_pos=np.array([t["alpha_pos"] for t in truth]),
        true_alpha_neg=np.array([t["alpha_neg"] for t in truth]),
        true_beta=np.array([t["beta"] for t in truth]),
        frac_choice0=frac_opt0,
    )


# ── Degenerate / adversarial cells ─────────────────────────────────
def _perseverative(p, rng):
    """Huge beta + high alpha: agent locks onto one option almost
    immediately. Choice entropy collapses -> alpha is unidentifiable."""
    p = dict(p); p["beta"] = 50.0
    p["alpha_pos"] = p["alpha_neg"] = 0.95
    return p


def _alpha_at_zero(p, rng):
    """alpha ~ 0: Q never moves, so choices are ~random and BOTH alpha
    and beta become unidentifiable (the classic flat direction)."""
    p = dict(p); p["alpha_pos"] = p["alpha_neg"] = 0.01
    return p


def _alpha_at_one(p, rng):
    """alpha = 1: Q jumps to the last outcome. Sigmoid-space truth is
    at +inf, so the MAP is pulled to the bound -> Mod 9's at_hard_bounds
    check and the Laplace interior-optimum assumption are both stressed."""
    p = dict(p); p["alpha_pos"] = p["alpha_neg"] = 0.99
    return p


DEGENERATE_CELLS = (
    # (name, generator, n_trials, beta, override)
    ("degen_perseverative", "RL", 100, 50.0, _perseverative),
    ("degen_alpha0", "RL", 100, 3.0, _alpha_at_zero),
    ("degen_alpha1", "RL", 100, 3.0, _alpha_at_one),
    ("degen_fewtrials", "RL", 15, 3.0, None),
)


# ══════════════════════════════════════════════════════════════════
# Value-function task (neuroeconomic risky choice) — added 2026-08-12
# ──────────────────────────────────────────────────────────────────
# Each trial: a SURE amount vs a GAMBLE (amount g with probability p).
# Amounts are on a 0-1 scale so x^rho stays numerically tame and beta
# means the same thing across models (see models.py).
#
# TWO SUB-GRID DESIGNS, because the two questions genuinely conflict —
# the RL grid tuned its ranges for model recovery and thereby throttled
# parameter recovery (found 2026-08-12; see DEV.md §8). Rather than
# compromise, each question gets the design that can answer it:
#
#   "recovery"  — WIDE true-parameter ranges. Correlation with truth is
#                 bounded by how much true variance exists, so a narrow
#                 range caps recovery no matter how good the fitter is.
#                 Includes rho near 1 (i.e. near-linear subjects).
#   "selection" — parameters held AWAY from the nesting point (rho far
#                 from 1) and more trials, so the two models are
#                 genuinely distinguishable and the confusion matrix
#                 measures the toolbox rather than the design.
# ══════════════════════════════════════════════════════════════════
VALUE_N_TRIALS = (30, 100, 300)
VALUE_BETAS = (0.3, 1.0, 3.0)      # on the AMOUNT_SCALE below
VALUE_GENERATORS = ("LIN", "POW")

# Amount scale — calibrated 2026-08-12, do not shrink back toward 1.
#   x^rho is nearly flat for x < 1, so amounts on a 0-1 scale make rho
#   weakly identified and let estimates run away (observed rho up to
#   31.5 before this was fixed). Measured rho recovery, T=100 / T=300:
#     amounts 0.05-1 : r = 0.60 / 0.82   (the original, bad choice)
#     amounts 0.5-10 : r = 0.92 / 0.97   <- adopted
#     amounts 5-100  : r = 0.67 / 0.79   (large x^rho overflows the
#                                         useful softmax range)
# This is a TASK-design constant, not a fitting constant: all three arms
# see the same amounts.
AMOUNT_SCALE = 10.0

# rho ranges per design intent (POW only; LIN has rho == 1 by definition)
RHO_RECOVERY = (0.30, 1.70)   # wide, spans risk-averse -> risk-seeking
RHO_SELECTION = (0.35, 0.60)  # far from 1, i.e. clearly non-linear


def simulate_value_subject(n_trials, rho, beta, rng, lapse=0.0):
    """One subject's risky choices under v(x) = x^rho.

    Gambles are drawn so the two options are often close in expected
    value — that is what makes the choice informative about curvature.
    A trial where one option dominates tells you nothing about rho.
    """
    sure = rng.uniform(0.10, 0.60, n_trials) * AMOUNT_SCALE
    prob = rng.uniform(0.20, 0.80, n_trials)
    # Draw the gamble so its EV brackets the sure amount: gamble
    # amount = sure/prob scaled by a factor around 1.
    gamble = np.clip(sure / prob * rng.uniform(0.70, 1.45, n_trials),
                     0.05 * AMOUNT_SCALE, 1.0 * AMOUNT_SCALE)
    dU = prob * np.power(gamble, rho) - np.power(sure, rho)
    p_gamble = 1.0 / (1.0 + np.exp(-beta * dU))
    chose = (rng.random(n_trials) < p_gamble).astype(float)
    if lapse > 0:
        lapsed = rng.random(n_trials) < lapse
        chose[lapsed] = (rng.random(lapsed.sum()) < 0.5).astype(float)
    return sure, gamble, prob, chose


def make_value_cell(name, generator, n_trials, beta, n_subjects, seed,
                    design):
    rng = np.random.default_rng(seed)
    lo, hi = RHO_RECOVERY if design == "recovery" else RHO_SELECTION
    S = np.zeros((n_subjects, n_trials))
    G = np.zeros((n_subjects, n_trials))
    P = np.zeros((n_subjects, n_trials))
    C = np.zeros((n_subjects, n_trials))
    true_rho, true_beta = [], []
    for i in range(n_subjects):
        rho = 1.0 if generator == "LIN" else float(rng.uniform(lo, hi))
        b = float(beta * np.exp(rng.normal(0, 0.10)))
        S[i], G[i], P[i], C[i] = simulate_value_subject(n_trials, rho, b, rng)
        true_rho.append(rho)
        true_beta.append(b)
    return dict(
        name=name, generator=generator, family="value", design=design,
        n_trials=int(n_trials), beta=float(beta),
        n_subjects=int(n_subjects), seed=int(seed),
        sure=S, gamble=G, prob=P, chose=C,
        true_rho=np.array(true_rho), true_beta=np.array(true_beta),
        frac_gamble=C.mean(axis=1),
    )


# ── Writing ────────────────────────────────────────────────────────
def write_cell(cell, out_dir):
    """Write one cell as .npz (Python arms) and .mat (MATLAB VBA arm)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{cell['name']}.npz"
    np.savez_compressed(npz, **{k: v for k, v in cell.items()})
    # MATLAB: same arrays; strings stay strings, ints become doubles.
    savemat(out_dir / f"{cell['name']}.mat",
            {k: v for k, v in cell.items()}, do_compression=True)
    return npz


def cell_seed(*parts):
    """Deterministic per-cell seed. Uses blake2b, NOT builtin hash():
    hash() is randomized per process (PYTHONHASHSEED), which would make
    the datasets differ between runs — fatal for a benchmark whose whole
    premise is that all three arms fit identical data."""
    key = "|".join(str(p) for p in parts).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return MASTER_SEED + int.from_bytes(digest, "big") % 100_000


def _wide_alpha(p, rng):
    """Wide-range RL truth (design='recovery'). The default RL ranges were
    tuned so the RL/RL2 evidence gap is detectable, which narrows the
    alpha spread and CAPS parameter recovery (correlation is bounded by
    true variance). This override restores the spread — see DEV.md §8."""
    p = dict(p)
    if p["alpha_pos"] == p["alpha_neg"]:          # RL: one rate
        a = float(rng.uniform(0.05, 0.95))
        p["alpha_pos"] = p["alpha_neg"] = a
    else:                                          # RL2: two rates
        p["alpha_pos"] = float(rng.uniform(0.05, 0.95))
        p["alpha_neg"] = float(rng.uniform(0.05, 0.95))
    return p


def build_grid(quick=False):
    """The original RL grid (design='selection' by construction)."""
    n_trials_grid = QUICK["n_trials"] if quick else N_TRIALS
    betas = QUICK["betas"] if quick else BETAS
    n_sub = QUICK["n_subjects"] if quick else N_SUBJECTS

    cells = []
    # Seeds are derived deterministically from the cell identity, so
    # adding a cell never perturbs the data of existing ones.
    for gen in GENERATORS:
        for nt in n_trials_grid:
            for b in betas:
                name = f"{gen}_T{nt}_b{str(b).replace('.', 'p')}"
                seed = cell_seed(gen, nt, b)
                c = make_cell(name, gen, nt, b, n_sub, seed)
                c["family"], c["design"] = "rl", "selection"
                cells.append(c)
    if not quick:
        for name, gen, nt, b, override in DEGENERATE_CELLS:
            seed = cell_seed(name)
            c = make_cell(name, gen, nt, b, n_sub, seed,
                          param_override=override)
            c["family"], c["design"] = "rl", "degenerate"
            cells.append(c)
    return cells


def build_rl_wide_grid():
    """RL/RL2 with WIDE true-parameter ranges (design='recovery').
    Answers 'how well can these toolboxes recover alpha when the design
    permits it', which the selection-tuned grid structurally cannot."""
    cells = []
    for gen in GENERATORS:
        for nt in N_TRIALS:
            for b in BETAS:
                name = f"{gen}wide_T{nt}_b{str(b).replace('.', 'p')}"
                c = make_cell(name, gen, nt, b, N_SUBJECTS,
                              cell_seed("wide", gen, nt, b),
                              param_override=_wide_alpha)
                c["family"], c["design"] = "rl", "recovery"
                cells.append(c)
    return cells


# ══════════════════════════════════════════════════════════════════
# THE CLEAN GRID — the one the report is built from (2026-08-12)
# ──────────────────────────────────────────────────────────────────
# Two generating models only:
#   RL  — reinforcement learning, single learning rate. theta = (alpha, beta)
#   POW — non-linear value function v(x)=x^rho.        theta = (rho, beta)
# Both get 10% lapse trials (see LAPSE_RATE above).
#
# Parameter ranges are WIDE and fixed, because the report's job is
# parameter recovery: correlation with truth is capped by how much true
# variance exists (DEV.md §9.7). Trial count is fixed at a realistic 200
# so the figures show toolbox differences rather than a data-quantity
# sweep — the stress-axis sweep lives in the older grids.
#
# Each dataset is still fitted with BOTH candidates of its family
# (RL vs RL2, POW vs LIN) because section 6's AUC is defined from the
# evidence gap between two competing models. Only the confusion-matrix
# figure was dropped.
CLEAN_T = 200
CLEAN_N_SUBJECTS = 120          # one cell per model, so all subjects here
CLEAN_ALPHA = (0.05, 0.95)
CLEAN_BETA_RL = (0.8, 8.0)
CLEAN_RHO = (0.30, 1.70)
CLEAN_BETA_POW = (0.3, 3.0)


def build_clean_grid(lapse=LAPSE_RATE):
    """One cell per generating model, wide predefined parameter ranges."""
    cells = []

    # ---- RL ----
    rng = np.random.default_rng(cell_seed("clean", "RL", lapse))
    ch = np.zeros((CLEAN_N_SUBJECTS, CLEAN_T), dtype=np.int64)
    rw = np.zeros((CLEAN_N_SUBJECTS, CLEAN_T))
    ta, tb = [], []
    for i in range(CLEAN_N_SUBJECTS):
        a = float(rng.uniform(*CLEAN_ALPHA))
        b = float(np.exp(rng.uniform(np.log(CLEAN_BETA_RL[0]),
                                     np.log(CLEAN_BETA_RL[1]))))
        ch[i], rw[i] = simulate_subject(CLEAN_T, a, a, b, REWARD_PROBS, rng,
                                        lapse=lapse)
        ta.append(a); tb.append(b)
    cells.append(dict(
        name="RL", generator="RL", family="rl", design="clean",
        n_trials=CLEAN_T, beta=0.0, n_subjects=CLEAN_N_SUBJECTS,
        seed=cell_seed("clean", "RL", lapse), lapse=float(lapse),
        choices=ch, rewards=rw,
        true_alpha_pos=np.array(ta), true_alpha_neg=np.array(ta),
        true_beta=np.array(tb), frac_choice0=ch.mean(axis=1)))

    # ---- POW ----
    rng = np.random.default_rng(cell_seed("clean", "POW", lapse))
    S = np.zeros((CLEAN_N_SUBJECTS, CLEAN_T))
    G = np.zeros((CLEAN_N_SUBJECTS, CLEAN_T))
    P = np.zeros((CLEAN_N_SUBJECTS, CLEAN_T))
    C = np.zeros((CLEAN_N_SUBJECTS, CLEAN_T))
    tr, tb2 = [], []
    for i in range(CLEAN_N_SUBJECTS):
        r_ = float(rng.uniform(*CLEAN_RHO))
        b = float(np.exp(rng.uniform(np.log(CLEAN_BETA_POW[0]),
                                     np.log(CLEAN_BETA_POW[1]))))
        S[i], G[i], P[i], C[i] = simulate_value_subject(CLEAN_T, r_, b, rng,
                                                        lapse=lapse)
        tr.append(r_); tb2.append(b)
    cells.append(dict(
        name="POW", generator="POW", family="value", design="clean",
        n_trials=CLEAN_T, beta=0.0, n_subjects=CLEAN_N_SUBJECTS,
        seed=cell_seed("clean", "POW", lapse), lapse=float(lapse),
        sure=S, gamble=G, prob=P, chose=C,
        true_rho=np.array(tr), true_beta=np.array(tb2),
        frac_gamble=C.mean(axis=1)))
    return cells


# ══════════════════════════════════════════════════════════════════
# BOUNDARY STRESS GRID (2026-08-12)
# ──────────────────────────────────────────────────────────────────
# The clean grid samples the well-behaved interior. This one walks each
# model's own parameter from deep inside its range out to (and past) the
# edge, holding everything else fixed, to answer three questions:
#   1. does cross-arm agreement survive when fits become unreliable?
#   2. how gracefully does recovery degrade?
#   3. does any diagnostic PREDICT the failure? (Mod 10 validation)
#
# A screening probe showed this axis is the one that actually bites:
# skewed predictors barely mattered (rho recovery 0.93 -> 0.93) while
# rho near zero destroyed it (0.93 -> -0.18). Each level is its own cell
# with the parameter held at a FIXED value, so per-cell recovery is not
# meaningful — |bias| and the diagnostics are what to read.
BOUND_T = 150
BOUND_N = 40
BOUND_ALPHA_LEVELS = (0.001, 0.01, 0.05, 0.20, 0.50, 0.80, 0.95, 0.99, 0.999)
BOUND_RHO_LEVELS = (0.02, 0.08, 0.20, 0.50, 0.90, 1.30, 1.70, 2.50, 4.00)


def build_boundary_grid(lapse=0.0):
    """One cell per parameter level, per model. Lapse off by default so
    the boundary effect is not confounded with contamination."""
    cells = []
    for a in BOUND_ALPHA_LEVELS:
        tag = str(a).replace('.', 'p')
        rng = np.random.default_rng(cell_seed("bound", "RL", a, lapse))
        ch = np.zeros((BOUND_N, BOUND_T), dtype=np.int64)
        rw = np.zeros((BOUND_N, BOUND_T))
        tb = []
        for i in range(BOUND_N):
            b = float(np.exp(rng.uniform(np.log(1.5), np.log(5.0))))
            ch[i], rw[i] = simulate_subject(BOUND_T, a, a, b, REWARD_PROBS,
                                            rng, lapse=lapse)
            tb.append(b)
        cells.append(dict(
            name=f"RLa{tag}", generator="RL", family="rl", design="boundary",
            level=float(a), n_trials=BOUND_T, beta=0.0, n_subjects=BOUND_N,
            seed=cell_seed("bound", "RL", a, lapse), lapse=float(lapse),
            choices=ch, rewards=rw,
            true_alpha_pos=np.full(BOUND_N, a),
            true_alpha_neg=np.full(BOUND_N, a),
            true_beta=np.array(tb), frac_choice0=ch.mean(axis=1)))
    for r_ in BOUND_RHO_LEVELS:
        tag = str(r_).replace('.', 'p')
        rng = np.random.default_rng(cell_seed("bound", "POW", r_, lapse))
        S = np.zeros((BOUND_N, BOUND_T)); G = np.zeros((BOUND_N, BOUND_T))
        P = np.zeros((BOUND_N, BOUND_T)); C = np.zeros((BOUND_N, BOUND_T))
        tb = []
        for i in range(BOUND_N):
            b = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
            S[i], G[i], P[i], C[i] = simulate_value_subject(
                BOUND_T, r_, b, rng, lapse=lapse)
            tb.append(b)
        cells.append(dict(
            name=f"POWr{tag}", generator="POW", family="value",
            design="boundary", level=float(r_), n_trials=BOUND_T, beta=0.0,
            n_subjects=BOUND_N, seed=cell_seed("bound", "POW", r_, lapse),
            lapse=float(lapse), sure=S, gamble=G, prob=P, chose=C,
            true_rho=np.full(BOUND_N, r_), true_beta=np.array(tb),
            frac_gamble=C.mean(axis=1)))
    return cells


# ══════════════════════════════════════════════════════════════════
# HBI GRID (2026-08-13)
# ──────────────────────────────────────────────────────────────────
# For the hierarchical benchmark (DEV.md §15). Two things make this grid
# different from every other one here:
#
# 1. MIXED POPULATIONS. HBI infers how many subjects belong to each model,
#    so a cell where every subject came from one model tells us almost
#    nothing — the answer is pinned at ~1 and cannot move. Each cell mixes
#    the two candidates of a family at a KNOWN ratio, which is the ground
#    truth for group-level recovery and, more importantly, puts the group
#    verdict somewhere it can actually be perturbed.
#
# 2. A CONTESTED MIDDLE. The 50/50 and 70/30 cells are where a small change
#    in individual fits can flip the group answer. That is precisely where
#    Mod 11 was predicted to matter, so those cells carry the experiment;
#    the 100/0 cells are the control that should be stable either way.
#
# Deliberately small — HBI refits every subject on every iteration, so it
# costs 10-30x an individual_fit run. 40 subjects x 150 trials keeps a full
# sweep (2 arms x 4 seeds x 6 cells) inside a lunch break.
HBI_T = 150
HBI_N = 40
# fraction of subjects drawn from the COMPLEX model of the pair
HBI_MIXES = (0.0, 0.3, 0.5, 0.7, 1.0)


def build_hbi_grid(lapse=LAPSE_RATE):
    """Mixed-population cells for the hierarchical benchmark.

    RL family:    simple = RL  (one alpha), complex = RL2 (two alphas)
    value family: simple = LIN (v(x)=x),    complex = POW (v(x)=x^rho)

    `true_model` records, per subject, which generator produced them (0 =
    simple, 1 = complex) — that is the ground truth HBI's responsibilities
    are scored against.
    """
    cells = []

    # ---- RL family: RL vs RL2 ----
    for mix in HBI_MIXES:
        tag = f"{int(round(mix * 100)):03d}"
        seed = cell_seed("hbi", "rl", mix, lapse)
        rng = np.random.default_rng(seed)
        n_complex = int(round(mix * HBI_N))
        # Interleave rather than block the two groups, so any subject-order
        # effect in the inference cannot be mistaken for a mixture effect.
        is_complex = np.zeros(HBI_N, dtype=int)
        if n_complex:
            is_complex[rng.choice(HBI_N, n_complex, replace=False)] = 1
        ch = np.zeros((HBI_N, HBI_T), dtype=np.int64)
        rw = np.zeros((HBI_N, HBI_T))
        tap, tan, tb = [], [], []
        for i in range(HBI_N):
            b = float(np.exp(rng.uniform(np.log(1.5), np.log(5.0))))
            if is_complex[i]:
                # RL2: learning rates far enough apart that the extra
                # parameter is genuinely earning its keep (MIN_ASYMMETRY
                # exists for the same reason — see DEV.md §9).
                ap = float(rng.uniform(0.55, 0.85))
                an = float(rng.uniform(0.05, 0.25))
            else:
                ap = an = float(rng.uniform(0.15, 0.65))
            ch[i], rw[i] = simulate_subject(HBI_T, ap, an, b, REWARD_PROBS,
                                            rng, lapse=lapse)
            tap.append(ap); tan.append(an); tb.append(b)
        cells.append(dict(
            name=f"RLmix{tag}", generator="RL", family="rl", design="hbi",
            level=float(mix), n_trials=HBI_T, beta=0.0, n_subjects=HBI_N,
            seed=seed, lapse=float(lapse), choices=ch, rewards=rw,
            true_model=is_complex,
            true_alpha_pos=np.array(tap), true_alpha_neg=np.array(tan),
            true_beta=np.array(tb), frac_choice0=ch.mean(axis=1)))

    # ---- value family: LIN vs POW ----
    for mix in HBI_MIXES:
        tag = f"{int(round(mix * 100)):03d}"
        seed = cell_seed("hbi", "value", mix, lapse)
        rng = np.random.default_rng(seed)
        n_complex = int(round(mix * HBI_N))
        is_complex = np.zeros(HBI_N, dtype=int)
        if n_complex:
            is_complex[rng.choice(HBI_N, n_complex, replace=False)] = 1
        S = np.zeros((HBI_N, HBI_T)); G = np.zeros((HBI_N, HBI_T))
        P = np.zeros((HBI_N, HBI_T)); C = np.zeros((HBI_N, HBI_T))
        tr, tb = [], []
        for i in range(HBI_N):
            b = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
            # rho = 1 IS the linear model, so simple subjects get exactly 1
            # and complex ones get a curvature well away from it.
            r_ = 1.0 if not is_complex[i] else float(
                rng.choice([-1, 1]) * rng.uniform(0.35, 0.60) + 1.0)
            S[i], G[i], P[i], C[i] = simulate_value_subject(
                HBI_T, r_, b, rng, lapse=lapse)
            tr.append(r_); tb.append(b)
        cells.append(dict(
            name=f"VALmix{tag}", generator="POW", family="value",
            design="hbi", level=float(mix), n_trials=HBI_T, beta=0.0,
            n_subjects=HBI_N, seed=seed, lapse=float(lapse),
            sure=S, gamble=G, prob=P, chose=C, true_model=is_complex,
            true_rho=np.array(tr), true_beta=np.array(tb),
            frac_gamble=C.mean(axis=1)))
    return cells


def build_value_grid(design):
    """Value-function task: LIN vs POW, one sub-grid per design intent."""
    cells = []
    for gen in VALUE_GENERATORS:
        for nt in VALUE_N_TRIALS:
            for b in VALUE_BETAS:
                tag = "R" if design == "recovery" else "S"
                name = f"{gen}{tag}_T{nt}_b{str(b).replace('.', 'p')}"
                cells.append(make_value_cell(
                    name, gen, nt, b, N_SUBJECTS,
                    cell_seed("value", design, gen, nt, b), design))
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="small grid for smoke tests")
    ap.add_argument("--grid", default="clean",
                    choices=("clean", "boundary", "hbi", "rl", "rl_wide",
                             "value_recovery", "value_selection"),
                    help="which grid to build (default clean)")
    ap.add_argument("--lapse", type=float, default=LAPSE_RATE,
                    help="lapse rate for the clean grid (default 0.10)")
    ap.add_argument("--out", default=None, help="output directory")
    args = ap.parse_args()

    builders = {
        "clean": lambda: build_clean_grid(lapse=args.lapse),
        "boundary": lambda: build_boundary_grid(lapse=0.0),
        "hbi": lambda: build_hbi_grid(lapse=args.lapse),
        "rl": lambda: build_grid(quick=args.quick),
        "rl_wide": build_rl_wide_grid,
        "value_recovery": lambda: build_value_grid("recovery"),
        "value_selection": lambda: build_value_grid("selection"),
    }
    default_dir = {"clean": "clean", "boundary": "boundary", "hbi": "hbi",
                   "rl": "quick" if args.quick else "grid",
                   "rl_wide": "rl_wide",
                   "value_recovery": "value_recovery",
                   "value_selection": "value_selection"}[args.grid]
    out_dir = Path(args.out) if args.out else (DATA_DIR / default_dir)
    cells = builders[args.grid]()

    manifest = []
    for c in cells:
        write_cell(c, out_dir)
        manifest.append({k: c[k] for k in
                         ("name", "generator", "n_trials", "beta",
                          "n_subjects", "seed", "family", "design")})
        extra = (f"gamble frac {c['frac_gamble'].mean():.2f}"
                 if c.get("family") == "value"
                 else f"choice0 frac {c['frac_choice0'].mean():.2f}")
        print(f"  {c['name']:28s} gen={c['generator']:4s} T={c['n_trials']:3d} "
              f"beta={c['beta']:5.1f} n={c['n_subjects']:2d}  {extra}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n{len(cells)} cells -> {out_dir}")
    print(f"manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
