"""
Publication-style figures for the cross-implementation benchmark.

============================================================================
WHY THIS BENCHMARK EXISTS
============================================================================

This fork of CBM changes how the posterior curvature is computed (Gauss-Newton
`H = J'J + prior_precision` instead of an eigenvalue-clipped finite-difference
Hessian), plus nine other modifications documented in DEV.md.

Any such change raises one question: **did we break agreement with the
reference implementations?**  So we fit the *same simulated subjects* with
three independent pieces of software and compare them:

    fork_gn    this fork              (cbm/, Gauss-Newton curvature)
    cbm_orig   pristine pre-fork CBM  (vendored at commit e72193f)
    vba        MATLAB VBA toolbox     (a wholly separate codebase)

The third arm matters because the first two share ancestry. If the fork and
pristine CBM both drifted the same way we would not notice; VBA was written by
different people in a different language using a different inference procedure
(variational Bayes rather than Laplace-MAP), so agreeing with it is real
evidence of correctness rather than shared lineage.

Ground truth is known — the data are simulated — so we can measure not just
"do the arms agree with each other" but "do they recover the right answer".

============================================================================
THE DATA BEING FITTED  (benchmark/data/clean, from simulate.py)
============================================================================

Two generating models, one cell of 120 subjects x 200 trials each, both with
10% lapse trials (a random choice on 1 trial in 10) as the noise source:

    RL   reinforcement learning, single learning rate.  Params: alpha, beta
         Rival candidate = RL2 (separate learning rates for gains/losses),
         i.e. the MORE complex model. RL is nested inside RL2.

    POW  non-linear value function v(x) = x^rho.        Params: rho, beta
         Rival candidate = LIN (v(x) = x), i.e. the LESS complex model.
         LIN is nested inside POW at rho = 1.

Every dataset is fitted with BOTH candidates of its family. Figures 1-4 use
only the correctly-specified fit; Figure 5 needs both, because model selection
is precisely a comparison between them.

============================================================================
WHAT EACH FIGURE ASKS
============================================================================

  Fig 1  Are the estimates unbiased, and equally so in all three arms?
  Fig 2  Does each parameter recover its own truth and not its neighbour's?
  Fig 3  Do the arms agree with EACH OTHER?   <-- the regression check
  Fig 4  What did Gauss-Newton actually change: the optimum, or the evidence?
  Fig 5  Do the arms make the same model-selection decisions?

Figure 3 is the one to look at after changing anything in cbm/. Figures 1, 2
and 5 can all move together for innocent reasons (the simulator changed, the
parameter ranges changed) without indicating a bug. Figure 3 compares arms to
each other on identical data, so it isolates *the fork* as the thing that
moved.

Deliberately absent: a model-recovery confusion matrix. With only two
candidates per family it adds nothing that Figure 5 does not already say.

    python benchmark/make_report_figures.py

Outputs to benchmark/results/figures/: fig1..fig5 as PDF + PNG, and
figures.md holding the captions.
"""

import math
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.colors import TwoSlopeNorm, Normalize            # noqa: E402
from matplotlib.patches import Patch                             # noqa: E402

BENCH = Path(__file__).resolve().parent
RESULTS = BENCH / "results"
FIGDIR = RESULTS / "figures"
GRID = "clean"

# ---------------------------------------------------------------------------
# Arms, in a FIXED order that never changes between figures. Colour follows
# the arm, so a reader who learns "orange = VBA" on figure 1 can carry that
# to figure 5. Colours are checked for CVD separation (deuteranopia /
# protanopia): blue-grey / vermilion / teal is a standard safe triad.
# ---------------------------------------------------------------------------
ARMS = ("fork_gn", "cbm_orig", "vba")
ARM_LABEL = {"fork_gn": "This fork (GN)",
             "cbm_orig": "Original CBM",
             "vba": "MATLAB VBA"}
ARM_COLOR = {"fork_gn": "#3B5B8C",     # deep blue
             "cbm_orig": "#C4622D",    # vermilion
             "vba": "#2E7D6F"}         # teal

# ---------------------------------------------------------------------------
# Per generating model: which rival it is compared against, and which
# parameters to track.
#
# The `log` flag says whether a parameter lives naturally on a log scale.
# beta (inverse temperature) is positive and multiplicative — a beta of 8 vs 4
# is the same perceptual step as 4 vs 2 — so all error and correlation work
# for beta is done on log(beta). alpha (a probability, 0-1) and rho (a
# curvature exponent near 1) are used as-is.
# ---------------------------------------------------------------------------
SPEC = {
    "RL": {
        "rival": "RL2",
        "rival_is_complex": True,       # RL2 has one more parameter than RL
        "params": [("alpha", "est_alpha_pos", "true_alpha_pos", False),
                   ("beta", "est_beta", "true_beta", True)],
    },
    "POW": {
        "rival": "LIN",
        "rival_is_complex": False,      # LIN has one FEWER parameter than POW
        "params": [("rho", "est_rho", "true_rho", False),
                   ("beta", "est_beta", "true_beta", True)],
    },
}

# Greek display names. Matplotlib mathtext, so they render as symbols.
PRETTY = {"alpha": r"$\alpha$", "beta": r"$\log\beta$", "rho": r"$\rho$"}


# ===========================================================================
# STYLE
# ===========================================================================

def set_style():
    """Journal-figure defaults: small sans text, hairline rules, no chart junk.

    Every figure in this script inherits these, so the five figures form a
    visually consistent set — the reader should never have to re-learn how to
    read one.
    """
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,       # despined: only left+bottom rules
        "axes.spines.right": False,
        "axes.axisbelow": True,         # grid behind the data, never over it
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,        # a legend box is pure chart junk
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.4,
        "grid.color": "#D8D8D8",
    })


def panel_letter(ax, letter, dx=-0.14, dy=1.06):
    """Bold A/B/C in the upper-left, outside the axes — journal convention."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    """Write both a vector PDF (for a manuscript) and a PNG (for screens)."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_rows():
    """All fits from all three arms, as a flat list of per-subject dicts.

    The two Python arms are pickled together by run_python_arms.py. The VBA
    arm comes from a MATLAB .mat and carries no ground truth of its own — so
    we join truth back on by (cell, subject), which is safe because all three
    arms fitted byte-identical simulated data.
    """
    rows = pickle.load(open(RESULTS / f"python_{GRID}.pkl", "rb"))

    vba_path = RESULTS / f"vba_{GRID}.mat"
    if not vba_path.exists():
        print(f"  ! {vba_path.name} missing — rendering Python arms only")
        return rows

    from scipy.io import loadmat
    m = loadmat(vba_path, squeeze_me=True, struct_as_record=False)
    truth = {(r["cell"], r["subject"]): r for r in rows}
    for r in np.atleast_1d(m["rows"]):
        t = truth.get((str(r.cell), int(r.subject)), {})
        rows.append(dict(
            arm="vba", cell=str(r.cell), generator=str(r.generator),
            fitted_model=str(r.fitted_model), subject=int(r.subject),
            est_alpha_pos=float(r.est_alpha_pos),
            est_alpha_neg=float(r.est_alpha_neg),
            est_rho=float(r.est_rho), est_beta=float(r.est_beta),
            # VBA reports a variational free energy. It is an evidence
            # approximation of a different KIND to CBM's Laplace value, so
            # absolute F is NOT comparable across arms. Only DIFFERENCES
            # between two models fitted by the SAME arm are — which is
            # exactly and only how figure 5 uses it.
            log_evidence=float(r.F),
            true_alpha_pos=t.get("true_alpha_pos", np.nan),
            true_rho=t.get("true_rho", np.nan),
            true_beta=t.get("true_beta", np.nan),
            log_det_hessian=np.nan))
    return rows


def val(row, key, logscale):
    """One numeric value, NaN when missing or unusable.

    Returning NaN rather than dropping the row keeps every arm's vector the
    same length and in the same subject order, which is what makes the
    cross-arm correlations in figure 3 valid.
    """
    v = row.get(key, np.nan)
    if v is None:
        return np.nan
    v = float(v)
    if logscale:
        return math.log(v) if v > 0 else np.nan
    return v


def corr(a, b):
    """Pearson r over the pairwise-complete cases."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def fits_of(rows, generator, arm):
    """Correctly-specified fits: data from `generator`, fitted BY `generator`.

    Figures 1-4 all condition on this. Fitting a model to its own data is the
    best case; if recovery fails here it will not improve under
    misspecification, so this is the right place to measure.
    """
    return [r for r in rows
            if r["arm"] == arm
            and r["generator"] == generator
            and r["fitted_model"] == generator]


# ===========================================================================
# FIGURE 1 — parameter estimate error
# ===========================================================================
#
# QUESTION: is each arm's estimate centred on the truth, and are the three
# arms equally biased?
#
# We plot the DIFFERENCE (estimated - predefined) rather than raw estimates.
# Raw estimates against truth would mostly show the range of the simulated
# parameters, which we chose ourselves and which is therefore uninformative.
# The error distribution shows the two things we actually care about:
#
#   where it is CENTRED   -> bias      (systematic over/under-estimation)
#   how WIDE it is        -> variance  (per-subject precision)
#
# A little shrinkage toward the prior mean is expected and correct — these
# are MAP estimates under N(0, 10) priors, not maximum likelihood — so the
# median sitting slightly off zero is not a bug. What would be a bug is one
# arm sitting somewhere different from the other two.
# ===========================================================================

def figure1(rows):
    models = list(SPEC)
    fig, axes = plt.subplots(len(models), len(ARMS),
                             figsize=(7.2, 4.6), sharey="row")

    letters = iter("ABCDEF")
    for i, gen in enumerate(models):
        params = SPEC[gen]["params"]

        # Collect first, so the shared y-limit for the row can be computed
        # before any drawing. Rows share y (sharey="row") because comparing
        # arms is the whole point; comparing alpha-error to rho-error across
        # rows is not, so rows are free to differ.
        per_arm = {}
        for arm in ARMS:
            sub = fits_of(rows, gen, arm)
            per_arm[arm] = [
                np.asarray([e for e in
                            (val(r, ek, lg) - val(r, tk, lg) for r in sub)
                            if np.isfinite(e)])
                for _, ek, tk, lg in params]

        for j, arm in enumerate(ARMS):
            ax = axes[i, j]
            data = per_arm[arm]
            pos = np.arange(len(params))

            # Violin for the shape of the distribution, box for the robust
            # summary, jittered points for the raw n. Together these show
            # the distribution without hiding the sample size, which a
            # bare boxplot would.
            vp = ax.violinplot(data, positions=pos, widths=0.66,
                               showextrema=False, showmedians=False)
            for body in vp["bodies"]:
                body.set_facecolor(ARM_COLOR[arm])
                body.set_alpha(0.22)
                body.set_edgecolor("none")

            bp = ax.boxplot(data, positions=pos, widths=0.16,
                            showfliers=False, patch_artist=True,
                            medianprops=dict(color="white", linewidth=1.1),
                            boxprops=dict(facecolor=ARM_COLOR[arm],
                                          edgecolor="none"),
                            whiskerprops=dict(color=ARM_COLOR[arm],
                                              linewidth=0.8),
                            capprops=dict(color=ARM_COLOR[arm],
                                          linewidth=0.8))
            del bp

            rng = np.random.default_rng(0)   # fixed: jitter must be stable
            for k, d in enumerate(data):
                ax.scatter(pos[k] + rng.uniform(-0.26, 0.26, len(d)), d,
                           s=1.6, color=ARM_COLOR[arm], alpha=0.35,
                           linewidths=0, zorder=3)

            # Zero = perfect recovery. The single most important reference
            # line on the panel, so it is drawn on top of the violins.
            ax.axhline(0, color="#333333", linewidth=0.7, linestyle=(0, (4, 3)),
                       zorder=4)

            ax.set_xticks(pos)
            ax.set_xticklabels([PRETTY[p[0]] for p in params])
            ax.grid(axis="y", alpha=0.5)
            ax.set_axisbelow(True)

            if j == 0:
                ax.set_ylabel(f"{gen}\nestimated $-$ true", labelpad=2)
            if i == 0:
                ax.set_title(ARM_LABEL[arm], pad=14, color=ARM_COLOR[arm],
                             fontweight="bold")
            panel_letter(ax, next(letters), dx=-0.10 if j == 0 else -0.05)

            # Numeric summary in-panel: a reader comparing arms should not
            # have to estimate a median by eye from a violin. Placed just
            # under the tick labels — far enough down to clear them, close
            # enough that it still reads as belonging to THIS panel and not
            # to the row below.
            txt = "\n".join(
                f"{p[0]}: med {np.median(d):+.3f}  "
                f"rmse {np.sqrt(np.mean(d ** 2)):.3f}"
                for p, d in zip(params, data) if len(d))
            ax.text(0.5, -0.16, txt, transform=ax.transAxes, ha="center",
                    va="top", fontsize=6.4, color="#555555", family="monospace")

    fig.suptitle("Figure 1 · Parameter estimate error, by arm",
                 fontsize=10, fontweight="bold", y=1.0)
    fig.text(0.5, 1.0 - 0.045,
             "Estimated minus predefined, per subject. Zero (dashed) is "
             "perfect recovery; a shifted median is bias, a wide violin is "
             "variance.",
             ha="center", fontsize=7.2, color="#555555")
    fig.subplots_adjust(hspace=0.46, wspace=0.16, top=0.86)
    save(fig, "fig1_estimate_error")


# ===========================================================================
# FIGURE 2 — parameter recovery
# ===========================================================================
#
# QUESTION: does each parameter recover ITSELF and not something else?
#
# Figure 1 shows whether estimates are centred correctly. It cannot show
# whether they TRACK the truth: an arm that returned the prior mean for every
# subject would have near-zero bias and be completely useless.
#
# So here we correlate each estimated parameter against each true parameter
# across the 120 subjects. Reading the resulting 2x2 matrix:
#
#   DIAGONAL   corr(est_x, true_x) — recovery. Want high.
#   OFF-DIAG   corr(est_x, true_y) — trade-off / confusion. Want ~0.
#
# A strong off-diagonal means the two parameters are not separately
# identifiable: the fit can raise one and lower the other and land in almost
# the same place, so their estimates contaminate each other.
#
# IMPORTANT INTERPRETIVE CEILING. The diagonal is bounded above by how much
# the TRUE parameters vary:
#
#       r  ~  SD_true / sqrt(SD_true^2 + SD_error^2)
#
# Narrowing the simulated parameter range lowers this ceiling even with a
# perfect estimator. So these numbers are only comparable between runs that
# used the SAME ranges — never quote them against a differently-configured
# grid. (Learned the hard way; see DEV.md section 9.)
# ===========================================================================

def figure2(rows):
    models = list(SPEC)
    fig, axes = plt.subplots(len(models), len(ARMS), figsize=(7.2, 4.9))

    # Diverging ramp centred on zero: correlation is a POLAR quantity
    # (negative and positive are opposite in meaning, zero is neutral), so a
    # sequential ramp would be the wrong encoding and would hide the sign.
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap = plt.get_cmap("RdBu_r")

    letters = iter("ABCDEF")
    for i, gen in enumerate(models):
        params = SPEC[gen]["params"]
        names = [p[0] for p in params]

        for j, arm in enumerate(ARMS):
            ax = axes[i, j]
            sub = fits_of(rows, gen, arm)

            # M[a][b] = corr(estimate of param a, truth of param b)
            M = np.array([[corr([val(r, ek, el) for r in sub],
                                [val(r, tk, tl) for r in sub])
                           for _, _, tk, tl in params]
                          for _, ek, _, el in params])

            ax.imshow(M, cmap=cmap, norm=norm, aspect="equal")

            for a in range(len(names)):
                for b in range(len(names)):
                    v = M[a, b]
                    # White text on saturated cells, dark on pale ones —
                    # otherwise the strong diagonal becomes unreadable.
                    ax.text(b, a, f"{v:+.2f}", ha="center", va="center",
                            fontsize=8,
                            fontweight="bold" if a == b else "normal",
                            color="white" if abs(v) > 0.55 else "#222222")

            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels([PRETTY[n] for n in names])
            ax.set_yticklabels([PRETTY[n] for n in names])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(length=0)

            # Axis roles are stated once per row/column rather than on every
            # panel — six copies of "true"/"estimated" is noise, and the
            # colourbar already names the quantity.
            if j == 0:
                ax.set_ylabel(f"{gen}\nestimated", labelpad=2)
            if i == len(models) - 1:
                ax.set_xlabel("true", labelpad=14)
            if i == 0:
                ax.set_title(ARM_LABEL[arm], pad=12, color=ARM_COLOR[arm],
                             fontweight="bold")
            panel_letter(ax, next(letters), dx=-0.24, dy=1.02)

            # n sits immediately under the x-label. It must not drift far
            # enough down to collide with the shared colourbar below.
            n_ok = sum(1 for r in sub
                       if np.isfinite(val(r, params[0][1], params[0][3])))
            ax.text(0.5, -0.19, f"n = {n_ok}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=6.8, color="#555555")

    fig.subplots_adjust(hspace=0.42, wspace=0.30, top=0.84, bottom=0.20)

    cax = fig.add_axes([0.30, 0.065, 0.40, 0.020])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax, orientation="horizontal")
    cb.set_label("Pearson r  (estimated vs true, across subjects)",
                 fontsize=7.2, labelpad=3)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2)

    fig.suptitle("Figure 2 · Parameter recovery", fontsize=10,
                 fontweight="bold", y=0.99)
    fig.text(0.5, 0.945,
             "Diagonal = each parameter recovering itself (want high). "
             "Off-diagonal = trade-off between parameters (want ~0).",
             ha="center", fontsize=7.2, color="#555555")
    save(fig, "fig2_parameter_recovery")


# ===========================================================================
# FIGURE 3 — cross-arm agreement    *** THE REGRESSION CHECK ***
# ===========================================================================
#
# QUESTION: do the three arms produce the same numbers on the same data?
#
# This is the figure to look at after changing anything in cbm/, and it is
# the only one that isolates the fork as the source of a change.
#
# Why it is more sensitive than figures 1-2: those compare each arm to the
# TRUTH, and truth-recovery is limited by noise in the simulated data. Every
# arm loses the same accuracy to that noise, so a real divergence between
# implementations can hide inside it. Here we compare arms to EACH OTHER on
# identical subjects. The data noise is common to both sides of the
# correlation and cancels; what remains is implementation difference alone.
#
# Expected: >= 0.999 in every cell. That is the documented reference value.
#
# SCALE NOTE — the colour ramp is deliberately clipped to [0.99, 1.00].
# On a full 0-1 ramp every cell would be the same colour and the figure would
# convey nothing. Clipping makes the meaningful variation visible, at the cost
# that the ramp exaggerates tiny differences. The clip range is stated on the
# colourbar so this cannot mislead.
#
# Note also that VBA agreement is expected to be very slightly below the
# fork-vs-CBM agreement: VBA runs variational Bayes rather than Laplace-MAP,
# so its point estimate is a posterior MEAN, not a MODE. Those coincide only
# for a Gaussian posterior. ~0.9999 rather than exactly 1.0 is correct.
# ===========================================================================

def figure3(rows):
    # One panel per (model, parameter): RL-alpha, RL-beta, POW-rho, POW-beta.
    panels = [(gen, p) for gen in SPEC for p in SPEC[gen]["params"]]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.6, 2.7))

    VMIN = 0.99          # clip floor — see the scale note above
    norm = Normalize(vmin=VMIN, vmax=1.0)
    cmap = plt.get_cmap("YlGnBu")
    short = {"fork_gn": "fork", "cbm_orig": "CBM", "vba": "VBA"}

    for ax, (gen, (pname, ek, _tk, logsc)) in zip(axes, panels):
        # Restrict to subjects EVERY arm fitted successfully. Without this
        # intersection the correlations would be computed over slightly
        # different subject sets and would not be comparable.
        idx = {a: {r["subject"]: r for r in fits_of(rows, gen, a)}
               for a in ARMS}
        subs = sorted(set.intersection(*[set(idx[a]) for a in ARMS]))
        series = {a: [val(idx[a][s], ek, logsc) for s in subs] for a in ARMS}

        M = np.array([[corr(series[a], series[b]) for b in ARMS]
                      for a in ARMS])

        ax.imshow(M, cmap=cmap, norm=norm, aspect="equal")

        for a in range(len(ARMS)):
            for b in range(len(ARMS)):
                v = M[a, b]
                # 4 decimals: at this level of agreement 2 would print
                # "1.00" everywhere and destroy the information.
                ax.text(b, a, "—" if a == b else f"{v:.4f}",
                        ha="center", va="center", fontsize=6.6,
                        color="white" if v > 0.9965 else "#222222")

        ax.set_xticks(range(len(ARMS)))
        ax.set_yticks(range(len(ARMS)))
        ax.set_xticklabels([short[a] for a in ARMS], fontsize=7)
        ax.set_yticklabels([short[a] for a in ARMS], fontsize=7)
        ax.set_title(f"{gen} · {PRETTY[pname]}", pad=6)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)

    for ax, L in zip(axes, "ABCD"):
        panel_letter(ax, L, dx=-0.30, dy=1.10)

    fig.subplots_adjust(wspace=0.42, top=0.70, bottom=0.24)

    cax = fig.add_axes([0.33, 0.045, 0.34, 0.030])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax, orientation="horizontal")
    cb.set_label(f"correlation between arms  (scale clipped to "
                 f"{VMIN:.2f}–1.00)", fontsize=7, labelpad=3)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=2)

    fig.suptitle("Figure 3 · Cross-arm agreement — the regression check",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.text(0.5, 0.94,
             "Same subjects, three implementations. Every off-diagonal cell "
             "should read $\\geq$ 0.999.",
             ha="center", fontsize=7.2, color="#555555")
    save(fig, "fig3_cross_arm_agreement")


# ===========================================================================
# FIGURE 4 — what Gauss-Newton actually changed
# ===========================================================================
#
# QUESTION: the fork replaced the finite-difference Hessian with a
# Gauss-Newton one. Did that move WHERE fits land, or only how confident
# they are?
#
# This distinction matters a great deal:
#
#   If the MAP moved      -> the fork reports different PARAMETERS, and every
#                            downstream parameter claim would need revisiting.
#   If only evidence moved -> the fork reports the same parameters with a
#                            different precision estimate, affecting only
#                            model COMPARISON.
#
# The curvature enters the log-evidence through the Laplace occam factor
# (the -0.5*log|H| term), so a changed H must change the evidence. Whether it
# changes the optimum is not guaranteed either way — the optimizer uses the
# curvature to choose steps, so in principle it could converge somewhere
# else.
#
# Each point is one subject: x = difference in the MAP estimate, y =
# difference in log-evidence, both fork minus original CBM.
#
#   points in a vertical line at x = 0  ->  evidence moved, MAP did not
#   points spread horizontally           ->  the MAP moved too
#
# AXIS CHOICE — this one is worth explaining, because the obvious choice is
# wrong. A symmetric-log x-axis (the first thing I tried) spreads MAP
# differences of 1e-7 across the full panel width. Those differences are pure
# floating-point convergence noise, so the figure ends up looking like the
# MAP moved for every subject while the annotation truthfully reports that it
# moved for none — the axis actively contradicts the message.
#
# So x is LINEAR, scaled to the tolerance below which a difference is not a
# real move. Points then pile onto the zero line, which is the honest
# picture: identical optima, differing evidence. A shaded band marks the
# tolerance, and any subject genuinely outside it is drawn in a warning
# colour so real movement remains visible rather than being hidden by the
# choice of scale.
#
# VBA is absent from this figure by design: it is a comparison of two
# curvature computations WITHIN the CBM lineage, and VBA does neither.
# ===========================================================================

# Below this, a MAP difference is optimizer convergence noise rather than a
# genuine relocation of the optimum.
MAP_TOL = 1e-4

def figure4(rows):
    models = list(SPEC)
    fig, axes = plt.subplots(1, len(models), figsize=(7.2, 3.5))

    for ax, gen, L in zip(np.atleast_1d(axes), models, "AB"):
        pe = SPEC[gen]["params"][0][1]      # the model's own parameter

        # Pair the two arms subject by subject. A paired comparison is
        # essential: subject-to-subject variation dwarfs the effect we are
        # measuring, so an unpaired one would show nothing.
        pair = defaultdict(dict)
        for r in rows:
            if (r["generator"] == gen and r["fitted_model"] == gen
                    and r["arm"] in ("fork_gn", "cbm_orig")):
                pair[r["subject"]][r["arm"]] = r

        dmap, dlme = [], []
        for v in pair.values():
            if len(v) != 2:
                continue                      # a failed CBM fit — skip
            f, c = v["fork_gn"], v["cbm_orig"]
            a, b = val(f, pe, False), val(c, pe, False)
            e1, e2 = f["log_evidence"], c["log_evidence"]
            if not all(np.isfinite(x) for x in (a, b, e1, e2)):
                continue
            dmap.append(a - b)
            dlme.append(e1 - e2)
        dmap, dlme = np.asarray(dmap), np.asarray(dlme)

        moved = np.abs(dmap) > MAP_TOL
        n_map, n_lme = int(moved.sum()), int((np.abs(dlme) > 1e-6).sum())

        # Shaded tolerance band: everything inside it is numerically the
        # same optimum. Drawn first so points sit on top of it.
        ax.axvspan(-MAP_TOL, MAP_TOL, color="#E8ECF2", zorder=0)
        ax.axvline(0, color="#BBBBBB", linewidth=0.7, zorder=1)
        ax.axhline(0, color="#BBBBBB", linewidth=0.7, zorder=1)

        ax.scatter(dmap[~moved], dlme[~moved], s=11,
                   color=ARM_COLOR["fork_gn"], alpha=0.6, linewidths=0,
                   zorder=3)
        # Genuine movers in the warning colour, so a real divergence can
        # never be mistaken for the noise cloud.
        if moved.any():
            ax.scatter(dmap[moved], dlme[moved], s=16, color="#C4622D",
                       alpha=0.85, linewidths=0, zorder=4)

        # Symmetric limits around zero, wide enough to hold the band and any
        # genuine mover with a little air.
        span = max(MAP_TOL * 3, np.abs(dmap).max() * 1.25) if len(dmap) else MAP_TOL * 3
        ax.set_xlim(-span, span)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(6.5)

        ax.set_xlabel(f"$\\Delta$ MAP {PRETTY[SPEC[gen]['params'][0][0]]}"
                      "   (fork $-$ CBM)")
        if L == "A":
            ax.set_ylabel("$\\Delta$ log-evidence  (fork $-$ CBM)")
        ax.set_title(f"{gen} model", pad=6)
        ax.grid(alpha=0.5)
        panel_letter(ax, L, dx=-0.16, dy=1.04)

        # Below the axes rather than inside them: in-panel the box collides
        # with the very outlier points it is describing.
        ax.text(0.0, -0.30,
                f"MAP differs:      {n_map} / {len(dmap)} fits\n"
                f"evidence differs: {n_lme} / {len(dlme)} fits\n"
                f"median $\\Delta$LME: {np.median(dlme):+.3f} nats",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=6.6, family="monospace", color="#333333")

    fig.suptitle("Figure 4 · Gauss-Newton deviation: fork vs original CBM",
                 fontsize=10, fontweight="bold", y=1.06)
    fig.text(0.5, 0.955,
             "One point per subject. Points stacked on $\\Delta$MAP $= 0$ "
             "mean the change moved the evidence, not the optimum. Shaded "
             f"band = $\\pm${MAP_TOL:g} convergence tolerance; any subject "
             "outside it is drawn in orange.",
             ha="center", fontsize=7.2, color="#555555")
    fig.subplots_adjust(wspace=0.30, top=0.78, bottom=0.30)
    save(fig, "fig4_gauss_newton_deviation")


# ===========================================================================
# FIGURE 5 — parsimony effect and discriminability (AUC)
# ===========================================================================
#
# QUESTION: do the three arms make the same MODEL-SELECTION decisions?
#
# Everything so far has been about parameters. But the reason to compute a
# log-evidence at all is to choose between models, and that is where the
# curvature change of figure 4 could actually bite — the evidence differed on
# every single fit.
#
# THE QUANTITY. For each subject we take the evidence gap
#
#       gap = log-evidence(complex model) - log-evidence(simple model)
#
# always signed complex-minus-simple so the direction means the same thing in
# both families. Note which is which differs by family:
#       RL family:  complex = RL2 (3 params), simple = RL  (2 params)
#       POW family: complex = POW (2 params), simple = LIN (1 param)
#
# A positive gap favours the complex model. Because the simpler model is
# nested inside the complex one, the complex model ALWAYS fits at least as
# well in raw likelihood; a good evidence approximation must therefore
# penalise the extra parameter and drive the gap negative when that parameter
# is not earning its keep. That penalty is the "parsimony effect".
#
# HOW AUC IS COMPUTED. We have two groups of subjects:
#
#   POSITIVES  the 120 POW-generated subjects — genuinely need the extra
#              curvature parameter, so their gap SHOULD be high
#   NEGATIVES  the 120 RL-generated subjects — a single learning rate
#              suffices, so their gap SHOULD be low
#
# Rank all 240 gaps together. AUC is then the Mann-Whitney U statistic
# normalised by n_pos * n_neg, using MID-RANKS for ties:
#
#       AUC = (sum of positive ranks - n_pos(n_pos+1)/2) / (n_pos * n_neg)
#
# which equals, exactly:
#
#       P(a random positive has a higher gap than a random negative)
#
# Read it as: 1.0 = perfect separation, 0.5 = chance, below 0.5 = the
# evidence is pointing the wrong way. It is THRESHOLD-FREE — no cutoff has to
# be picked — which is why it is preferable to "% correctly classified" here.
#
# The absolute value is a property of the benchmark's difficulty (the lapse
# rate, the trial count, how far rho sits from 1), NOT a grade for the
# toolboxes. What matters is that all three arms land in the same place: they
# would make the same selection decisions on the same data.
# ===========================================================================

def _gaps(rows, arm, cell):
    """Per-subject evidence gap, always complex minus simple."""
    spec = SPEC[cell]
    rival, rival_is_complex = spec["rival"], spec["rival_is_complex"]

    by_subject = defaultdict(dict)
    for r in rows:
        if r["arm"] == arm and r["cell"] == cell:
            by_subject[r["subject"]][r["fitted_model"]] = r

    out = []
    for d in by_subject.values():
        if cell not in d or rival not in d:
            continue                     # one of the two fits failed
        e_own, e_riv = d[cell]["log_evidence"], d[rival]["log_evidence"]
        if not (np.isfinite(e_own) and np.isfinite(e_riv)):
            continue
        out.append(e_riv - e_own if rival_is_complex else e_own - e_riv)
    return np.asarray(out, float)


def auc_and_roc(pos, neg):
    """AUC by Mann-Whitney U with mid-ranks, plus the ROC curve to draw.

    Implemented directly rather than via sklearn to keep the benchmark
    dependency-free, and because the mid-rank handling of ties is worth
    being explicit about — evidence gaps can tie exactly when two fits
    converge to the same point.
    """
    pos = np.asarray([v for v in pos if np.isfinite(v)], float)
    neg = np.asarray([v for v in neg if np.isfinite(v)], float)
    if not len(pos) or not len(neg):
        return np.nan, np.array([0, 1]), np.array([0, 1])

    v = np.r_[pos, neg]
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]

    # Mid-ranks: average the ranks within each group of tied values.
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), float)
    ranks[order] = np.arange(1, len(v) + 1)
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]

    n1, n0 = len(pos), len(neg)
    a = float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    # ROC by sweeping the threshold from high to low.
    o = np.argsort(-v, kind="mergesort")
    tpr = np.r_[0, np.cumsum(lab[o] == 1) / n1]
    fpr = np.r_[0, np.cumsum(lab[o] == 0) / n0]
    return a, fpr, tpr


def figure5(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # --- A: the gap distributions the AUC is computed from ---------------
    ax = axes[0]
    offsets = {"fork_gn": 0.0, "cbm_orig": 1.0, "vba": 2.0}

    # The axis is symlog (see below), but violinplot's KDE works in linear
    # data space — it would smooth with a kernel of constant linear width,
    # which after the transform renders as a wildly varying visual width.
    # So the density is estimated on the TRANSFORMED values and the result
    # mapped back, matching what the eye actually sees on the axis.
    LT, LS = 5.0, 1.6

    def fwd(v):
        v = np.asarray(v, float)
        s, a = np.sign(v), np.abs(v)
        return np.where(a <= LT, v,
                        s * LT * (1 + LS * np.log10(np.maximum(a, LT) / LT)))

    def inv(u):
        u = np.asarray(u, float)
        s, a = np.sign(u), np.abs(u)
        return np.where(a <= LT, u,
                        s * LT * 10 ** ((a / LT - 1) / LS))

    for arm in ARMS:
        pos = _gaps(rows, arm, "POW")      # need the extra parameter
        neg = _gaps(rows, arm, "RL")       # do not need it
        y = offsets[arm]
        for grp, sign in ((neg, -1), (pos, +1)):
            if not len(grp):
                continue
            # Half-violins facing opposite directions: the two groups the
            # AUC separates, on one shared axis, without overplotting.
            vp = ax.violinplot([fwd(grp)], positions=[y], vert=False,
                               widths=0.85, showextrema=False)
            for body in vp["bodies"]:
                verts = body.get_paths()[0].vertices
                verts[:, 0] = inv(verts[:, 0])      # back to data space
                if sign > 0:
                    verts[:, 1] = np.clip(verts[:, 1], y, np.inf)
                else:
                    verts[:, 1] = np.clip(verts[:, 1], -np.inf, y)
                body.set_facecolor(ARM_COLOR[arm])
                body.set_alpha(0.75 if sign > 0 else 0.30)
                body.set_edgecolor("none")
            ax.scatter([np.median(grp)], [y + sign * 0.06], s=9,
                       color="white", edgecolor=ARM_COLOR[arm],
                       linewidth=0.8, zorder=5)

    # Zero is the decision boundary: right of it the evidence prefers the
    # complex model, left of it the simple one.
    ax.axvline(0, color="#333333", linewidth=0.7, linestyle=(0, (4, 3)),
               zorder=6)
    ax.set_yticks(list(offsets.values()))
    ax.set_yticklabels([ARM_LABEL[a] for a in ARMS])
    for t, a in zip(ax.get_yticklabels(), ARMS):
        t.set_color(ARM_COLOR[a])

    # SCALE — the POW gaps run to about +60 nats while the RL gaps stay
    # inside +/-4. On a linear axis the long right tail flattens every
    # violin into a featureless ribbon and the near-zero region, which is
    # where the actual selection decisions happen, becomes unreadable.
    # Symmetric-log keeps +/-5 linear (so the decision boundary and the RL
    # distribution are shown faithfully) and compresses the tail beyond it.
    ax.set_xscale("symlog", linthresh=5, linscale=1.6)
    ax.set_xticks([-5, 0, 5, 10, 100])
    ax.set_xticklabels(["$-5$", "0", "5", "10", "100"])
    ax.set_xlabel("evidence gap   (complex $-$ simple model, nats)\n"
                  "linear within $\\pm$5, log beyond", labelpad=2)
    ax.set_title("Distribution of the evidence gap", pad=6)
    ax.grid(axis="x", alpha=0.5)
    ax.invert_yaxis()
    panel_letter(ax, "A", dx=-0.30, dy=1.04)

    ax.legend(handles=[
        Patch(facecolor="#777777", alpha=0.75, label="complex-generated "
              "(POW) — gap should be high"),
        Patch(facecolor="#777777", alpha=0.30, label="simple-generated "
              "(RL) — gap should be low")],
        loc="upper center", bbox_to_anchor=(0.5, -0.30), fontsize=6.6)

    # --- B: the ROC curves those distributions imply ---------------------
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color="#BBBBBB", linewidth=0.8,
            linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.62, 0.55, "chance", fontsize=6.5, color="#999999",
            rotation=38, ha="center", va="center")

    for arm in ARMS:
        a, fpr, tpr = auc_and_roc(_gaps(rows, arm, "POW"),
                                  _gaps(rows, arm, "RL"))
        ax.plot(fpr, tpr, color=ARM_COLOR[arm], linewidth=1.4,
                label=f"{ARM_LABEL[arm]}   AUC = {a:.3f}", zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Discriminability of the gap", pad=6)
    ax.grid(alpha=0.5)
    ax.legend(loc="lower right", fontsize=6.6)
    panel_letter(ax, "B", dx=-0.22, dy=1.04)

    fig.suptitle("Figure 5 · Parsimony effect and discriminability",
                 fontsize=10, fontweight="bold", y=1.05)
    fig.text(0.5, 0.965,
             "AUC = P(a complex-generated subject has a larger evidence gap "
             "than a simple-generated one). Agreement between arms is the "
             "result of interest, not the absolute value.",
             ha="center", fontsize=7.2, color="#555555")
    fig.subplots_adjust(wspace=0.42, top=0.80, bottom=0.26)
    save(fig, "fig5_parsimony_auc")


# ===========================================================================
# CAPTIONS
# ===========================================================================

def write_captions(rows):
    """Emit figures.md with a caption per figure and the key numbers inline.

    Captions are generated from the same data as the figures so they can
    never drift out of date — a hand-written caption survives a re-run with
    changed numbers, which is exactly how a report starts lying.
    """
    # Recompute the handful of numbers the captions quote.
    rec, agree, gn, aucs = {}, {}, {}, {}
    for gen in SPEC:
        params = SPEC[gen]["params"]
        sub = fits_of(rows, gen, "fork_gn")
        rec[gen] = {p[0]: corr([val(r, p[1], p[3]) for r in sub],
                               [val(r, p[2], p[3]) for r in sub])
                    for p in params}

        idx = {a: {r["subject"]: r for r in fits_of(rows, gen, a)}
               for a in ARMS}
        subs = sorted(set.intersection(*[set(idx[a]) for a in ARMS]))
        worst = 1.0
        for _, ek, _, lg in params:
            s = {a: [val(idx[a][x], ek, lg) for x in subs] for a in ARMS}
            for ia, a in enumerate(ARMS):
                for b in ARMS[ia + 1:]:
                    c = corr(s[a], s[b])
                    if np.isfinite(c):
                        worst = min(worst, c)
        agree[gen] = worst

        pe = params[0][1]
        pair = defaultdict(dict)
        for r in rows:
            if (r["generator"] == gen and r["fitted_model"] == gen
                    and r["arm"] in ("fork_gn", "cbm_orig")):
                pair[r["subject"]][r["arm"]] = r
        dm, dl = [], []
        for v in pair.values():
            if len(v) != 2:
                continue
            f, c = v["fork_gn"], v["cbm_orig"]
            a_, b_ = val(f, pe, False), val(c, pe, False)
            if all(np.isfinite(x) for x in (a_, b_, f["log_evidence"],
                                            c["log_evidence"])):
                dm.append(abs(a_ - b_))
                dl.append(f["log_evidence"] - c["log_evidence"])
        gn[gen] = (int((np.asarray(dm) > 1e-4).sum()),
                   int((np.abs(np.asarray(dl)) > 1e-6).sum()), len(dm))

    for arm in ARMS:
        aucs[arm] = auc_and_roc(_gaps(rows, arm, "POW"),
                                _gaps(rows, arm, "RL"))[0]

    meta = np.load(BENCH / "data" / GRID / "RL.npz")
    n_sub, n_tr = int(meta["n_subjects"]), int(meta["n_trials"])
    lapse = float(meta["lapse"])

    fail = {a: sum(1 for gen in SPEC
                   for r in fits_of(rows, gen, a)
                   if not np.isfinite(val(r, SPEC[gen]["params"][0][1], False)))
            for a in ARMS}

    md = f"""# Benchmark figures — `{GRID}` grid

Three implementations fitted to identical simulated data: **this fork**
(Gauss-Newton curvature), **pristine pre-fork CBM** (commit `e72193f`), and the
**MATLAB VBA toolbox**. Two generating models, {n_sub} subjects x {n_tr} trials
each, {lapse:.0%} lapse trials.

Failed fits — fork {fail['fork_gn']}, CBM {fail['cbm_orig']}, VBA {fail['vba']}.
(The CBM failure is a known upstream bug; see RERUN.md.)

---

**Figure 1 · Parameter estimate error.** Distribution of estimated minus
predefined parameter values across subjects, for each generating model (rows)
and each implementation (columns). Violins show the full distribution, boxes the
median and interquartile range, points individual subjects. The dashed line at
zero marks perfect recovery. Beta is on a log scale. A displaced median
indicates bias — mild shrinkage toward the prior mean is expected under MAP
estimation and is not an error. *Reading: the three arms should be
indistinguishable.*

**Figure 2 · Parameter recovery.** Pearson correlation between estimated and
predefined parameters across subjects. Diagonal cells measure recovery of each
parameter; off-diagonal cells measure trade-off between parameters, where a
large value would indicate the two are not separately identifiable. Fork
diagonals: {', '.join(f'{g} {k} = {v:.2f}'
                      for g, d in rec.items() for k, v in d.items())}. The
attainable maximum is set by the variance of the true parameters
(r ~ SD_true / sqrt(SD_true^2 + SD_error^2)), so these values are comparable
only across runs using identical simulation ranges.

**Figure 3 · Cross-arm agreement.** Correlation between the estimates produced
by each pair of implementations, over the same subjects. Because both sides of
each correlation carry the same data noise, it cancels, leaving only
implementation difference — making this the most sensitive test of whether the
fork has diverged. The colour scale is clipped to 0.99–1.00; on a full 0–1 scale
all cells would appear identical. Lowest observed agreement:
{', '.join(f'{g} {v:.4f}' for g, v in agree.items())}. *This is the panel to
check after any change to `cbm/`.*

**Figure 4 · Gauss-Newton deviation.** Per-subject difference between this fork
and pristine CBM in the MAP estimate (x, symmetric-log) and in the log-evidence
(y). {'; '.join(f'For {g}, the MAP differs on {a} of {n} fits and the evidence on {b} of {n}' for g, (a, b, n) in gn.items())}. The change
is therefore confined to the evidence — it alters how confident a fit is, not
where it lands — which restricts its consequences to model comparison.

**Figure 5 · Parsimony effect and discriminability.** (A) Distribution of the
per-subject evidence gap, always signed complex-model minus simple-model (RL2 −
RL; POW − LIN). Filled violins are subjects generated by the complex model,
which should show a high gap; pale violins are subjects generated by the simple
model, which should show a low one. (B) ROC curves obtained by sweeping a
threshold over that gap. AUC is the Mann-Whitney U statistic with mid-ranks,
normalised by the product of group sizes, and equals the probability that a
randomly chosen complex-generated subject has a larger gap than a randomly
chosen simple-generated one; 0.5 is chance. AUC =
{', '.join(f'{ARM_LABEL[a]} {v:.3f}' for a, v in aucs.items())}. The absolute
value reflects the difficulty of the benchmark rather than the quality of any
implementation; the agreement between arms is the result of interest.
"""
    (FIGDIR / "figures.md").write_text(md)
    print("  wrote figures.md")


# ===========================================================================

def main():
    set_style()
    print(f"loading {GRID} grid …")
    rows = load_rows()
    print(f"  {len(rows)} fits across {len(set(r['arm'] for r in rows))} arms")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    figure1(rows)
    figure2(rows)
    figure3(rows)
    figure4(rows)
    figure5(rows)
    write_captions(rows)
    print(f"\nall output in {FIGDIR}")


if __name__ == "__main__":
    main()
