"""
Diagnostic figures for a fit (MODIFICATION 14).

WHAT THIS IS
------------
`Config.display = True` makes the optimizer retain what a diagnostic plot
needs — the L-BFGS-B evaluation path, the Newton-polish trace, and the
warnings raised per subject. `FitResult.plot()` then draws it. Nothing here
runs during fitting, and with `display=False` (the default) nothing is
retained, so the cost to every existing caller is exactly zero.

Maps the intent of VBA's display options (`options.DisplayWin`,
`VBA_initDisplay`) onto this toolbox's structures. It is not a port: VBA
runs its own Gauss-Newton loop and can show a clean per-iteration trace,
whereas here the bulk of the work is inside scipy's L-BFGS-B.

TWO FIGURES, BECAUSE THE QUESTION DIFFERS
-----------------------------------------
ONE SUBJECT   How did this fit go? Trajectories, the objective climbing,
              observed vs predicted, and the status/warnings that apply.

MANY SUBJECTS What does the population look like? Parameter distributions,
              evidence spread, quality counts, timing. Trajectories are
              deliberately absent — twenty overlaid zigzags say nothing,
              and per-subject detail is what `plot(subject=i)` is for.

THREE HONESTY NOTES, ALL VISIBLE IN THE FIGURES
-----------------------------------------------
1. `search_path` is function EVALUATIONS, including line-search probes —
   not clean iterations. The path zigzags. Axes say "function evaluations"
   so it cannot be misread as an iteration count.

2. A per-step LOG-EVIDENCE exists only for the Newton-polish steps. The
   Laplace evidence needs |H|, and the polish loop is the only place H is
   recomputed each step; during L-BFGS-B there is no Hessian at all. So
   the objective panel plots the log-JOINT over the search and the
   log-EVIDENCE over the polish, as two segments, never as one curve.

3. Without `predict=`/`observed=` there is no way to compute a residual —
   `data` is opaque to the toolbox. The panel then shows per-trial
   log-likelihood instead and says so in its title (individual_fit also
   warns at fit time).

REFERENCE  DEV.md §17.
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np

# Warnings that say nothing about the fit. scipy emits one per L-BFGS-B
# call about `disp`/`iprint`; showing 3 identical copies per subject in a
# status panel would bury any real alert.
_NOISE_WARNINGS = ("`disp` and `iprint` options of the L-BFGS-B",)

# ── Greyscale palette (requested 2026-08-14) ──────────────────────────
# Both figures are greyscale. Ordered light to dark and monotone in
# lightness, so every distinction survives black-and-white printing,
# photocopying, and any form of colour vision deficiency.
_INK_LIGHT, _INK, _INK_DARK = "#C9C9CE", "#7A7A80", "#3A3A3E"
_GREY = "#888888"          # secondary text and reference lines

# WHERE HUE WAS DOING REAL WORK, IT IS REPLACED — NOT DROPPED.
#
# Panel B of the subject figure overlays d parameter traces on ONE axis;
# colour was the only thing telling them apart.
#
# Identity is carried by a GREY RAMP, light to dark, spaced evenly for
# whatever d happens to be. A ramp is continuous, so unlike a fixed list
# of line styles it never runs out: with 4 styles, theta_0 and theta_4
# would be drawn identically at d=5.
#
# Line style is kept as a REDUNDANT second channel for the first four
# traces. The two reinforce each other — shade alone gets hard to read
# past ~6 parameters (adjacent steps fall below ~0.10 in lightness), and
# style alone collides at 5. Together they stay separable further than
# either would on its own, and the legend is always present as the final
# fallback.
_LINE_STYLES = ["-", "--", "-.", ":"]
_RAMP_LIGHT, _RAMP_DARK = 0.70, 0.12


def _grey_ramp(n: int) -> List[str]:
    """n greys from light to dark, as matplotlib grey-level strings.

    Evenly spaced in lightness so no two adjacent traces sit closer than
    they have to for the given n.
    """
    if n <= 1:
        return [str(_RAMP_DARK)]
    step = (_RAMP_DARK - _RAMP_LIGHT) / (n - 1)
    return [f"{_RAMP_LIGHT + step * j:.3f}" for j in range(n)]

# The status strip's alerts stay visually distinct too. Colour is not
# load-bearing there — every alert line is already prefixed "!" and set
# in bold — so the strongest ink is enough to make it stand out.
_ALERT = _INK_DARK


def _mpl():
    """Import matplotlib lazily.

    Deliberately not a module-level import: matplotlib is an optional
    extra, and `import cbm` must work without it. Only calling plot()
    requires it.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "plotting requires matplotlib, which is an optional extra of "
            "this toolbox. Install it with `pip install matplotlib` "
            "(or `pip install .[display]`)."
        ) from e
    return plt


def _show(plt):
    """Open the figure window.

    Non-blocking when possible. A plain `plt.show()` blocks until the
    window is closed, so a script drawing two figures shows the second
    only after you dismiss the first — which reads like the second one
    never appeared. `block=False` draws both, then the script's own
    `plt.show()` (or the interactive prompt) keeps them alive.

    On a non-interactive backend ("Agg", i.e. headless, CI, SSH without
    X) no window is possible; say so once rather than failing silently,
    because "nothing popped up" is otherwise indistinguishable from a
    broken plot.
    """
    import matplotlib
    backend = matplotlib.get_backend()
    if backend.lower() == "agg":
        warnings.warn(
            f"show=True but the matplotlib backend is {backend!r}, which "
            f"cannot open a window (headless session, or something called "
            f"matplotlib.use('Agg')). The figure was still created and "
            f"save=... works. Use an interactive backend for a window.",
            UserWarning, stacklevel=3)
        return
    try:
        plt.show(block=False)
        plt.pause(0.1)          # let the GUI event loop actually paint it
    except Exception:
        plt.show()              # any backend that dislikes block=False


def _style(plt):
    return {
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 8,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "axes.titlesize": 8.5, "axes.labelsize": 8, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.fontsize": 7, "legend.frameon": False,
        "grid.linewidth": 0.4, "grid.color": "#DDDDDD",
    }


def _panel(ax, letter, title=None, dx=-0.13, dy=1.03):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="bottom", ha="left")
    if title:
        ax.set_title(title, pad=5)


def _require_display(result):
    dd = getattr(result, "_display_data", None)
    if dd is None:
        raise ValueError(
            "this result carries no display data. Re-fit with "
            "config=dict(display=True) — the optimizer only retains the "
            "trajectories, traces and warnings that plot() needs when "
            "display is enabled (Mod 14, DEV.md §17).")
    return dd


def _real_warnings(diag) -> List[str]:
    """Per-subject warnings with the known-noise ones filtered out."""
    ws = getattr(diag, "warnings", None) or []
    out, seen = [], set()
    for w in ws:
        if any(n in w for n in _NOISE_WARNINGS):
            continue
        if w not in seen:            # collapse duplicates across restarts
            seen.add(w)
            out.append(w)
    return out


def _flag_lines(result, i: int) -> List[tuple]:
    """(text, is_alert) status lines for subject i."""
    diag = None
    dl = getattr(result.math, "diagnostics", None)
    if dl is not None and i < len(dl):
        diag = dl[i]
    flag = float(np.ravel(result.math.flag)[i])

    lines = []
    if diag is None:
        lines.append(("fit FAILED — prior substituted "
                      "(config.prior_for_failed)", True))
        return lines

    status = getattr(diag, "convergence_status", None)
    lines.append((f"convergence: {status}",
                  status not in ("converged_df", "no_improvement")))
    lines.append((f"curvature: {getattr(diag, 'hess_method', '?')}", False))
    lines.append((f"|gradient|: {getattr(diag, 'abs_grad', float('nan')):.2e}",
                  False))

    nia = getattr(diag, "n_inits_agreeing", None)
    nr = getattr(diag, "n_runs", None)
    if nia is not None:
        lines.append((f"inits agreeing: {nia}/{nr}", nia == 0))

    wi = getattr(diag, "weak_identifiability", None)
    if wi is not None:
        lines.append((f"identifiability: {wi:.2f}x prior precision",
                      wi < 2.0))

    nclip = getattr(diag, "hess_n_clipped", None)
    if nclip:
        lines.append((f"clipped eigenvalues: {nclip}", True))

    ahb = getattr(diag, "at_hard_bounds", None)
    if ahb is not None and np.any(ahb):
        idx = [j for j, b in enumerate(np.ravel(ahb)) if b]
        lines.append((f"at hard bounds: theta{idx}", True))

    if flag == 0.5:
        lines.append(("flag 0.5 — singular Hessian at the optimum", True))
    return lines


# ===========================================================================
# SINGLE SUBJECT
# ===========================================================================

def plot_subject(result, subject: int = 0, figsize=(9.0, 7.2),
                 save: Optional[str] = None, show: bool = False):
    """Diagnostic figure for one subject.

    A  observed vs predicted (or per-trial log-likelihood — see note 3)
    B  parameter trajectories over the search, with the final +/-1 SE band
    C  objective evolution: log-joint over the search, log-evidence over
       the polish (note 2)
    D  per-trial log-likelihood at the final estimate
    E  status, flags and warnings — full width across the bottom
    """
    plt = _mpl()
    dd = _require_display(result)
    from .reporting import standard_errors

    with plt.rc_context(_style(plt)):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.88],
                              hspace=0.55, wspace=0.28,
                              top=0.90, bottom=0.04,
                              left=0.09, right=0.97)
        axA = fig.add_subplot(gs[0, 0])
        axB = fig.add_subplot(gs[0, 1])
        axC = fig.add_subplot(gs[1, 0])
        axD = fig.add_subplot(gs[1, 1])
        axE = fig.add_subplot(gs[2, :])

        theta = np.atleast_2d(np.asarray(result.output.parameters,
                                         dtype=float))[subject]
        se = standard_errors(result)[subject]
        d = theta.size
        dl = getattr(result.math, "diagnostics", None)
        diag = dl[subject] if (dl is not None and subject < len(dl)) else None
        data_i = dd["data"][subject]
        predict, observed = dd["predict"], dd["observed"]
        mtrials = dd["model_trials"]

        # ---- A: observed vs predicted, or the fallback ------------------
        did_scatter = False
        if predict is not None and observed is not None:
            try:
                y = np.ravel(np.asarray(observed(data_i), dtype=float))
                yh = np.ravel(np.asarray(predict(theta, data_i),
                                         dtype=float))
                ok = np.isfinite(y) & np.isfinite(yh)
                y, yh = y[ok], yh[ok]
                if y.size:
                    lo = float(min(y.min(), yh.min()))
                    hi = float(max(y.max(), yh.max()))
                    pad = 0.05 * (hi - lo or 1.0)
                    axA.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                             color=_GREY, lw=0.9, ls=(0, (4, 3)), zorder=1)
                    axA.scatter(yh, y, s=9, color=_INK, alpha=0.6,
                                linewidths=0, zorder=3)
                    ss = float(np.sum((y - yh) ** 2))
                    st = float(np.sum((y - y.mean()) ** 2))
                    r2 = 1.0 - ss / st if st > 0 else np.nan
                    axA.set_xlabel("predicted")
                    axA.set_ylabel("observed")
                    axA.text(0.04, 0.95,
                             f"$R^2$ = {r2:.3f}\nRMSE = "
                             f"{np.sqrt(ss / y.size):.4g}",
                             transform=axA.transAxes, va="top", fontsize=7,
                             color="#333333")
                    axA.set_xlim(lo - pad, hi + pad)
                    axA.set_ylim(lo - pad, hi + pad)
                    axA.set_aspect("equal", adjustable="box")
                    _panel(axA, "A", "observed vs predicted")
                    did_scatter = True
            except Exception as e:
                # A user-supplied callable that raises must not lose the
                # whole figure; fall through to the per-trial panel.
                axA.text(0.5, 0.5, f"predict/observed raised:\n"
                                   f"{type(e).__name__}: {e}",
                         transform=axA.transAxes, ha="center", va="center",
                         fontsize=7, color=_INK_DARK, wrap=True)
                _panel(axA, "A", "observed vs predicted — failed")
                did_scatter = True

        if not did_scatter:
            ll = None
            if mtrials is not None:
                try:
                    ll = np.ravel(np.asarray(mtrials(theta, data_i),
                                             dtype=float))
                except Exception:
                    ll = None
            if ll is not None and ll.size:
                axA.plot(np.arange(ll.size), ll, lw=0.8, color=_INK,
                         alpha=0.85)
                axA.axhline(float(np.mean(ll)), color=_INK_DARK, lw=0.9,
                            ls=(0, (4, 3)),
                            label=f"mean {np.mean(ll):.3f}")
                axA.set_xlabel("trial")
                axA.set_ylabel("log-likelihood")
                axA.legend(loc="lower right")
                _panel(axA, "A", "per-trial fit  (no predict/observed)")
                axA.text(0.02, 0.04,
                         "pass predict= and observed= for\n"
                         "observed-vs-predicted instead",
                         transform=axA.transAxes, fontsize=6.2,
                         color=_GREY, va="bottom")
            else:
                axA.text(0.5, 0.5, "no predict/observed and no\n"
                                   "model_trials — nothing to show",
                         transform=axA.transAxes, ha="center", va="center",
                         fontsize=7.5, color=_GREY)
                _panel(axA, "A", "fit quality — unavailable")

        # ---- B: parameter trajectories ---------------------------------
        sp = getattr(diag, "search_path", None) if diag is not None else None
        if sp is not None and len(sp):
            sp = np.atleast_2d(np.asarray(sp, dtype=float))
            x = np.arange(sp.shape[0])
            n_tr = min(d, sp.shape[1])
            shades = _grey_ramp(n_tr)
            for j in range(n_tr):
                axB.plot(x, sp[:, j], lw=1.0, color=shades[j], alpha=0.95,
                         linestyle=_LINE_STYLES[j % len(_LINE_STYLES)],
                         label=f"$\\theta_{{{j}}}$")
                if np.isfinite(se[j]):
                    axB.axhspan(theta[j] - se[j], theta[j] + se[j],
                                color=shades[j], alpha=0.15, lw=0)
            axB.set_xlabel("function evaluations")
            axB.set_ylabel(r"$\theta$")
            # Pinned upper-right rather than "best": with many traces
            # matplotlib's "best" lands on top of either the clip note
            # (bottom-right) or the title. One row per two entries keeps
            # it compact.
            axB.legend(loc="upper right", ncol=max(1, (d + 1) // 2),
                       fontsize=6.4, columnspacing=1.0,
                       handlelength=1.7, labelspacing=0.3,
                       borderaxespad=0.3)
            axB.grid(alpha=0.6)
            # Early line-search probes can shoot far out (L-BFGS-B tests
            # wide steps). Left unclipped, one -9.5 excursion flattens the
            # region where the fit actually converges. Clip to a robust
            # range and mark that it is clipped rather than hide it.
            allv = sp[:, :d][np.isfinite(sp[:, :d])]
            if allv.size:
                lo_q, hi_q = np.percentile(allv, [2, 98])
                pad = 0.20 * (hi_q - lo_q or 1.0)
                lo_c, hi_c = lo_q - pad, hi_q + pad
                if allv.min() < lo_c or allv.max() > hi_c:
                    # Extra headroom at the top for the legend.
                    axB.set_ylim(lo_c, hi_c + 0.28 * (hi_c - lo_c))
                    axB.text(0.98, 0.03,
                             "y clipped to 2-98% of the path",
                             transform=axB.transAxes, ha="right",
                             va="bottom", fontsize=5.8, color=_GREY)
            _panel(axB, "B", "parameter path")
        else:
            axB.text(0.5, 0.5, "no trajectory retained", ha="center",
                     va="center", transform=axB.transAxes, fontsize=7.5,
                     color=_GREY)
            _panel(axB, "B", "parameter path")

        # ---- C: objective evolution ------------------------------------
        # See note 2: two different quantities, plotted as two segments.
        sf = getattr(diag, "search_f", None) if diag is not None else None
        plme = getattr(diag, "polish_lme", None) if diag is not None else None
        drew = False
        if sf is not None and len(sf):
            sf = np.asarray(sf, dtype=float)
            axC.plot(np.arange(sf.size), np.maximum.accumulate(sf),
                     lw=1.1, color=_INK_DARK, label="log-joint (best so far)")
            axC.plot(np.arange(sf.size), sf, lw=0.5, color=_INK,
                     alpha=0.35, label="log-joint (each evaluation)")
            axC.set_xlabel("function evaluations")
            axC.set_ylabel("log-joint")
            drew = True
        # The polish is typically 2-3 steps against tens of evaluations.
        # Sharing the x-axis would stretch those few points across the
        # whole panel and make a 1e-10 change look like a sweep — the
        # exact misreading note 2 warns about. So the evidence goes in an
        # inset with its OWN x-axis, and its total change is stated in
        # words rather than left to the eye.
        if plme is not None and len(plme) and np.isfinite(plme).any():
            pl = np.asarray(plme, dtype=float)
            fin = pl[np.isfinite(pl)]
            span = float(fin.max() - fin.min()) if fin.size > 1 else 0.0
            if drew:
                # Square, upper-right. The previous placement put the
                # inset's x-label on top of the host axis's own x-axis.
                ins = axC.inset_axes([0.62, 0.44, 0.33, 0.33])
            else:
                ins = axC
            # Plot the CHANGE from the first step, not the absolute value.
            # Absolute log-evidence over 2-3 near-identical steps forces
            # matplotlib into an offset label like "1e-6 - 1.0496e2",
            # which is unreadable and buries the one number that matters:
            # how much the evidence actually moved.
            ins.plot(np.arange(pl.size), pl - fin[0], lw=1.2, marker="o",
                     ms=3.0, color=_INK_DARK)
            ins.set_title("log-evidence", fontsize=6.2, pad=2,
                          color=_INK_DARK)
            ins.tick_params(labelsize=5.5, length=1.8, pad=1)
            ins.set_xlabel("polish step", fontsize=5.8, labelpad=1)
            # Polish steps are counts: 0.5 of a step does not exist.
            ins.xaxis.get_major_locator().set_params(integer=True)
            # Baseline on the y-label rather than a title that overflows
            # the panel; keeps the inset square and self-explanatory.
            ins.set_ylabel(f"$\\Delta$ from {fin[0]:.2f}", fontsize=5.3,
                           labelpad=1)
            ins.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            ins.yaxis.get_offset_text().set_fontsize(5.0)
            for sp_ in ins.spines.values():
                sp_.set_linewidth(0.5)
            # A flat line is the normal, healthy case; say so instead of
            # letting matplotlib autoscale noise into a dramatic curve.
            if span < 1e-6:
                ins.set_ylim(-1.0, 1.0)
                ins.text(0.5, 0.80, f"flat ({span:.1e})", fontsize=5.5,
                         color=_GREY, ha="center", transform=ins.transAxes)
            drew = True
        if drew:
            axC.grid(alpha=0.6)
            _panel(axC, "C", "objective evolution")

        else:
            axC.text(0.5, 0.5, "no trace retained", ha="center",
                     va="center", transform=axC.transAxes, fontsize=7.5,
                     color=_GREY)
            _panel(axC, "C", "objective evolution")

        # ---- D: parameter estimates with CI ----------------------------
        yy = np.arange(d)
        axD.errorbar(theta, yy, xerr=1.96 * se, fmt="o", ms=4.5,
                     color=_INK_DARK, ecolor=_INK_DARK, elinewidth=1.1,
                     capsize=2.5)
        axD.axvline(0, color=_GREY, lw=0.8, ls=(0, (4, 3)))
        axD.set_yticks(yy)
        axD.set_yticklabels([f"$\\theta_{{{j}}}$" for j in range(d)])
        axD.invert_yaxis()
        axD.set_xlabel(r"estimate  $\pm$ 95% CI")
        axD.grid(axis="x", alpha=0.6)
        _panel(axD, "D", "estimates")
        # Numeric values are in fit.summary() and fit.table(); repeating
        # them on the markers only crowded the panel.
        axD.set_ylim(d - 0.5, -0.5)
        x0, x1 = axD.get_xlim()
        axD.set_xlim(x0 - 0.06 * (x1 - x0), x1 + 0.06 * (x1 - x0))

        # ---- E: status, flags, warnings (full width) --------------------
        axE.axis("off")
        lines = _flag_lines(result, subject)
        wmsgs = _real_warnings(diag)
        prof = getattr(result, "profile", None)
        nit = getattr(diag, "n_runs", None) if diag is not None else None

        axE.text(0.0, 1.0, "Status", transform=axE.transAxes,
                 fontweight="bold", va="top", fontsize=8.5)
        yv = 0.84
        for txt, alert in lines:
            axE.text(0.0, yv, ("!  " if alert else "   ") + txt,
                     transform=axE.transAxes, va="top", fontsize=7.2,
                     family="monospace",
                     color=_ALERT if alert else "#555555",
                     fontweight="bold" if alert else "normal")
            yv -= 0.125

        axE.text(0.42, 1.0, "Cost", transform=axE.transAxes,
                 fontweight="bold", va="top", fontsize=8.5)
        el = getattr(prof, "telapsed", None)
        cost = []
        if nit is not None:
            cost.append(f"initializations: {nit}")
        if sf is not None:
            cost.append(f"evaluations: {len(sf)}")
        if plme is not None:
            cost.append(f"polish steps: {len(plme)}")
        if el is not None:
            n_sub = np.atleast_2d(result.output.parameters).shape[0]
            cost.append(f"elapsed (all {n_sub}): {el:.2f}s")
        for k, txt in enumerate(cost):
            axE.text(0.42, 0.84 - k * 0.125, "   " + txt,
                     transform=axE.transAxes, va="top", fontsize=7.2,
                     family="monospace", color="#333333")

        axE.text(0.70, 1.0, "Warnings", transform=axE.transAxes,
                 fontweight="bold", va="top", fontsize=8.5)
        if wmsgs:
            for k, w in enumerate(wmsgs[:4]):
                axE.text(0.70, 0.84 - k * 0.165,
                         "!  " + (w[:78] + ("…" if len(w) > 78 else "")),
                         transform=axE.transAxes, va="top", fontsize=6.4,
                         family="monospace", color=_ALERT, wrap=True)
            if len(wmsgs) > 4:
                axE.text(0.70, 0.84 - 4 * 0.165,
                         f"   … {len(wmsgs) - 4} more",
                         transform=axE.transAxes, va="top", fontsize=6.4,
                         family="monospace", color=_GREY)
        else:
            axE.text(0.70, 0.84, "   none", transform=axE.transAxes,
                     va="top", fontsize=7.2, family="monospace",
                     color="#666666")

        # ---- prior, across the bottom of the strip (Mod 15) -------------
        # The prior is part of how the estimate was produced — a figure
        # showing a posterior without it is missing half the recipe. Kept
        # on its own row so it reads as context for the whole fit rather
        # than as another status line.
        from .reporting import prior_spec
        pm_, pv_, pdef_ = prior_spec(result)
        if pm_.size:
            uniform = (np.allclose(pm_, pm_[0])
                       and np.isfinite(pv_).all()
                       and np.allclose(pv_, pv_[0]))
            if uniform:
                spec = (f"N({pm_[0]:g}, {pv_[0]:g}) on all "
                        f"{pm_.size} parameters")
            else:
                spec = "   ".join(
                    f"$\\theta_{{{j}}}$ N({pm_[j]:g}, {pv_[j]:g})"
                    for j in range(pm_.size))
            axE.text(0.0, 0.02, "Prior", transform=axE.transAxes,
                     fontweight="bold", va="bottom", fontsize=8.5)
            axE.text(0.115, 0.02, spec, transform=axE.transAxes,
                     va="bottom", fontsize=7.2, family="monospace",
                     color="#333333")
            if pdef_:
                axE.text(0.115, 0.02, "\n" + " " * 2
                         + "toolbox default (" + ", ".join(pdef_) + ") — "
                         "weakly informative, not neutral",
                         transform=axE.transAxes, va="top", fontsize=6.3,
                         color=_GREY)

        model = getattr(getattr(result, "input", None), "model_name", "?")
        fig.suptitle(f"Fit diagnostics · {model} · subject {subject}",
                     fontsize=10.5, fontweight="bold", y=0.975)

        if save:
            fig.savefig(save, bbox_inches="tight")
        if show:
            _show(plt)
        return fig


# ===========================================================================
# MANY SUBJECTS
# ===========================================================================

def plot_group(result, figsize=(9.0, 6.2), save: Optional[str] = None,
               show: bool = False):
    """Population-level figure: no trajectories, by design.

    A  parameter distributions across subjects (violin + points)
    B  log-evidence distribution
    C  fit-quality counts
    D  cost: evaluations per subject, and elapsed time
    """
    plt = _mpl()
    # Checked here too, not only in plot_subject: the group figure needs
    # the retained traces for panel D, and without display the message
    # below is far more useful than an empty panel or a later TypeError.
    _require_display(result)
    from .reporting import standard_errors, rows

    with plt.rc_context(_style(plt)):
        theta = np.atleast_2d(np.asarray(result.output.parameters,
                                         dtype=float))
        n, d = theta.shape
        se = standard_errors(result)
        lme = np.ravel(np.asarray(result.output.log_evidence, dtype=float))
        recs = rows(result)
        dl = getattr(result.math, "diagnostics", None) or [None] * n

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.subplots_adjust(hspace=0.55, wspace=0.30, top=0.87,
                            bottom=0.10, left=0.09, right=0.97)
        axA, axB, axC, axD = axes.ravel()

        # ---- A: parameter distributions --------------------------------
        # Horizontal: parameters on y, estimate on x. Reads the same way
        # as panel D of the subject figure, and parameter labels stay
        # upright however many there are.
        pos = np.arange(d)
        cell = [theta[np.isfinite(theta[:, j]), j] for j in range(d)]
        keep = [k for k, c in enumerate(cell) if c.size > 1]
        if keep:
            vp = axA.violinplot([cell[k] for k in keep],
                                positions=[pos[k] for k in keep],
                                widths=0.6, vert=False, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(_INK_LIGHT)
                b.set_alpha(0.35)
                b.set_edgecolor("none")
        rng = np.random.default_rng(0)      # fixed: jitter must be stable
        for j in range(d):
            c = cell[j]
            if not c.size:
                continue
            axA.scatter(c, pos[j] + rng.uniform(-0.16, 0.16, c.size),
                        s=6, color=_INK, alpha=0.55, linewidths=0)
            axA.vlines(np.median(c), pos[j] - 0.22, pos[j] + 0.22,
                       color="white", lw=1.8, zorder=5)
            axA.vlines(np.median(c), pos[j] - 0.22, pos[j] + 0.22,
                       color=_INK_DARK, lw=1.0, zorder=6)
        axA.set_yticks(pos)
        axA.set_yticklabels([f"$\\theta_{{{j}}}$" for j in range(d)])
        axA.set_ylim(d - 0.5, -0.5)
        axA.set_xlabel("estimate")
        axA.grid(axis="x", alpha=0.6)
        _panel(axA, "A", f"parameter estimate (n={n})")

        # ---- B: evidence ----------------------------------------------
        fin = lme[np.isfinite(lme)]
        if fin.size:
            axB.hist(fin, bins=min(20, max(5, fin.size // 2)),
                     color=_INK, alpha=0.75, edgecolor="white", lw=0.5)
        axB.set_xlabel("log-evidence")
        axB.set_ylabel("subjects")
        axB.grid(axis="y", alpha=0.6)
        # Counts are integers; matplotlib's default 0.5 steps on a small
        # histogram imply half a subject.
        axB.yaxis.get_major_locator().set_params(integer=True)
        _panel(axB, "B", "log-evidence")

        # ---- C: quality counts ----------------------------------------
        counts: Dict[str, int] = {}
        for r in recs:
            q = r["quality"]
            if q == "-":
                continue
            if q == "ok":
                counts["ok"] = counts.get("ok", 0) + 1
            else:
                for part in q.split(","):
                    counts[part] = counts.get(part, 0) + 1
        if counts:
            order = ["ok"] + sorted(k for k in counts if k != "ok")
            order = [k for k in order if k in counts]
            vals = [counts[k] for k in order]
            # "ok" light, anything flagged dark: the distinction survives
            # greyscale because it is carried by lightness, not hue.
            colr = [_INK_LIGHT if k == "ok" else _INK_DARK for k in order]
            axC.barh(np.arange(len(order)), vals, height=0.62,
                     color=colr, edgecolor="white", lw=0.6)
            axC.set_yticks(np.arange(len(order)))
            axC.set_yticklabels(order)
            # Pad the category axis so a single bar occupies a sensible
            # slice of the panel instead of filling it edge to edge. The
            # padding shrinks as categories accumulate.
            pad_y = max(0.5, (3 - len(order)) * 0.9)
            axC.set_ylim(len(order) - 1 + pad_y, -pad_y)
            for k, v in enumerate(vals):
                axC.text(v, k, f" {v}", va="center", fontsize=7,
                         color="#444444")
            axC.set_xlim(0, max(vals) * 1.18)
        else:
            axC.text(0.5, 0.5, "no diagnostics available", ha="center",
                     va="center", transform=axC.transAxes, fontsize=7.5,
                     color=_GREY)
        axC.set_xlabel("subjects")
        axC.grid(axis="x", alpha=0.6)
        _panel(axC, "C", "fit quality")

        # ---- D: cost ---------------------------------------------------
        # `or []` would be a bug here: search_f is an ndarray, and an
        # array with >1 element has no truth value.
        nev = []
        for x in dl:
            sf = getattr(x, "search_f", None) if x is not None else None
            if sf is not None and len(sf):
                nev.append(len(sf))
        if nev:
            axD.hist(nev, bins=min(15, max(4, len(nev) // 2)),
                     color=_INK, alpha=0.8, edgecolor="white", lw=0.5)
            axD.set_xlabel("function evaluations per subject")
            axD.set_ylabel("subjects")
            axD.grid(axis="y", alpha=0.6)
            axD.yaxis.get_major_locator().set_params(integer=True)
        else:
            axD.text(0.5, 0.5, "no evaluation counts retained",
                     ha="center", va="center", transform=axD.transAxes,
                     fontsize=7.5, color=_GREY)
        _panel(axD, "D", "computational cost")
        model = getattr(getattr(result, "input", None), "model_name", "?")
        fig.suptitle(f"Fit diagnostics · {model} · {n} subjects",
                     fontsize=10.5, fontweight="bold", y=0.965)
        fig.text(0.5, 0.915,
                 "Population view. For one subject's trajectories and "
                 "warnings use fit.plot(subject=i).",
                 ha="center", fontsize=7, color="#666666")

        if save:
            fig.savefig(save, bbox_inches="tight")
        if show:
            _show(plt)
        return fig


# ===========================================================================
# BACKEND-AGNOSTIC OUTPUT
# ===========================================================================
#
# WHY THIS EXISTS
#   Whether a matplotlib window appears depends on the backend, which
#   depends on the environment: a GUI toolkit on the desktop, "Agg" over
#   SSH or in CI, an IPython magic in Jupyter. "Nothing popped up" is the
#   most common confusion with plot(), and it is environmental rather
#   than a property of the figure.
#
#   The HTML path removes the question. The SAME matplotlib figure is
#   embedded in a small self-contained page and opened in a browser.
#   Every machine has one, none of it depends on a GUI toolkit, and it
#   works unchanged over SSH.
#
# WHY NOT A SECOND PLOTTING LIBRARY
#   pyqtgraph is not backend-agnostic — it IS a backend, and it needs Qt,
#   so it trades one toolkit dependency for another. Plotly would be
#   genuinely portable but means a parallel implementation of every panel
#   that has to stay in sync with the matplotlib one, plus a heavy
#   optional dependency, and its static export is worse for the
#   publication PDFs this repo already produces. Rendering the existing
#   figure to HTML keeps ONE figure definition.
#
# TRADE-OFF, STATED PLAINLY
#   The result is a static image in a page: no hover, no zoom. If
#   interactivity is ever needed, a `backend="plotly"` renderer can be
#   added alongside — this design does not preclude it.
# ===========================================================================

_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ margin:0; padding:24px; background:#f5f5f7;
        font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",
                    Helvetica,Arial,sans-serif; color:#1d1d1f; }}
 .wrap {{ max-width:1200px; margin:0 auto; }}
 h1 {{ font-size:15px; font-weight:600; margin:0 0 4px; }}
 p.sub {{ font-size:12px; color:#6e6e73; margin:0 0 18px; }}
 .card {{ background:#fff; border-radius:10px; padding:16px;
         box-shadow:0 1px 3px rgba(0,0,0,.12); }}
 img {{ width:100%; height:auto; display:block; }}
 @media (prefers-color-scheme: dark) {{
   body {{ background:#1c1c1e; color:#f5f5f7; }}
   .card {{ background:#2c2c2e; box-shadow:0 1px 3px rgba(0,0,0,.5); }}
   p.sub {{ color:#98989d; }}
 }}
</style></head><body><div class="wrap">
<h1>{title}</h1><p class="sub">{subtitle}</p>
<div class="card"><img src="data:image/png;base64,{b64}" alt="{title}"></div>
</div></body></html>
"""


def to_html(fig, path: Optional[str] = None, title: str = "cbm diagnostics",
            subtitle: str = "", open_browser: bool = True) -> str:
    """Write a matplotlib figure into a self-contained HTML page.

    The image is embedded as a base64 data URI, so the file is a single
    portable artifact — no sidecar PNG to lose when emailing it or
    copying it off a cluster.

    Returns the path written.
    """
    import base64
    import io
    import tempfile

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    if path is None:
        fd, path = tempfile.mkstemp(suffix=".html", prefix="cbm_fit_")
        import os
        os.close(fd)
    path = str(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_HTML_TEMPLATE.format(title=title, subtitle=subtitle,
                                       b64=b64))
    if open_browser:
        _open_browser(f"file://{path}")
    return path


def _open_browser(url: str) -> None:
    """Open a URL in the default browser.

    A module-level function rather than an inline `webbrowser.open` call
    so tests can substitute it — a verification run must not spawn
    browser tabs on the developer's desktop.
    """
    import webbrowser
    webbrowser.open(url)


def _in_notebook() -> bool:
    """True inside Jupyter/IPython with a display frontend.

    Used only to pick a sensible default: in a notebook the figure is
    rendered inline by the frontend, so opening a browser tab as well
    would be an annoyance rather than a help.
    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in getattr(ip, "config", {})
    except Exception:
        return False


def plot(result, subject: Optional[int] = None, backend: str = "auto",
         html_path: Optional[str] = None, **kw):
    """Diagnostic figure. Dispatches on how many subjects were fitted.

    subject=None (default): the group figure for a multi-subject fit, or
    the single-subject figure when only one subject was fitted — so a
    one-subject fit never shows a "distribution" of one point.
    subject=i: force the per-subject figure.

    backend:
      "auto"  (default) behave as before — build the figure and return
              it. `show=True` opens a matplotlib window where the
              environment allows one.
      "html"  render the same figure into a self-contained HTML page and
              open it in a browser. Independent of the matplotlib
              backend, so it works headless, over SSH, and anywhere a
              browser exists. Returns the figure; the path is also
              printed and available as `fig._cbm_html_path`.
      "mpl"   explicit synonym for "auto".

    html_path: where to write the page (default: a temp file).

    Extra keyword arguments go to the underlying plotter: `save=path`
    writes a PNG/PDF, `show=True` opens a matplotlib window.
    """
    backend = (backend or "auto").lower()
    if backend not in ("auto", "mpl", "matplotlib", "html"):
        raise ValueError(
            f"unknown backend {backend!r}; expected 'auto', 'mpl' or "
            f"'html'")

    n = np.atleast_2d(np.asarray(result.output.parameters)).shape[0]
    if subject is not None and not 0 <= subject < n:
        raise IndexError(
            f"subject {subject} out of range: this fit has {n}")

    # The HTML path never wants a matplotlib window as well.
    if backend == "html":
        kw.pop("show", None)

    if subject is not None:
        fig = plot_subject(result, subject=subject, **kw)
        which = f"subject {subject}"
    elif n == 1:
        fig = plot_subject(result, subject=0, **kw)
        which = "subject 0"
    else:
        fig = plot_group(result, **kw)
        which = f"{n} subjects"

    if backend == "html":
        model = getattr(getattr(result, "input", None), "model_name", "?")
        prof = getattr(result, "profile", None)
        when = getattr(prof, "datetime", "")
        path = to_html(
            fig, path=html_path,
            title=f"Fit diagnostics · {model} · {which}",
            subtitle=f"{getattr(result, 'method', '')}"
                     + (f" · {when}" if when else ""),
            open_browser=not _in_notebook())
        fig._cbm_html_path = path
        print(f"wrote {path}")
    return fig
