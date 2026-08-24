"""
Diagnostic plots for individual CBM fits.

``Config(display=True)`` retains diagnostics and automatically shows the fit
figure after fitting.

Data convention
---------------
Each subject is standardized as:

    data = {"y": observed_outcomes, "X": model_inputs}

When an ``observation(theta, data)`` function is supplied, panel A is chosen
automatically:

- binary y + probability vector (T,) -> calibration plot
- categorical y + probability matrix (T, K) -> probabilistic class matrix
- continuous y + prediction vector (T,) -> observed-vs-predicted scatter

The optimization panel shows the log joint during both L-BFGS-B and optional
GN polishing. GN curvature is never displayed as evidence curvature.
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
    """Show the current matplotlib figure.

    There is intentionally no user-facing ``show`` or ``block`` option.
    ``display`` is the only visibility switch used by the toolbox.
    """
    import matplotlib

    backend = matplotlib.get_backend()
    if backend.lower() == "agg":
        warnings.warn(
            f"display=True but the matplotlib backend is {backend!r}, "
            "which cannot open a window. The figure was still created; "
            "use an interactive matplotlib backend to display it.",
            UserWarning,
            stacklevel=3,
        )
        return

    # Plain matplotlib behavior is used deliberately. No separate blocking
    # policy is maintained by CBM.
    plt.show()

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
            "Config(display=True). The display option both retains the "
            "diagnostics and shows the fit figure after fitting.")
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
    """Return ``(text, is_alert)`` status lines for one subject."""
    diag = None
    dl = getattr(result.math, "diagnostics", None)
    if dl is not None and i < len(dl):
        diag = dl[i]

    flag = float(np.ravel(result.math.flag)[i])

    if diag is None:
        return [
            (
                "MAP optimization failed — prior substituted "
                "(config.prior_for_failed)",
                True,
            )
        ]

    lines = []

    status = getattr(diag, "convergence_status", None)
    healthy_status = status in (
        "converged_df",
        "no_improvement",
        "skipped_no_trial_func",
    )
    lines.append(
        (
            f"MAP optimization: {status}",
            not healthy_status,
        )
    )

    lines.append(
        (
            f"observed Hessian: {getattr(diag, 'hess_method', '?')}",
            False,
        )
    )

    laplace_valid = bool(getattr(diag, "laplace_valid", False))
    lines.append(
        (
            f"Laplace valid: {'yes' if laplace_valid else 'NO'}",
            not laplace_valid,
        )
    )

    min_eig = getattr(diag, "hess_raw_min_eig", None)
    if min_eig is not None:
        lines.append(
            (
                f"min Hessian eigenvalue: {min_eig:.2e}",
                min_eig <= 0,
            )
        )

    lines.append(
        (
            f"|gradient|: {getattr(diag, 'abs_grad', float('nan')):.2e}",
            False,
        )
    )

    nia = getattr(diag, "n_inits_agreeing", None)
    nr = getattr(diag, "n_runs", None)
    if nia is not None:
        lines.append(
            (f"inits agreeing: {nia}/{nr}", nia == 0)
        )

    ahb = getattr(diag, "at_hard_bounds", None)
    if ahb is not None and np.any(ahb):
        idx = [
            j for j, value in enumerate(np.ravel(ahb)) if value
        ]
        lines.append(
            (f"MAP at hard bounds: theta{idx}", True)
        )

    if flag == 1.0:
        lines.append(("fit flag 1.0 — accepted", False))
    elif flag == 0.5:
        lines.append(
            (
                "fit flag 0.5 — MAP retained with warning",
                True,
            )
        )
    elif flag == 0.0:
        lines.append(("fit flag 0.0 — fit failed", True))
    else:
        lines.append((f"fit flag {flag:g}", flag < 1.0))

    return lines

# ===========================================================================
# SINGLE SUBJECT
# ===========================================================================

def plot_subject(result, subject: int = 0, figsize=(9.0, 7.2),
                 save: Optional[str] = None, display: bool = True):
    """Diagnostic figure for one subject.

    A  prediction diagnostic chosen from y and observation(theta, data)
    B  parameter trajectories over the search, with the final +/-1 SE band
    C  objective evolution: log-joint during L-BFGS-B and GN polish
    D  parameter estimates with 95% intervals
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
        observation = dd.get("observation")

        # ---- A: outcome-specific prediction diagnostic -------------------
        y = np.asarray(data_i["y"])
        if observation is not None:
            try:
                pred = np.asarray(
                    observation(theta, data_i),
                    dtype=float,
                )

                y_flat = np.asarray(y).reshape(-1)

                if pred.ndim == 1 and pred.size == y_flat.size:
                    unique_y = np.unique(y_flat)
                    is_binary = (
                        unique_y.size <= 2
                        and np.all(np.isin(unique_y, [0, 1]))
                        and np.all((pred >= 0.0) & (pred <= 1.0))
                    )

                    if is_binary:
                        # Binary probabilistic calibration. Quantile bins keep
                        # useful occupancy even for peaked choice models.
                        n_bins = min(8, max(3, int(np.sqrt(pred.size))))
                        edges = np.linspace(0.0, 1.0, n_bins + 1)

                        xs = []
                        ys = []
                        ns = []

                        for b in range(n_bins):
                            if b == n_bins - 1:
                                mask = (
                                    (pred >= edges[b])
                                    & (pred <= edges[b + 1])
                                )
                            else:
                                mask = (
                                    (pred >= edges[b])
                                    & (pred < edges[b + 1])
                                )

                            if np.any(mask):
                                xs.append(float(np.mean(pred[mask])))
                                ys.append(float(np.mean(y_flat[mask])))
                                ns.append(int(np.sum(mask)))

                        axA.plot(
                            [0, 1],
                            [0, 1],
                            ls="--",
                            lw=0.8,
                            color=_GREY,
                        )
                        axA.plot(
                            xs,
                            ys,
                            marker="o",
                            lw=1.1,
                            color=_INK_DARK,
                        )
                        axA.set_xlim(-0.02, 1.02)
                        axA.set_ylim(-0.02, 1.02)
                        axA.set_xlabel("predicted P(y=1)")
                        axA.set_ylabel("observed frequency y=1")
                        axA.grid(alpha=0.5)

                        accuracy = np.mean(
                            (pred >= 0.5).astype(int) == y_flat.astype(int)
                        )
                        axA.text(
                            0.03,
                            0.97,
                            f"n={len(y_flat)} · accuracy={accuracy:.2f}",
                            transform=axA.transAxes,
                            ha="left",
                            va="top",
                            fontsize=6.7,
                            color=_GREY,
                        )
                        _panel(axA, "A", "binary calibration")

                    else:
                        # Continuous outcome: predictions and observations live
                        # in the same one-dimensional space.
                        axA.scatter(
                            pred,
                            y_flat,
                            s=12,
                            alpha=0.65,
                            color=_INK,
                        )

                        lo = float(np.nanmin([pred.min(), y_flat.min()]))
                        hi = float(np.nanmax([pred.max(), y_flat.max()]))

                        if np.isfinite(lo) and np.isfinite(hi):
                            axA.plot(
                                [lo, hi],
                                [lo, hi],
                                ls="--",
                                lw=0.8,
                                color=_GREY,
                            )

                        axA.set_xlabel("predicted")
                        axA.set_ylabel("observed")
                        axA.grid(alpha=0.5)

                        rmse = float(
                            np.sqrt(np.mean((y_flat - pred) ** 2))
                        )
                        axA.text(
                            0.03,
                            0.97,
                            f"RMSE={rmse:.3g}",
                            transform=axA.transAxes,
                            ha="left",
                            va="top",
                            fontsize=6.7,
                            color=_GREY,
                        )
                        _panel(axA, "A", "continuous prediction")

                elif pred.ndim == 2 and pred.shape[0] == y_flat.size:
                    # Categorical probabilistic matrix:
                    # row i = trials where observed class is i
                    # cell (i,j) = mean P(predicted class j | observed class i)
                    classes = np.unique(y_flat)
                    K = pred.shape[1]

                    if (
                        not np.all(np.equal(np.mod(classes, 1), 0))
                        or np.min(classes) < 0
                        or np.max(classes) >= K
                    ):
                        raise ValueError(
                            "categorical observation expects integer labels "
                            "between 0 and K-1"
                        )

                    classes = classes.astype(int)
                    matrix = np.full((len(classes), K), np.nan)

                    for i, cls in enumerate(classes):
                        mask = y_flat.astype(int) == cls
                        matrix[i, :] = np.mean(pred[mask, :], axis=0)

                    image = axA.imshow(
                        matrix,
                        vmin=0.0,
                        vmax=1.0,
                        aspect="auto",
                    )

                    axA.set_xticks(np.arange(K))
                    axA.set_xticklabels(np.arange(K))
                    axA.set_yticks(np.arange(len(classes)))
                    axA.set_yticklabels(classes)
                    axA.set_xlabel("predicted category")
                    axA.set_ylabel("observed category")

                    for i in range(matrix.shape[0]):
                        for j in range(matrix.shape[1]):
                            value = matrix[i, j]
                            if np.isfinite(value):
                                axA.text(
                                    j,
                                    i,
                                    f"{value:.2f}",
                                    ha="center",
                                    va="center",
                                    fontsize=6.2,
                                )

                    _panel(axA, "A", "categorical predicted probabilities")

                else:
                    raise ValueError(
                        "observation output must be (T,) for binary/continuous "
                        "data or (T, K) for categorical data"
                    )

            except Exception as exc:
                axA.text(
                    0.5,
                    0.5,
                    f"observation() raised:\n{type(exc).__name__}: {exc}",
                    ha="center",
                    va="center",
                    transform=axA.transAxes,
                    fontsize=7.0,
                    color=_GREY,
                )
                _panel(axA, "A", "prediction diagnostic — failed")

        else:
            axA.text(
                0.5,
                0.5,
                "no observation() supplied",
                ha="center",
                va="center",
                transform=axA.transAxes,
                fontsize=7.5,
                color=_GREY,
            )
            _panel(axA, "A", "prediction diagnostic")

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
        # Both traces show the same quantity: the log joint. The observed
        # Hessian is computed only after optimization, so evidence is not
        # defined at intermediate GN polish steps.
        sf = getattr(diag, "search_f", None) if diag is not None else None
        pf = getattr(diag, "polish_f", None) if diag is not None else None

        drew = False

        if sf is not None and len(sf):
            sf = np.asarray(sf, dtype=float)
            x_search = np.arange(sf.size)
            axC.plot(
                x_search,
                np.maximum.accumulate(sf),
                lw=1.1,
                color=_INK_DARK,
                label="L-BFGS-B log-joint (best so far)",
            )
            axC.plot(
                x_search,
                sf,
                lw=0.5,
                color=_INK,
                alpha=0.30,
                label="L-BFGS-B evaluations",
            )
            axC.set_xlabel("function evaluations")
            axC.set_ylabel("log-joint")
            drew = True

        if pf is not None and len(pf):
            pf = np.asarray(pf, dtype=float)
            finite_pf = pf[np.isfinite(pf)]

            if finite_pf.size:
                # GN usually contains only a few accepted steps, so an inset
                # preserves their scale without pretending that polish steps
                # are L-BFGS-B function evaluations.
                ins = (
                    axC.inset_axes([0.60, 0.43, 0.35, 0.36])
                    if drew
                    else axC
                )
                ins.plot(
                    np.arange(pf.size),
                    pf,
                    lw=1.2,
                    marker="o",
                    ms=3.0,
                    color=_INK_DARK,
                )
                ins.set_title("GN polish · log-joint", fontsize=6.2, pad=2)
                ins.set_xlabel("polish step", fontsize=5.8, labelpad=1)
                ins.tick_params(labelsize=5.5, length=1.8, pad=1)
                ins.xaxis.get_major_locator().set_params(integer=True)
                for spine in ins.spines.values():
                    spine.set_linewidth(0.5)
                drew = True

        if drew:
            axC.grid(alpha=0.6)
            if sf is not None and len(sf):
                axC.legend(loc="lower right", fontsize=6.2)
            _panel(axC, "C", "log-joint during optimization")
        else:
            axC.text(
                0.5,
                0.5,
                "no optimization trace retained",
                ha="center",
                va="center",
                transform=axC.transAxes,
                fontsize=7.5,
                color=_GREY,
            )
            _panel(axC, "C", "log-joint during optimization")

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
        n_polish = (
            getattr(diag, "n_polish_steps", None)
            if diag is not None
            else None
        )
        if n_polish is not None:
            cost.append(f"GN polish steps: {n_polish}")
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
        if display:
            _show(plt)
        return fig


# ===========================================================================
# MANY SUBJECTS
# ===========================================================================

def plot_group(result, figsize=(9.0, 6.2), save: Optional[str] = None,
               display: bool = True):
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
        if display:
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


def plot(
    result,
    subject: Optional[int] = None,
    backend: str = "auto",
    html_path: Optional[str] = None,
    display: bool = True,
    **kw,
):
    """Create a diagnostic figure.

    ``display`` is the only visibility switch. ``Config(display=True)`` calls
    this function automatically after fitting. Calling ``fit.plot()`` later
    defaults to ``display=True`` and shows the retained diagnostics again.

    ``subject=None`` shows the group figure for multi-subject fits and the
    single-subject figure for a one-subject fit. ``subject=i`` forces the
    per-subject figure.

    ``backend='html'`` writes the figure to a self-contained HTML page; in
    that case no matplotlib window is opened.
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

    # HTML rendering never opens a matplotlib window as well.
    plot_display = bool(display and backend != "html")

    if subject is not None:
        fig = plot_subject(
            result, subject=subject, display=plot_display, **kw
        )
        which = f"subject {subject}"
    elif n == 1:
        fig = plot_subject(
            result, subject=0, display=plot_display, **kw
        )
        which = "subject 0"
    else:
        fig = plot_group(result, display=plot_display, **kw)
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
