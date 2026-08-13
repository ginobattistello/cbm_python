"""
Figures for the hierarchical (HBI) benchmark — this fork vs pristine CBM.

============================================================================
WHAT IS BEING COMPARED, AND WHY IT IS TWO ARMS
============================================================================

    cbm_orig   cbm/hbi_legacy.py — frozen pre-fork snapshot; `hbi_main` has
               no `model_trials` parameter, so every internal refit uses a
               finite-difference Hessian.
    fork_gn    cbm/hbi.py — MODIFICATION 11: `model_trials` reaches
               `optimize_map`, so the refits use Gauss-Newton curvature.

The HBI variational update equations are byte-identical between them; only
the Mod 11/12 plumbing differs. So anything measured here is attributable to
the curvature and nothing else.

There is no MATLAB VBA arm, deliberately. `VBA_MFX` fits one model at a time
and returns a group free energy — it has no Dirichlet over model identity and
therefore no `model_frequency` to compare against. A third arm would mean
correlating incommensurable quantities.

============================================================================
THE TWO QUESTIONS
============================================================================

Fig 1  STABILITY. HBI refits every subject starting from the supplied
       individual fits, so those fits are a SEED. The seed a real user
       varies is whether they passed `model_trials` to `individual_fit` —
       Gauss-Newton or finite-difference maps. (Varying the random restarts
       instead was tried and rejected: it moves the MAPs by ~1e-5, so there
       is nothing for HBI to be sensitive to. See DEV.md §15.5.)

       Each cell is therefore run from BOTH map sources, and we measure how
       far apart the two group verdicts land.

           small spread -> the verdict is a property of the data
           large spread -> it depends on an upstream choice the user may
                           not know they made

Fig 2  CONVERGENCE. What each arm's run looks like from the inside:
       iterations to converge, the free-energy bound reached, wall-clock
       cost, and how many refits fell back to the prior mean because the
       optimizer failed (the mechanism DEV.md §13.3 blamed the instability
       on).

Reading both together: Fig 1 is the claim about reliability, Fig 2 is the
mechanism that explains it.

    python benchmark/make_hbi_figures.py

Outputs to benchmark/results/figures/: hbi_fig1, hbi_fig2 (PDF + PNG) and
hbi_figures.md with generated captions.
"""

import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.patches import Patch                          # noqa: E402

BENCH = Path(__file__).resolve().parent
RESULTS = BENCH / "results"
FIGDIR = RESULTS / "figures"
GRID = "hbi"

# Same arm colours as the individual-fit figures, so a reader moving between
# the two figure sets does not have to re-learn the encoding.
ARMS = ("cbm_orig", "fork_gn")
ARM_LABEL = {"cbm_orig": "Original CBM (finite diff.)",
             "fork_gn": "This fork (Gauss-Newton)"}
ARM_SHORT = {"cbm_orig": "Original CBM", "fork_gn": "This fork"}
ARM_COLOR = {"cbm_orig": "#C4622D", "fork_gn": "#3B5B8C"}

FAM_LABEL = {"rl": "RL family  (RL vs RL2)",
             "value": "Value family  (LIN vs POW)"}


def set_style():
    """Identical to make_report_figures.py — one visual language."""
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "legend.fontsize": 7.5, "legend.frameon": False,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.4, "grid.color": "#D8D8D8",
    })


def panel_letter(ax, letter, dx=-0.14, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def load_rows():
    p = RESULTS / f"hbi_{GRID}.pkl"
    if not p.exists():
        raise SystemExit(f"missing {p} — run benchmark/run_hbi_arms.py first")
    return pickle.load(open(p, "rb"))


def by_cell(rows):
    """(arm, cell) -> list of per-seed rows, ordered by mixture fraction."""
    d = defaultdict(list)
    for r in rows:
        d[(r["arm"], r["cell"])].append(r)
    return d


def cells_in_order(rows, family):
    """Cell names for one family, sorted by true mixture fraction."""
    seen = {}
    for r in rows:
        if r["family"] == family:
            seen[r["cell"]] = r["mix"]
    return [c for c, _ in sorted(seen.items(), key=lambda kv: kv[1])]


# ===========================================================================
# FIGURE 1 — stability under seed perturbation
# ===========================================================================
#
# Panels A/B: the recovered group frequency of the COMPLEX model against the
# true mixture, one line per arm, with every individual seed drawn as a point
# and the seed range as a vertical bar. The dashed diagonal is perfect
# recovery. What to look for is not distance from the diagonal (that is the
# difficulty of the benchmark) but the HEIGHT OF THE BARS — a tall bar means
# the same data gave materially different answers depending on where the
# optimizer started.
#
# Panel C reduces that to one number per cell: max - min across seeds. This
# is the figure's actual claim, and putting it on its own axis stops a large
# recovery offset from visually drowning a small stability difference.
# ===========================================================================

def figure1(rows):
    fams = [f for f in ("rl", "value")
            if any(r["family"] == f for r in rows)]
    idx = by_cell(rows)

    fig = plt.figure(figsize=(7.2, 5.6))
    # bottom=0.16 leaves room for panel C's rotated cell labels; hspace is
    # tight because A/B's x-labels and C's title are the only things between
    # the rows.
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85],
                          hspace=0.52, wspace=0.28,
                          top=0.86, bottom=0.16)
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(len(fams))]
    ax_c = fig.add_subplot(gs[1, :])

    # -- A/B: recovery with the seed range shown ------------------------
    for ax, fam, L in zip(axes_top, fams, "AB"):
        names = cells_in_order(rows, fam)
        xs = [idx[(ARMS[0], c)][0]["mix"] if idx[(ARMS[0], c)]
              else idx[(ARMS[1], c)][0]["mix"] for c in names]

        ax.plot([0, 1], [0, 1], color="#BBBBBB", linewidth=0.8,
                linestyle=(0, (4, 3)), zorder=1)
        ax.text(0.72, 0.63, "perfect", fontsize=6.5, color="#999999",
                rotation=38, ha="center", va="center")

        for arm in ARMS:
            med, lo, hi, xok = [], [], [], []
            for c, x in zip(names, xs):
                vals = [r["freq_complex"] for r in idx[(arm, c)]]
                if not vals:
                    continue
                xok.append(x)
                med.append(np.median(vals))
                lo.append(min(vals))
                hi.append(max(vals))
                # both map sources as faint points: with only two, a
                # summary alone would hide which way each one went
                ax.scatter([x] * len(vals), vals, s=7,
                           color=ARM_COLOR[arm], alpha=0.45, linewidths=0,
                           zorder=3)
            if not xok:
                continue
            ax.vlines(xok, lo, hi, color=ARM_COLOR[arm], linewidth=2.2,
                      alpha=0.35, zorder=2)
            ax.plot(xok, med, color=ARM_COLOR[arm], linewidth=1.3,
                    marker="o", markersize=3.2, zorder=4,
                    label=ARM_SHORT[arm])

        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.06, 1.06)
        ax.set_xlabel("true fraction from the complex model")
        if L == "A":
            ax.set_ylabel("recovered group frequency\nof the complex model")
        ax.set_title(FAM_LABEL[fam], pad=6)
        ax.grid(alpha=0.5)
        panel_letter(ax, L, dx=-0.20 if L == "A" else -0.12)
        if L == "A":
            ax.legend(loc="upper left", fontsize=6.8)

    # -- C: the spread itself -------------------------------------------
    labels, spread = [], {a: [] for a in ARMS}
    for fam in fams:
        for c in cells_in_order(rows, fam):
            labels.append(c)
            for arm in ARMS:
                vals = [r["freq_complex"] for r in idx[(arm, c)]]
                spread[arm].append(max(vals) - min(vals) if vals else np.nan)

    x = np.arange(len(labels))
    w = 0.36
    for i, arm in enumerate(ARMS):
        ax_c.bar(x + (i - 0.5) * w, spread[arm], w * 0.92,
                 color=ARM_COLOR[arm], label=ARM_SHORT[arm],
                 edgecolor="white", linewidth=0.6)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels, rotation=30, ha="right", fontsize=6.8)
    ax_c.set_ylabel("spread in group frequency\n(GN maps vs FD maps)")
    ax_c.set_title("Sensitivity to how the supplied maps were fitted   "
                   "(lower is better)", pad=6)
    ax_c.grid(axis="y", alpha=0.5)
    ax_c.legend(loc="upper right", fontsize=7)
    panel_letter(ax_c, "C", dx=-0.075, dy=1.03)

    fig.suptitle("HBI Figure 1 · Does the group verdict depend on the "
                 "optimizer's starting point?",
                 fontsize=10, fontweight="bold", y=0.98)
    fig.text(0.5, 0.925,
             "Each cell run from both Gauss-Newton and finite-difference "
             "individual fits. Bars in A/B span that range; panel C is the "
             "range as one number.",
             ha="center", fontsize=7.2, color="#555555")
    save(fig, "hbi_fig1_stability")


# ===========================================================================
# FIGURE 2 — convergence behaviour
# ===========================================================================
#
# Four small panels, each one number per (arm, cell), paired so the same cell
# is directly comparable across arms:
#
#   A  iterations to convergence     — does one arm need more passes?
#   B  free-energy bound at exit     — which found the better optimum?
#                                      (higher is better; plotted as the
#                                      fork's advantage so the zero line
#                                      is the meaningful reference)
#   C  wall-clock seconds            — what the curvature costs in practice
#   D  weakly-identified refits      — Mod 10/12; only the fork can report
#                                      this, which is itself a difference
#
# Panel B is a difference rather than two absolute series because the bound
# varies by hundreds of nats between cells; plotted absolutely, the
# within-cell difference we care about would be invisible.
# ===========================================================================

def figure2(rows):
    idx = by_cell(rows)
    fams = [f for f in ("rl", "value") if any(r["family"] == f for r in rows)]
    labels = []
    for fam in fams:
        labels += cells_in_order(rows, fam)

    def per_cell(arm, key, reduce=np.median):
        out = []
        for c in labels:
            vals = [r[key] for r in idx[(arm, c)]
                    if r.get(key) is not None and np.isfinite(r[key])]
            out.append(reduce(vals) if vals else np.nan)
        return np.asarray(out, float)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    x = np.arange(len(labels))
    w = 0.36

    # -- A: iterations ---------------------------------------------------
    ax = axes[0, 0]
    for i, arm in enumerate(ARMS):
        ax.bar(x + (i - 0.5) * w, per_cell(arm, "n_iter"), w * 0.92,
               color=ARM_COLOR[arm], label=ARM_SHORT[arm],
               edgecolor="white", linewidth=0.6)
    ax.set_ylabel("iterations to converge")
    ax.set_title("Iterations", pad=6)
    panel_letter(ax, "A", dx=-0.17, dy=1.04)

    # -- B: bound advantage ----------------------------------------------
    ax = axes[0, 1]
    db = per_cell("fork_gn", "bound") - per_cell("cbm_orig", "bound")
    cols = ["#3B5B8C" if v >= 0 else "#C4622D" for v in db]
    ax.bar(x, db, w * 1.6, color=cols, edgecolor="white", linewidth=0.6)
    ax.axhline(0, color="#333333", linewidth=0.7)
    ax.set_ylabel("$\\Delta$ free-energy bound\n(fork $-$ CBM, nats)")
    ax.set_title("Bound reached   (above 0 = fork better)", pad=6)
    panel_letter(ax, "B", dx=-0.19, dy=1.04)

    # -- C: wall clock ---------------------------------------------------
    ax = axes[1, 0]
    for i, arm in enumerate(ARMS):
        ax.bar(x + (i - 0.5) * w, per_cell(arm, "seconds_hbi"), w * 0.92,
               color=ARM_COLOR[arm], edgecolor="white", linewidth=0.6)
    ax.set_ylabel("seconds per HBI run")
    ax.set_title("Wall-clock cost", pad=6)
    panel_letter(ax, "C", dx=-0.17, dy=1.04)

    # -- D: weakly identified refits (fork only) --------------------------
    ax = axes[1, 1]
    nw = per_cell("fork_gn", "n_weak")
    nd = per_cell("fork_gn", "n_diag")
    frac = np.where(nd > 0, nw / np.maximum(nd, 1), np.nan) * 100
    ax.bar(x, frac, w * 1.6, color="#7A6BA8", edgecolor="white",
           linewidth=0.6)
    ax.set_ylabel("% of refits flagged\nweakly identified")
    ax.set_title("Fit-quality reporting (Mod 12)", pad=6)
    ax.set_ylim(0, max(100.0, float(np.nanmax(frac)) * 1.15)
                if np.isfinite(np.nanmax(frac)) else 100.0)
    panel_letter(ax, "D", dx=-0.19, dy=1.04)

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6.0)
        ax.grid(axis="y", alpha=0.5)

    fig.suptitle("HBI Figure 2 · Convergence behaviour",
                 fontsize=10, fontweight="bold", y=0.99)
    fig.text(0.5, 0.935,
             "Median across seeds, per cell. Panel B is a within-cell "
             "difference because the bound itself varies by hundreds of "
             "nats between cells.",
             ha="center", fontsize=7.2, color="#555555")
    fig.legend(handles=[Patch(facecolor=ARM_COLOR[a], label=ARM_SHORT[a])
                        for a in ARMS],
               loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2,
               fontsize=7)
    fig.text(0.5, 0.02,
             "Panel D: only this fork has bars — original CBM discards "
             "these records entirely (that absence is one of the "
             "differences under test).",
             ha="center", fontsize=6.6, color="#777777", style="italic")
    fig.subplots_adjust(hspace=0.78, wspace=0.34, top=0.82, bottom=0.14)
    save(fig, "hbi_fig2_convergence")


# ===========================================================================

def write_captions(rows):
    """Captions computed from the same rows as the figures, so they cannot
    drift out of date."""
    idx = by_cell(rows)
    fams = [f for f in ("rl", "value") if any(r["family"] == f for r in rows)]
    labels = []
    for fam in fams:
        labels += cells_in_order(rows, fam)

    def spread(arm):
        out = []
        for c in labels:
            v = [r["freq_complex"] for r in idx[(arm, c)]]
            if v:
                out.append(max(v) - min(v))
        return np.asarray(out, float)

    s_old, s_new = spread("cbm_orig"), spread("fork_gn")
    worst_old = float(np.nanmax(s_old)) if len(s_old) else float("nan")
    worst_new = float(np.nanmax(s_new)) if len(s_new) else float("nan")
    # NO POOLED MEAN. Most cells are controls where GN and FD maps are
    # nearly identical, so an average over all cells is dominated by cells
    # that cannot show an effect and understates the ones that can. Report
    # the affected cells explicitly instead (DEV.md §9.7 lesson).
    AFFECTED = 0.05          # a cell counts as affected if EITHER arm
    #                          moves more than this between map sources
    aff = [(c, o, n) for c, o, n in zip(labels, s_old, s_new)
           if max(o, n) > AFFECTED]
    n_aff = len(aff)
    closed = sum(1 for _, o, n in aff if n < 0.01)
    partial = sum(1 for _, o, n in aff if 0.01 <= n < o * 0.75)

    def med(arm, key):
        v = [r[key] for r in rows
             if r["arm"] == arm and r.get(key) is not None
             and np.isfinite(r[key])]
        return float(np.median(v)) if v else float("nan")

    n_seeds = max((len(v) for v in idx.values()), default=0)
    n_sub = rows[0]["n_subjects"]
    n_tr = rows[0]["n_trials"]

    # per-cell table so no pooled number goes unqualified (DEV.md §9.7)
    tbl = ["| cell | true mix | spread CBM | spread fork |",
           "|---|---:|---:|---:|"]
    for c in labels:
        r0 = (idx[("cbm_orig", c)] or idx[("fork_gn", c)])[0]
        vo = [r["freq_complex"] for r in idx[("cbm_orig", c)]]
        vn = [r["freq_complex"] for r in idx[("fork_gn", c)]]
        tbl.append(
            f"| {c} | {r0['mix']:.2f} | "
            f"{(max(vo) - min(vo)) if vo else float('nan'):.4f} | "
            f"{(max(vn) - min(vn)) if vn else float('nan'):.4f} |")

    md = f"""# HBI benchmark figures — `{GRID}` grid

Two arms: **pristine pre-fork CBM** (`cbm/hbi_legacy.py`, finite-difference
refits) and **this fork** (`cbm/hbi.py`, MOD 11 Gauss-Newton refits). The HBI
update equations are byte-identical between them, so every difference below is
attributable to the curvature.

No MATLAB VBA arm: `VBA_MFX` fits one model at a time and returns a group free
energy, with no Dirichlet over model identity and therefore no
`model_frequency` to compare. That is a property of VBA's design, not a gap in
this benchmark.

{len(labels)} cells, {n_sub} subjects x {n_tr} trials. Each (arm, cell) run from
both map sources.

---

**HBI Figure 1 · Stability under seed perturbation.** HBI refits every subject
starting from the supplied individual fits, so those fits act as a seed. The
seed varied here is the one a real user varies: whether `model_trials` was
passed to `individual_fit`, i.e. Gauss-Newton versus finite-difference maps.
(Varying the random restarts instead was measured and rejected — it moves the
MAPs by only ~1e-5, leaving nothing for HBI to be sensitive to; DEV.md §15.5.)
(A, B) Recovered group frequency of the complex model against the
true mixture; points are individual seeds, the vertical bar spans the seed
range, the dashed diagonal is perfect recovery. Distance from the diagonal is
benchmark difficulty; **bar height is the result** — it is the extent to which
the same data gave a different answer depending on an upstream choice the user
may not know they made. (C) That range as one number per cell.

**{n_aff} of {len(labels)} cells are affected at all** (either arm moving more
than {AFFECTED:g} between map sources); on the rest, GN and FD individual fits
are nearly identical and there is nothing for either arm to be sensitive to.
Of the affected cells the fork closes the gap entirely (< 0.01) on **{closed}**
and reduces it substantially on **{partial}**. Worst cell: {worst_old:.4f} for
original CBM versus {worst_new:.4f} for the fork. No pooled average is quoted —
it would be dominated by the unaffected control cells. Per-cell values:

{chr(10).join(tbl)}

**HBI Figure 2 · Convergence behaviour.** Median across seeds. (A) Iterations
to convergence — median {med('cbm_orig', 'n_iter'):.0f} for CBM,
{med('fork_gn', 'n_iter'):.0f} for the fork. (B) Difference in the
free-energy bound at exit, within cell; above zero means the fork reached the
better optimum. Plotted as a difference because the bound itself varies by
hundreds of nats between cells, which would hide the within-cell effect.
(C) Wall-clock seconds per run — median {med('cbm_orig', 'seconds_hbi'):.1f}s
for CBM, {med('fork_gn', 'seconds_hbi'):.1f}s for the fork. (D) Percentage of
refits the fork flags as weakly identified (MOD 10 criterion, surfaced by MOD
12). Original CBM has no bar here because it discards these records entirely —
the absence is itself one of the differences under test.
"""
    (FIGDIR / "hbi_figures.md").write_text(md)
    print("  wrote hbi_figures.md")


def main():
    set_style()
    rows = load_rows()
    arms = sorted(set(r["arm"] for r in rows))
    cells = sorted(set(r["cell"] for r in rows))
    print(f"loaded {len(rows)} runs — {len(arms)} arms x {len(cells)} cells")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    figure1(rows)
    figure2(rows)
    write_captions(rows)
    print(f"\nall output in {FIGDIR}")


if __name__ == "__main__":
    main()
