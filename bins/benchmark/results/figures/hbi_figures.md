# HBI benchmark figures — `hbi` grid

Two arms: **pristine pre-fork CBM** (`cbm/hbi_legacy.py`, finite-difference
refits) and **this fork** (`cbm/hbi.py`, MOD 11 Gauss-Newton refits). The HBI
update equations are byte-identical between them, so every difference below is
attributable to the curvature.

No MATLAB VBA arm: `VBA_MFX` fits one model at a time and returns a group free
energy, with no Dirichlet over model identity and therefore no
`model_frequency` to compare. That is a property of VBA's design, not a gap in
this benchmark.

10 cells, 40 subjects x 150 trials. Each (arm, cell) run from
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

**3 of 10 cells are affected at all** (either arm moving more
than 0.05 between map sources); on the rest, GN and FD individual fits
are nearly identical and there is nothing for either arm to be sensitive to.
Of the affected cells the fork closes the gap entirely (< 0.01) on **2**
and reduces it substantially on **1**. Worst cell: 0.5540 for
original CBM versus 0.1242 for the fork. No pooled average is quoted —
it would be dominated by the unaffected control cells. Per-cell values:

| cell | true mix | spread CBM | spread fork |
|---|---:|---:|---:|
| RLmix000 | 0.00 | 0.0000 | 0.0000 |
| RLmix030 | 0.30 | 0.0205 | 0.0212 |
| RLmix050 | 0.50 | 0.0055 | 0.0063 |
| RLmix070 | 0.70 | 0.0323 | 0.0050 |
| RLmix100 | 1.00 | 0.0000 | 0.0000 |
| VALmix000 | 0.00 | 0.0013 | 0.0013 |
| VALmix030 | 0.30 | 0.0038 | 0.0038 |
| VALmix050 | 0.50 | 0.2487 | 0.1242 |
| VALmix070 | 0.70 | 0.5540 | 0.0018 |
| VALmix100 | 1.00 | 0.4180 | 0.0000 |

**HBI Figure 2 · Convergence behaviour.** Median across seeds. (A) Iterations
to convergence — median 6 for CBM,
7 for the fork. (B) Difference in the
free-energy bound at exit, within cell; above zero means the fork reached the
better optimum. Plotted as a difference because the bound itself varies by
hundreds of nats between cells, which would hide the within-cell effect.
(C) Wall-clock seconds per run — median 25.7s
for CBM, 22.2s for the fork. (D) Percentage of
refits the fork flags as weakly identified (MOD 10 criterion, surfaced by MOD
12). Original CBM has no bar here because it discards these records entirely —
the absence is itself one of the differences under test.
