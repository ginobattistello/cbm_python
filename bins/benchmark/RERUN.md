# Re-running the cross-arm benchmark

The purpose of this pipeline is **regression testing across implementations**.
After any change to `cbm/`, re-run it and check that this fork still agrees with
the two references (pristine CBM at commit `e72193f`, and MATLAB VBA).

## The four commands

```bash
cd ~/Documents/cbm_python

# 1. simulate  (deterministic — same bytes every time)
python benchmark/simulate.py --grid clean

# 2. Python arms: this fork + pristine CBM          (~4 min)
python benchmark/run_python_arms.py --grid clean

# 3. MATLAB VBA arm                                  (~2.5 min)
/Users/gino.diez/Applications/Matlab_R2025b.app/bin/matlab \
    -batch "run('benchmark/run_vba_clean.m')"

# 4. figures + captions
python benchmark/make_report_figures.py
```

Output lands in `benchmark/results/figures/` — five figures as PDF (vector, for
a manuscript) and PNG, plus `figures.md` with a generated caption for each.
Captions are computed from the same data as the figures, so they cannot drift
out of date.

Step 1 can be skipped if `benchmark/data/clean/` already exists — the data is
seeded deterministically, so regenerating gives identical bytes.

Steps 2 and 3 are independent and can run at the same time.

## What to check, in priority order

| # | Check | Expected | Meaning if it moves |
|---|---|---|---|
| 1 | **Figure 3, cross-arm agreement** | every cell ≥ 0.999 | **The fork has diverged from BOTH references.** Investigate before anything else. |
| 2 | Figure 2 diagonal (recovery) | RL α 0.82 / β 0.90 · POW ρ 0.94 / β 0.77 | A drop with figure 3 still at 1.0 means all three arms moved together — suspect the simulator, not the fork. |
| 3 | Figure 1 medians | the three arms within ~0.005 of each other | One arm separating = that arm changed behaviour. |
| 4 | Failed fits (`figures.md` header) | fork 0, VBA 0, CBM 1 | A new fork failure is a regression. (The 1 CBM failure is a known upstream bug — see below.) |
| 5 | Figure 4 | MAP ~unchanged (RL 0/120, POW 15/119), evidence changed on ~all fits | This is the Gauss-Newton signature. Orange points are subjects whose MAP genuinely moved; a jump in their number means a modification has begun changing *where* fits land, not just their evidence. |
| 6 | Figure 5 AUC | the three arms within ~0.01 (≈0.62) | A spread here means one arm's model *selection* has shifted. |

Reference values from the 2026-08-12 run are in DEV.md §10.

## Known, expected failure

`cbm_orig` fails on exactly **1 of 960** fits. This is a real bug in the pristine
upstream code (`benchmark/external/cbm_original/optimization.py:394`): when every
initialization returns a non-finite objective, `best_result` is never assigned and
it raises `AttributeError: 'NoneType' object has no attribute 'x'`. The comparison
`result.f < best_f` is always False against NaN, and the retry loop has no guard.

The fork does not have this problem — MODIFICATION 4a wraps the objective so
non-finite values become a large finite penalty. `run_python_arms.py` fits the CBM
arm subject-by-subject so one pathological subject costs one subject rather than
the whole cell; failures are recorded as NaN and counted in the report header.

## Changing the design

All in `benchmark/simulate.py`:

- **noise level** — `LAPSE_RATE` (default 0.10), or `--lapse` on the command line
- **subjects / trials** — `CLEAN_N_SUBJECTS`, `CLEAN_T`
- **parameter ranges** — `CLEAN_ALPHA`, `CLEAN_BETA_RL`, `CLEAN_RHO`, `CLEAN_BETA_POW`

Two warnings from experience, both documented at length in DEV.md §9:

1. **Widening or narrowing the parameter ranges changes section 2 even if the
   toolboxes are untouched.** Recovery correlation is capped by how much the *true*
   parameters vary: `r ≈ SD_true / sqrt(SD_true² + SD_error²)`. Compare recovery
   numbers only between runs with the same ranges.
2. **The POW amount scale matters.** `AMOUNT_SCALE = 10` is calibrated — `x^ρ` is
   nearly flat below 1, and on a 0–1 scale ρ recovery drops from 0.92 to 0.60 with
   estimates running away to 30+.

## Adding a model

1. Add the likelihood (scalar) and the per-trial version to `benchmark/models.py`,
   register it in `MODELS` and in a `FAMILIES` entry.
2. Add a generator to `benchmark/simulate.py` and include it in `build_clean_grid`.
3. For the VBA arm, add an observation/evolution function under `benchmark/matlab/`
   and a branch in `benchmark/run_vba_clean.m`. **Verify its analytic gradients
   against finite differences before trusting any result** — both existing MATLAB
   models were checked this way (max error ~1e-10).
4. Add the model to `SPEC` in `benchmark/make_report_figures.py`, giving its
   rival candidate, whether that rival is the more complex of the two, and its
   parameter list. Everything else in the figures follows from `SPEC`.

## The HBI benchmark (separate pipeline)

The four commands above cover `individual_fit`. The **hierarchical** layer has its
own two-arm pipeline (DEV.md §15):

```bash
python benchmark/simulate.py --grid hbi        # 10 mixed-population cells
python benchmark/run_hbi_arms.py --grid hbi    # 2 arms x 2 map sources
python benchmark/make_hbi_figures.py
```

Two arms, not three: `cbm/hbi_legacy.py` (frozen pre-fork, finite-difference
refits) versus `cbm/hbi.py` (MOD 11, Gauss-Newton refits). **There is no VBA arm
by design** — `VBA_MFX` fits one model at a time and returns a group free energy,
with no Dirichlet over model identity and so no `model_frequency` to compare
against.

What to check:

| # | Check | Expected |
|---|---|---|
| 1 | **Fig 1C, map-source spread** | fork ≤ CBM in every cell |
| 2 | Fig 1A/B, recovery | both arms track the diagonal; bar height is the result, not distance from it |
| 3 | Fig 2B, bound difference | ≥ 0 (fork reaches the same or better optimum) |
| 4 | Fig 2D | only the fork has bars — CBM discards these records |

The cells mix two candidate models at known ratios (0/30/50/70/100 %). The
perturbation is **how the supplied individual maps were fitted** — Gauss-Newton
or finite-difference — because that is what actually moves them and it is a real
user choice. Varying the optimizer's random restarts instead was tried and
rejected: it moves the MAPs by ~1e-5, so it measures nothing (DEV.md §15.5).

The **value cells carry the experiment** — GN and FD maps differ by up to 2.7 in
theta there (6 of 12 subjects on VALmix070), while all RL cells agree to ~1e-5
and are effectively controls.

## Older grids

The exploratory grids from the first two rounds still simulate and fit, and are
documented in DEV.md §8–§9: `--grid rl` (stress axes), `rl_wide`,
`value_recovery`, `value_selection`, driven by `run_vba_arm.m` /
`run_vba_value.m`, plus `--grid boundary` (§12) driven by `run_vba_grid.m`.
They answer *how performance varies with trial count, noise, and proximity to
the parameter bounds*; the `clean` grid answers *do the implementations still
agree*.

Their bespoke renderers have been removed — `make_report_figures.py` targets the
`clean` grid. To render another grid, change `GRID` at the top of the script; the
`SPEC` entries carry over for any grid using the same models.
