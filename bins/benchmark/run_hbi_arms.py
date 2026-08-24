"""
Hierarchical (HBI) benchmark: this fork vs pristine pre-fork CBM.

============================================================================
WHY THIS IS A TWO-ARM BENCHMARK, NOT THREE
============================================================================

The §10 individual-fit benchmark has three arms, the third (MATLAB VBA) being
the independent check that keeps the two CBM-lineage arms honest. That design
does not transfer here, for a reason worth stating rather than quietly
working around:

  **VBA_MFX answers a different question.** It fits ONE model at a time and
  returns a group-level free energy. CBM's HBI fits all K models jointly with
  a Dirichlet over model identity, and reports how the population divides
  between them. There is no `model_frequency` in VBA_MFX to correlate
  against. Forcing a third arm would mean comparing incommensurable
  quantities, so this benchmark has two arms and says so.

============================================================================
WHAT THE TWO ARMS ARE
============================================================================

    cbm_orig   cbm/hbi_legacy.py  — frozen pre-fork snapshot. `hbi_main` has
               no `model_trials` parameter at all, so every internal refit
               uses the Mod 2 finite-difference Hessian.
    fork_gn    cbm/hbi.py         — MOD 11: `model_trials` threaded through
               to `optimize_map`, so the refits use Gauss-Newton curvature.

The HBI variational update equations are BYTE-IDENTICAL between the two
(`diff` shows only the Mod 11/12 plumbing). This matters for interpretation:
with `model_trials=None` the arms are provably the same code path, so any
difference measured here is attributable to the curvature and nothing else.

============================================================================
THE TWO ANALYSES
============================================================================

1. STABILITY UNDER SEED PERTURBATION  (the headline)

   HBI does not fit subjects from scratch — it refits them starting from the
   supplied `cbm_map` parameters. So the supplied maps are a SEED, and a
   trustworthy inference should not depend much on which reasonable seed it
   was given.

   WHICH PERTURBATION. The first version of this script varied the random
   restarts inside `individual_fit` (different `np.random.seed`, same
   `num_init`). **That was measured and rejected**: the multi-start optimizer
   converges to the same optimum regardless of restart, so the supplied MAPs
   moved by only ~1e-5 and there was nothing for HBI to be sensitive to. All
   four "seeds" gave the same group frequency to 4 decimals — a property of
   the perturbation, not a finding about the arms.

   What actually moves the MAPs is the CURVATURE used to produce them:
   Gauss-Newton versus finite-difference individual fits differ by up to 2.7
   in theta on the value cells (0 subjects affected on RL cells, 6 of 12 on
   VALmix070). That is the seed §13.3 was about, and it is a choice a real
   user makes — whether or not they pass `model_trials` to `individual_fit`.

   So each cell is run from BOTH map sources, and the measured quantity is
   how far apart the two group verdicts land:

       small spread -> the verdict is a property of the data
       large spread -> it depends on an upstream choice the user may not
                       even know they made

   Crossing that with the two HBI arms gives four combinations; the pair
   that matters is whether GN refits (Mod 11) close a gap that FD refits
   leave open. DEV.md §14.3 found 0.382 vs 0.0004 on one dataset — this asks
   whether it holds across the grid.

2. CONVERGENCE BEHAVIOUR

   Three things HBI reports about its own run:
     - iterations to convergence
     - the free-energy bound at exit
     - how often a subject's refit FAILED and fell back to the prior mean
       (the `flag_kn == 0` branch in hbi_qhquad, which substitutes the prior
       and abandons that subject's own fit for that iteration)

   The third is the mechanism §13.3 identified: finite-difference refits
   stalling at or near the prior. This counts it at scale.

============================================================================

Output: benchmark/results/hbi_<grid>.pkl, one row per
(arm, cell, seed) with the group result, timings and diagnostics.

    python benchmark/run_hbi_arms.py --grid hbi
    python benchmark/run_hbi_arms.py --grid hbi --seeds 3 --cells RLmix050
"""

import argparse
import json
import pickle
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BENCH_DIR))

from models import MODELS, PRIOR_VARIANCE, FAMILIES          # noqa: E402
from cbm.individual_fit import individual_fit                # noqa: E402
from cbm.hbi import hbi_main as hbi_fork                     # noqa: E402
from cbm.hbi_legacy import hbi_main as hbi_legacy            # noqa: E402

# Individual-fit config. num_init=3 keeps the per-seed cost down; the
# restarts are what the seed perturbation actually varies.
FIT_CONFIG = dict(num_init=3, verbose=False)
# tolx loosened exactly as examples/example_RL.py does — HBI is slow and
# the default tolerance buys precision we do not need for a group frequency.
HBI_CONFIG = {"save_prog": False, "verbose": False, "tolx": 0.05}


def load_cell(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def cell_to_data(cell):
    n = int(cell["n_subjects"])
    if str(cell.get("family")) == "value":
        return [(cell["sure"][i], cell["gamble"][i],
                 cell["prob"][i], cell["chose"][i]) for i in range(n)]
    return [(cell["choices"][i], cell["rewards"][i]) for i in range(n)]


def fit_maps(data, candidates, seed, tmpdir, tag, use_gn):
    """Individual fits for every candidate model, written as pickles.

    `use_gn` is the perturbation axis (see the module docstring): it selects
    Gauss-Newton or finite-difference curvature for the SUPPLIED maps, which
    is what actually moves them. `seed` only fixes the restarts so a rerun
    is reproducible — it is not the thing being varied.
    """
    paths = []
    for mi, mname in enumerate(candidates):
        model, model_trials, d, prior_mean = MODELS[mname]
        np.random.seed(seed + 1000 * mi)
        fit = individual_fit(data, model, prior_mean, PRIOR_VARIANCE,
                             config=dict(FIT_CONFIG),
                             model_trials=model_trials if use_gn else None)
        p = tmpdir / f"map_{tag}_{mname}.pkl"
        with open(p, "wb") as f:
            pickle.dump(fit, f)
        paths.append(str(p))
    return paths


def diag_summary(result):
    """Aggregate the Mod 12 per-refit diagnostics.

    Only the fork arm has them: `cbm_orig` is the frozen pre-Mod-12
    snapshot whose IndividualPosterior has no `diagnostics` field at all,
    so every field here comes back None for that arm. That asymmetry is
    the point — it IS one of the differences under test.

    `n_prior_fallback` counts (model, subject) refits with no diagnostics
    record. In hbi_qhquad a refit that returns flag 0 is discarded and
    replaced by the prior mean; those slots carry no optimizer record.
    This is the failure mode DEV.md §13.3 traced the instability to.
    """
    none_row = dict(hess_methods=None, n_inits_min=None, n_weak=None,
                    n_diag=None, n_prior_fallback=None)
    qh = getattr(getattr(result, "math", None), "qhquad", None)
    if qh is None:
        return none_row
    dg = getattr(qh, "diagnostics", None)
    if dg is None:
        return none_row

    flat = list(np.asarray(dg, dtype=object).ravel())
    recs = [x for x in flat if x is not None]
    if not recs:
        return dict(hess_methods=None, n_inits_min=None, n_weak=None,
                    n_diag=0, n_prior_fallback=len(flat))

    nia = [v for v in (getattr(r, "n_inits_agreeing", None) for r in recs)
           if v is not None]
    weak = [v for v in (getattr(r, "weak_identifiability", None)
                        for r in recs) if v is not None]
    return dict(
        hess_methods=dict(Counter(getattr(r, "hess_method", None)
                                  for r in recs)),
        n_inits_min=int(min(nia)) if nia else None,
        n_weak=int(sum(1 for v in weak if v < 2.0)) if weak else None,
        n_diag=len(recs),
        n_prior_fallback=len(flat) - len(recs))


def run_one(arm, data, candidates, cell, seed, tmpdir, map_source):
    """One (arm, cell, map_source) HBI run. Returns a result row.

    `arm`        which HBI does the REFITTING (legacy FD vs Mod 11 GN)
    `map_source` how the SUPPLIED maps were fitted ("gn" or "fd")

    These are independent: the fork can be handed finite-difference maps,
    and the legacy arm can be handed Gauss-Newton ones. Crossing them is
    what separates "the seed moved" from "the refit fixed it".
    """
    models = [MODELS[m][0] for m in candidates]
    trials = [MODELS[m][1] for m in candidates]
    use_gn = (map_source == "gn")
    tag = f"{arm}_{cell['name']}_{map_source}"

    t0 = time.perf_counter()
    paths = fit_maps(data, candidates, seed, tmpdir, tag, use_gn)
    t_maps = time.perf_counter() - t0

    fname = str(tmpdir / f"hbi_{tag}.pkl")
    t0 = time.perf_counter()
    np.random.seed(seed)
    if arm == "fork_gn":
        res = hbi_fork(data, models, paths, fname, config=dict(HBI_CONFIG),
                       model_trials=trials)
    else:
        # The frozen pre-fork entry point: six arguments, no model_trials.
        res = hbi_legacy(data, models, paths, fname, config=dict(HBI_CONFIG))
    t_hbi = time.perf_counter() - t0

    freq = np.ravel(np.asarray(res.output.model_frequency, dtype=float))
    xp = np.ravel(np.asarray(res.output.exceedance_prob, dtype=float))
    resp = np.asarray(res.output.responsibility, dtype=float)
    # responsibility is (K, N) or (N, K) depending on version; normalise to
    # per-subject assignment of the COMPLEX model (index 1 of candidates).
    if resp.ndim == 2 and resp.shape[0] == len(candidates):
        resp_complex = resp[1, :]
    else:
        resp_complex = resp[:, 1]

    # Final free-energy bound. HBIResult keeps only the LAST iteration's
    # state (the per-iteration `prog` list is local to hbi_run and never
    # returned), so this is the converged value, not a trajectory.
    try:
        bound = float(res.math.bound.bound.L)
    except Exception:
        bound = None

    # Iteration count is likewise not on the result object. It IS written
    # to the .log file that hbi_run opens alongside `fname` — and that
    # happens regardless of `verbose`, because hbi_log writes to the file
    # handle independently of the console flag. Counting the "Iteration"
    # lines is therefore reading HBI's own record, not inferring.
    n_iter = None
    log_path = Path(fname).with_suffix(".log")
    if log_path.exists():
        n_iter = sum(1 for line in log_path.read_text().splitlines()
                     if line.startswith("Iteration"))

    row = dict(
        arm=arm, cell=str(cell["name"]), family=str(cell.get("family")),
        mix=float(cell.get("level", np.nan)),
        n_subjects=int(cell["n_subjects"]), n_trials=int(cell["n_trials"]),
        seed=int(seed), map_source=map_source, candidates=list(candidates),
        freq_complex=float(freq[1]), freq=freq.tolist(),
        xp_complex=float(xp[1]),
        resp_complex=resp_complex.tolist(),
        n_iter=n_iter, bound=bound,
        seconds_maps=t_maps, seconds_hbi=t_hbi,
    )
    row.update(diag_summary(res))
    tm = cell.get("true_model")
    row["true_frac_complex"] = (float(np.mean(tm)) if tm is not None
                                else float("nan"))
    row["true_model"] = (np.asarray(tm).tolist() if tm is not None else None)
    return row


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="hbi")
    ap.add_argument("--arms", nargs="+", default=["cbm_orig", "fork_gn"],
                    choices=["cbm_orig", "fork_gn"])
    ap.add_argument("--map-sources", nargs="+", default=["gn", "fd"],
                    choices=["gn", "fd"],
                    help="how the SUPPLIED individual maps are fitted — "
                         "this is the perturbation axis")
    ap.add_argument("--cells", nargs="+", default=None)
    args = ap.parse_args()

    data_dir = BENCH_DIR / "data" / args.grid
    if not data_dir.exists():
        raise SystemExit(f"no such grid: {data_dir} "
                         f"(run: python benchmark/simulate.py --grid hbi)")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    names = [m["name"] for m in manifest]
    if args.cells:
        names = [n for n in names if n in args.cells]

    out_dir = BENCH_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = out_dir / "_hbi_tmp"
    tmpdir.mkdir(exist_ok=True)

    rows = []
    t_start = time.time()
    total = len(names) * len(args.arms) * len(args.map_sources)
    done = 0

    for name in names:
        cell = load_cell(data_dir / f"{name}.npz")
        data = cell_to_data(cell)
        candidates = FAMILIES[str(cell.get("family"))]
        seed = int(cell["seed"])
        for arm in args.arms:
            for src in args.map_sources:
                with warnings.catch_warnings(record=True) as rec:
                    warnings.simplefilter("always")
                    warnings.filterwarnings("ignore",
                                            category=DeprecationWarning)
                    try:
                        row = run_one(arm, data, candidates, cell, seed,
                                      tmpdir, src)
                    except Exception as e:
                        done += 1
                        print(f"  [{done}/{total}] {name:12s} {arm:9s} "
                              f"maps={src} FAILED: "
                              f"{type(e).__name__}: {e}", flush=True)
                        continue
                row["n_warnings"] = len(rec)
                rows.append(row)
                done += 1
                print(f"  [{done}/{total}] {name:12s} {arm:9s} "
                      f"maps={src}  "
                      f"freq_complex={row['freq_complex']:.4f}  "
                      f"iter={row['n_iter']}  "
                      f"{row['seconds_hbi']:5.1f}s", flush=True)

    out = out_dir / f"hbi_{args.grid}.pkl"
    with open(out, "wb") as f:
        pickle.dump(rows, f)
    print(f"\n{len(rows)} rows -> {out}   "
          f"({time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    main()
