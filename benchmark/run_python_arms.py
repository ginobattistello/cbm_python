"""
Run the two Python arms of the benchmark over the simulated grid.

Arms
----
fork_gn   this fork (cbm/), Gauss-Newton curvature — `model_trials`
          passed, so the Hessian/evidence come from Mod 5.
cbm_orig  the pristine pre-fork CBM vendored at
          benchmark/external/cbm_original (commit e72193f). Its
          `individual_fit` has no `model_trials` parameter at all —
          that missing capability IS the difference under test.

Both arms fit the SAME objective (benchmark/models.py) on the SAME data
(benchmark/data/), with the same prior and the same Config, so any
divergence is attributable to the optimizer/curvature.

Every cell is fitted with BOTH candidate models (RL and RL2), not just
the generating one — that is what makes the model-recovery confusion
matrix possible.

Output: benchmark/results/python_<grid>.pkl, one row per
(arm, cell, fitted_model, subject), plus per-fit diagnostics where the
arm provides them.

Usage
-----
    python benchmark/run_python_arms.py --grid quick
    python benchmark/run_python_arms.py --grid grid --arms fork_gn cbm_orig
"""

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(REPO_ROOT))                      # this fork
sys.path.insert(0, str(BENCH_DIR / "external"))         # cbm_original
sys.path.insert(0, str(BENCH_DIR))                      # models.py

from models import MODELS, PRIOR_VARIANCE, to_native, FAMILIES  # noqa: E402

# Fitting config shared by both arms. num_init=5 gives the
# multimodality diagnostic something to measure while staying fast;
# verbose off or the grid drowns in output.
FIT_CONFIG = dict(num_init=5, verbose=False)


def load_cell(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def cell_family(cell):
    """'rl' or 'value'. Older cells predate the field — infer from keys."""
    fam = cell.get("family")
    if fam is not None:
        return str(fam)
    return "value" if "sure" in cell else "rl"


def cell_to_data(cell):
    """Per-subject data tuples in the format that family's models expect."""
    n = int(cell["n_subjects"])
    if cell_family(cell) == "value":
        return [(cell["sure"][i], cell["gamble"][i],
                 cell["prob"][i], cell["chose"][i]) for i in range(n)]
    return [(cell["choices"][i], cell["rewards"][i]) for i in range(n)]


def fit_fork(data, model_name, seed):
    """This fork, Gauss-Newton path (model_trials supplied)."""
    from cbm.individual_fit import individual_fit
    model, model_trials, d, prior_mean = MODELS[model_name]
    np.random.seed(seed)
    t0 = time.perf_counter()
    fit = individual_fit(data, model, prior_mean, PRIOR_VARIANCE,
                         config=dict(FIT_CONFIG), model_trials=model_trials)
    elapsed = time.perf_counter() - t0
    diags = getattr(fit.math, "diagnostics", None)
    return fit, elapsed, diags


def fit_cbm_orig(data, model_name, seed):
    """Pristine pre-fork CBM. No model_trials parameter exists.

    KNOWN BUG in this arm (cbm_original/optimization.py:394): if every
    initialization returns a non-finite objective, `best_result` is never
    assigned and the code raises
    `AttributeError: 'NoneType' object has no attribute 'x'`.
    It happens because `result.f < best_f` is always False when result.f
    is NaN, and the retry loop has no guard. The fork does not have this
    problem — MOD 4a wraps the objective so non-finite values become a
    large finite penalty.

    Fitting subject-by-subject so ONE pathological subject costs one
    subject, not the whole cell. Without this the entire CBM arm would
    vanish from any figure containing such a subject, which would
    misrepresent the comparison — the bug is worth reporting precisely,
    not worth deleting the arm over. Failed subjects come back as NaN
    and are counted in the report.
    """
    from cbm_original.individual_fit import individual_fit as orig_fit
    model, _unused, d, prior_mean = MODELS[model_name]
    t0 = time.perf_counter()
    per_subject, n_failed = [], 0
    for i, dat in enumerate(data):
        np.random.seed(seed + i)
        try:
            per_subject.append(orig_fit([dat], model, prior_mean,
                                        PRIOR_VARIANCE, config=dict(FIT_CONFIG)))
        except Exception:
            per_subject.append(None)
            n_failed += 1
    elapsed = time.perf_counter() - t0
    return _merge_single_fits(per_subject, d), elapsed, None


class _MergedFit:
    """Minimal stand-in for a FitResult assembled from per-subject fits."""
    def __init__(self, math_, output):
        self.math, self.output = math_, output


def _merge_single_fits(fits, d):
    """Stack per-subject FitResults into one, NaN-filling the failures."""
    n = len(fits)
    params = np.full((n, d), np.nan)
    lme = np.full(n, np.nan)
    loglik = np.full(n, np.nan)
    ldh = np.full(n, np.nan)
    flag = np.full(n, np.nan)
    for i, f in enumerate(fits):
        if f is None:
            continue
        params[i] = f.output.parameters[0]
        lme[i] = f.output.log_evidence[0]
        loglik[i] = f.math.loglik[0]
        ldh[i] = f.math.log_det_hessian[0]
        flag[i] = f.math.flag[0]
    math_ = SimpleNamespace(loglik=loglik, log_det_hessian=ldh, flag=flag,
                            diagnostics=None)
    output = SimpleNamespace(parameters=params, log_evidence=lme)
    return _MergedFit(math_, output)


ARMS = {"fork_gn": fit_fork, "cbm_orig": fit_cbm_orig}


def rows_from_fit(arm, cell, model_name, fit, elapsed, diags, n_warnings):
    """Flatten one (arm, cell, model) fit into per-subject records."""
    n_sub = int(cell["n_subjects"])
    fam = cell_family(cell)
    rows = []
    for i in range(n_sub):
        theta = fit.output.parameters[i]
        # A NaN row means this arm failed on this subject (see
        # fit_cbm_orig). Record it as NaN rather than dropping it, so the
        # failure is visible in the report instead of silently shrinking n.
        if np.all(np.isnan(theta)):
            nat = {k: np.nan for k in ("alpha_pos", "alpha_neg", "beta", "rho")}
        else:
            nat = to_native(theta, model_name)
        row = dict(
            arm=arm, cell=str(cell["name"]), generator=str(cell["generator"]),
            family=fam, design=str(cell.get("design", "selection")),
            n_trials=int(cell["n_trials"]), beta_cond=float(cell["beta"]),
            fitted_model=model_name, subject=i,
            # ground truth (family-specific keys are filled below)
            true_beta=float(cell["true_beta"][i]),
            # estimates (native space)
            est_beta=float(nat["beta"]),
            theta=np.asarray(theta, dtype=float).tolist(),
            # evidence + fit quality
            log_evidence=float(fit.output.log_evidence[i]),
            loglik=float(fit.math.loglik[i]),
            log_det_hessian=float(fit.math.log_det_hessian[i]),
            flag=float(fit.math.flag[i]),
            seconds_per_cell=elapsed, n_warnings=n_warnings,
        )
        if fam == "value":
            row.update(true_rho=float(cell["true_rho"][i]),
                       est_rho=float(nat["rho"]),
                       frac_gamble=float(cell["frac_gamble"][i]))
        else:
            row.update(true_alpha_pos=float(cell["true_alpha_pos"][i]),
                       true_alpha_neg=float(cell["true_alpha_neg"][i]),
                       est_alpha_pos=float(nat["alpha_pos"]),
                       est_alpha_neg=float(nat["alpha_neg"]),
                       frac_choice0=float(cell["frac_choice0"][i]))
        if diags is not None and diags[i] is not None:
            dg = diags[i]
            row.update(
                convergence_status=dg.convergence_status,
                hess_method=dg.hess_method,
                abs_grad=dg.abs_grad,
                hess_raw_min_eig=dg.hess_raw_min_eig,
                hess_n_clipped=dg.hess_n_clipped,
                n_inits_agreeing=dg.n_inits_agreeing,
                at_hard_bounds=bool(np.any(dg.at_hard_bounds))
                if dg.at_hard_bounds is not None else None,
                # Mod 10 — getattr so results produced before it existed
                # still load (older pickles have no such field).
                weak_identifiability=getattr(dg, "weak_identifiability", None),
            )
        else:
            row.update(convergence_status=None, hess_method=None,
                       abs_grad=None, hess_raw_min_eig=None,
                       hess_n_clipped=None, n_inits_agreeing=None,
                       at_hard_bounds=None, weak_identifiability=None)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="quick", help="subdir of benchmark/data")
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    choices=list(ARMS))
    ap.add_argument("--cells", nargs="+", default=None,
                    help="only these cell names")
    args = ap.parse_args()

    data_dir = BENCH_DIR / "data" / args.grid
    if not data_dir.exists():
        raise SystemExit(f"no such grid: {data_dir} "
                         f"(run benchmark/simulate.py first)")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    names = [m["name"] for m in manifest]
    if args.cells:
        names = [n for n in names if n in args.cells]

    out_dir = BENCH_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t_start = time.time()

    for ci, name in enumerate(names, 1):
        cell = load_cell(data_dir / f"{name}.npz")
        data = cell_to_data(cell)
        candidates = FAMILIES[cell_family(cell)]
        for arm in args.arms:
            for model_name in candidates:
                # Same seed per (cell, model) across arms: the random
                # initializations are then identical, so the arms are
                # not separated by luck of the draw.
                seed = int(cell["seed"]) + candidates.index(model_name)
                with warnings.catch_warnings(record=True) as rec:
                    warnings.simplefilter("always")
                    # scipy emits one DeprecationWarning per L-BFGS-B call
                    # about `disp`/`iprint` (both arms pass it, equally).
                    # Silence it so n_warnings counts only substantive
                    # alerts — Mod 9's hard-bound / singular-Hessian ones.
                    warnings.filterwarnings(
                        "ignore", category=DeprecationWarning,
                        module="scipy.optimize")
                    try:
                        fit, elapsed, diags = ARMS[arm](data, model_name, seed)
                    except Exception as e:
                        print(f"  [{ci}/{len(names)}] {name:24s} {arm:9s} "
                              f"{model_name:4s} FAILED: {type(e).__name__}: {e}")
                        continue
                rows.extend(rows_from_fit(arm, cell, model_name, fit,
                                          elapsed, diags, len(rec)))
                print(f"  [{ci}/{len(names)}] {name:24s} {arm:9s} "
                      f"{model_name:4s} {elapsed:6.2f}s  "
                      f"sum(lme)={fit.output.log_evidence.sum():10.2f}  "
                      f"warn={len(rec)}", flush=True)

    out = out_dir / f"python_{args.grid}.pkl"
    with open(out, "wb") as f:
        pickle.dump(rows, f)
    print(f"\n{len(rows)} rows -> {out}   ({time.time() - t_start:.1f}s total)")


if __name__ == "__main__":
    main()
