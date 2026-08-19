"""Run the MAP curvature experiment.

The primary scientific comparison is deliberately made at ONE parameter
location: the MAP obtained with the current fork-like L-BFGS-B + GN polish.

At that fixed MAP we compare:
    H_GN = J.T @ J + prior precision
    H_FD = central finite-difference Hessian of the full negative log posterior
    H_AD = autodiff Hessian of the full negative log posterior

We also compare GN and AD Newton polishing as a separate optimization question.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import csv
from pathlib import Path
import time
import numpy as np

from .models import generate_dataset, trial_loglik_np
from .derivatives import (
    negative_log_posterior_np,
    negative_log_posterior_jax,
    finite_difference_gradient,
    finite_difference_hessian,
    gn_curvature,
    autodiff_gradient_hessian,
    compare_hessians,
)
from .optimizers import fit_map


GRAD_STEPS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
FD_HESSIAN_STEPS = (1e-3, 1e-4, 1e-5, 1e-6)


@dataclass
class RunRecord:
    model: str
    seed: int
    n_trials: int

    # Primary GN-polished MAP
    optimization_method: str
    objective_map: float
    gradient_norm_map_fd: float
    optimization_seconds: float
    n_polish_steps: int

    # AD-polish optimization comparison
    ad_polish_objective: float
    ad_polish_seconds: float
    ad_polish_gradient_norm: float
    ad_polish_n_steps: int
    gn_minus_ad_polish_objective: float

    theta_true: str
    theta_map: str

    # NumPy/JAX implementation validation
    numpy_jax_objective_abs_error_map: float
    numpy_jax_objective_abs_error_probe: float
    ad_gradient_norm_probe: float
    grad_h001_rel_error: float
    grad_h0001_rel_error: float
    grad_h00001_rel_error: float
    grad_h000001_rel_error: float
    grad_h0000001_rel_error: float

    # Curvature comparison at exactly the same MAP
    gn_ad_rel_fro_error: float
    fd_ad_rel_fro_error: float

    gn_logdet: float
    fd_logdet: float
    ad_logdet: float

    gn_min_eig: float
    fd_min_eig: float
    ad_min_eig: float

    fd_h001_rel_error: float
    fd_h0001_rel_error: float
    fd_h00001_rel_error: float
    fd_h000001_rel_error: float

    fd_h001_seconds: float
    fd_h0001_seconds: float
    fd_h00001_seconds: float
    fd_h000001_seconds: float
    ad_hessian_seconds: float

    # Curvature contribution to the Laplace approximation.
    # Same MAP and same objective for all three methods.
    laplace_logevidence_gn: float
    laplace_logevidence_fd: float
    laplace_logevidence_ad: float
    laplace_gn_minus_ad: float
    laplace_fd_minus_ad: float


def _sym(H):
    H = np.asarray(H, dtype=float)
    return 0.5 * (H + H.T)


def _logdet(H):
    sign, value = np.linalg.slogdet(_sym(H))
    return float(value) if sign > 0 else np.nan


def _min_eig(H):
    return float(np.linalg.eigvalsh(_sym(H))[0])


def _laplace_logevidence(objective_map, H):
    """Laplace log evidence using the same MAP/objective for each curvature.

    objective_map is the negative log joint/posterior objective used for MAP:
        objective_map = -log p(y, theta_MAP)

    Constants in the normalized Gaussian prior are omitted by the objective
    implementation, but that omission is identical across curvature methods.
    Therefore differences between GN/FD/AD here are exact curvature-only
    differences:
        Delta = -0.5 * Delta log(det(H)).
    """
    H = _sym(H)
    sign, logdet = np.linalg.slogdet(H)
    if sign <= 0:
        return np.nan
    d = H.shape[0]
    return float(
        -objective_map
        + 0.5 * d * np.log(2.0 * np.pi)
        - 0.5 * logdet
    )


def _gradient_probe(theta_map, bounds):
    """Construct a deterministic interior point away from the MAP.

    Relative gradient error is ill-conditioned at the MAP because the true
    gradient should be close to zero. We therefore validate NumPy FD gradients
    against AD at a nearby but clearly nonstationary interior point.
    """
    theta_map = np.asarray(theta_map, dtype=float)
    lower = np.asarray([b[0] for b in bounds], dtype=float)
    upper = np.asarray([b[1] for b in bounds], dtype=float)
    width = upper - lower

    signs = np.where(np.arange(theta_map.size) % 2 == 0, 1.0, -1.0)
    probe = theta_map + 0.15 * signs * width

    margin = 0.10 * width
    return np.clip(probe, lower + margin, upper - margin)


def _relative_error(x, ref):
    x = np.asarray(x, dtype=float)
    ref = np.asarray(ref, dtype=float)
    denom = max(np.linalg.norm(ref), np.finfo(float).eps)
    return float(np.linalg.norm(x - ref) / denom)


def run_one(
    model,
    seed,
    n_trials=250,
    n_starts=5,
    fd_steps=FD_HESSIAN_STEPS,
    grad_steps=GRAD_STEPS,
):
    rng = np.random.default_rng(seed)
    dataset = generate_dataset(model, rng, n_trials=n_trials)

    data = dataset.data
    bounds = dataset.bounds
    prior_mean = dataset.prior_mean
    prior_precision = dataset.prior_precision

    objective = lambda theta: negative_log_posterior_np(
        theta, model, data, prior_mean, prior_precision
    )
    trial_func = lambda theta: trial_loglik_np(model, theta, data)

    # ------------------------------------------------------------------
    # 1. Primary MAP: current fork-like L-BFGS-B + GN polish.
    # ------------------------------------------------------------------
    fit = fit_map(
        objective,
        bounds,
        rng,
        polish="gn",
        trial_func=trial_func,
        prior_precision=prior_precision,
        n_starts=n_starts,
    )
    theta_map = fit.theta

    # ------------------------------------------------------------------
    # 2. JAX setup and NumPy/JAX objective validation.
    # ------------------------------------------------------------------
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
    except ImportError as exc:
        raise ImportError(
            "JAX is required for this experiment. Install with `pip install jax`."
        ) from exc

    objective_jax = lambda theta: negative_log_posterior_jax(
        theta, model, data, prior_mean, prior_precision
    )

    obj_np_map = float(objective(theta_map))
    obj_ad_map = float(objective_jax(theta_map))
    objective_error_map = abs(obj_np_map - obj_ad_map)

    # Validate gradients away from the stationary point.
    theta_probe = _gradient_probe(theta_map, bounds)
    obj_np_probe = float(objective(theta_probe))
    obj_ad_probe = float(objective_jax(theta_probe))
    objective_error_probe = abs(obj_np_probe - obj_ad_probe)

    g_ad_probe, _ = autodiff_gradient_hessian(
        theta_probe, model, data, prior_mean, prior_precision
    )

    gradient_errors = {}
    for eps in grad_steps:
        g_fd_probe = finite_difference_gradient(objective, theta_probe, eps=eps)
        gradient_errors[eps] = _relative_error(g_fd_probe, g_ad_probe)

    # ------------------------------------------------------------------
    # 3. Curvatures at EXACTLY the same GN-polished MAP.
    # ------------------------------------------------------------------
    ad_hessian_t0 = time.perf_counter()
    g_ad_map, H_ad = autodiff_gradient_hessian(
        theta_map, model, data, prior_mean, prior_precision
    )
    ad_hessian_seconds = time.perf_counter() - ad_hessian_t0

    H_gn = gn_curvature(
        theta_map,
        trial_func,
        prior_precision=prior_precision,
    )

    fd_results = {}
    fd_errors = {}
    for eps in fd_steps:
        t0 = time.perf_counter()
        H_fd = finite_difference_hessian(objective, theta_map, eps=eps)
        seconds = time.perf_counter() - t0
        fd_results[eps] = (H_fd, seconds)
        fd_errors[eps] = _relative_error(H_fd, H_ad)

    gn_error = compare_hessians(H_gn, H_ad)["relative_frobenius_error"]

    # h=1e-4 is retained as the conventional FD summary, but all step sizes
    # are stored so stability can be checked explicitly.
    H_fd_summary = fd_results[1e-4][0]

    # ------------------------------------------------------------------
    # 4. Curvature-only effect on the Laplace term at the SAME MAP.
    # ------------------------------------------------------------------
    le_gn = _laplace_logevidence(fit.objective, H_gn)
    le_fd = _laplace_logevidence(fit.objective, H_fd_summary)
    le_ad = _laplace_logevidence(fit.objective, H_ad)

    # ------------------------------------------------------------------
    # 5. Separate optimization question: AD Hessian Newton polish.
    # Use the same random seed so L-BFGS-B starts match the GN comparison.
    # ------------------------------------------------------------------
    rng_ad = np.random.default_rng(seed)
    fit_ad = fit_map(
        objective,
        bounds,
        rng_ad,
        polish="ad",
        prior_precision=prior_precision,
        model=model,
        data=data,
        prior_mean=prior_mean,
        n_starts=n_starts,
    )

    g_map_fd = finite_difference_gradient(objective, theta_map, eps=1e-6)

    return {
        "fit_gn": fit,
        "fit_ad": fit_ad,
        "theta_true": dataset.theta_true,
        "theta_probe": theta_probe,
        "gradient_norm_map_fd": float(np.linalg.norm(g_map_fd)),
        "H_gn": H_gn,
        "H_ad": H_ad,
        "H_fd": {eps: value[0] for eps, value in fd_results.items()},
        "H_fd_seconds": {eps: value[1] for eps, value in fd_results.items()},
        "ad_hessian_seconds": ad_hessian_seconds,
        "gn_ad_error": gn_error,
        "fd_ad_errors": fd_errors,
        "objective_jax_error_map": objective_error_map,
        "objective_jax_error_probe": objective_error_probe,
        "gradient_errors": gradient_errors,
        "ad_gradient_norm_probe": float(np.linalg.norm(g_ad_probe)),
        "laplace_logevidence_gn": le_gn,
        "laplace_logevidence_fd": le_fd,
        "laplace_logevidence_ad": le_ad,
        "dataset": dataset,
    }


def run_experiment(
    models=("binary", "categorical", "ces"),
    n_datasets=20,
    n_trials=250,
    n_starts=5,
    seed=123,
    output_dir="results",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    matrices = {}

    for model in models:
        print(f"\n=== {model} ===")

        for rep in range(n_datasets):
            run_seed = seed + rep
            print(
                f"  dataset {rep + 1}/{n_datasets} (seed={run_seed})",
                flush=True,
            )

            result = run_one(
                model=model,
                seed=run_seed,
                n_trials=n_trials,
                n_starts=n_starts,
            )

            fit = result["fit_gn"]
            fit_ad = result["fit_ad"]
            H_gn = result["H_gn"]
            H_ad = result["H_ad"]
            fd = result["H_fd"]
            fd_err = result["fd_ad_errors"]
            ge = result["gradient_errors"]

            row = RunRecord(
                model=model,
                seed=run_seed,
                n_trials=n_trials,
                optimization_method=fit.method,
                objective_map=fit.objective,
                gradient_norm_map_fd=result["gradient_norm_map_fd"],
                optimization_seconds=fit.seconds,
                n_polish_steps=fit.n_polish_steps,
                ad_polish_objective=fit_ad.objective,
                ad_polish_seconds=fit_ad.seconds,
                ad_polish_gradient_norm=fit_ad.gradient_norm,
                ad_polish_n_steps=fit_ad.n_polish_steps,
                gn_minus_ad_polish_objective=fit.objective - fit_ad.objective,
                theta_true=np.array2string(result["theta_true"], precision=6),
                theta_map=np.array2string(fit.theta, precision=6),
                numpy_jax_objective_abs_error_map=result["objective_jax_error_map"],
                numpy_jax_objective_abs_error_probe=result["objective_jax_error_probe"],
                ad_gradient_norm_probe=result["ad_gradient_norm_probe"],
                grad_h001_rel_error=ge[1e-3],
                grad_h0001_rel_error=ge[1e-4],
                grad_h00001_rel_error=ge[1e-5],
                grad_h000001_rel_error=ge[1e-6],
                grad_h0000001_rel_error=ge[1e-7],
                gn_ad_rel_fro_error=result["gn_ad_error"],
                fd_ad_rel_fro_error=fd_err[1e-4],
                gn_logdet=_logdet(H_gn),
                fd_logdet=_logdet(fd[1e-4]),
                ad_logdet=_logdet(H_ad),
                gn_min_eig=_min_eig(H_gn),
                fd_min_eig=_min_eig(fd[1e-4]),
                ad_min_eig=_min_eig(H_ad),
                fd_h001_rel_error=fd_err[1e-3],
                fd_h0001_rel_error=fd_err[1e-4],
                fd_h00001_rel_error=fd_err[1e-5],
                fd_h000001_rel_error=fd_err[1e-6],
                fd_h001_seconds=result["H_fd_seconds"][1e-3],
                fd_h0001_seconds=result["H_fd_seconds"][1e-4],
                fd_h00001_seconds=result["H_fd_seconds"][1e-5],
                fd_h000001_seconds=result["H_fd_seconds"][1e-6],
                ad_hessian_seconds=result["ad_hessian_seconds"],
                laplace_logevidence_gn=result["laplace_logevidence_gn"],
                laplace_logevidence_fd=result["laplace_logevidence_fd"],
                laplace_logevidence_ad=result["laplace_logevidence_ad"],
                laplace_gn_minus_ad=(
                    result["laplace_logevidence_gn"]
                    - result["laplace_logevidence_ad"]
                ),
                laplace_fd_minus_ad=(
                    result["laplace_logevidence_fd"]
                    - result["laplace_logevidence_ad"]
                ),
            )
            rows.append(asdict(row))

            key = f"{model}_{run_seed}"
            matrices[f"{key}_H_gn"] = H_gn
            matrices[f"{key}_H_ad"] = H_ad
            matrices[f"{key}_theta_map"] = fit.theta
            matrices[f"{key}_theta_probe"] = result["theta_probe"]
            for eps, H in fd.items():
                tag = str(eps).replace(".", "p").replace("-", "m")
                matrices[f"{key}_H_fd_{tag}"] = H

            print(
                f"    GN obj={fit.objective:.8g}; "
                f"AD-polish obj={fit_ad.objective:.8g}; "
                f"GN/AD Hess err={result['gn_ad_error']:.3g}; "
                f"Laplace GN-AD={row.laplace_gn_minus_ad:+.4g}"
            )

    csv_path = output_dir / "map_curvature_results.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    npz_path = output_dir / "map_curvature_matrices.npz"
    np.savez(npz_path, **matrices)

    print(f"\nSaved:\n  {csv_path}\n  {npz_path}")
    return rows
