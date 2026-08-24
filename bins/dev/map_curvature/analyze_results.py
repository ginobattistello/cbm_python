"""Print the quantities needed to decide how to update the MAP optimizer."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def _median_range(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    return np.median(x), np.min(x), np.max(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="results/map_curvature/map_curvature_results.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.path)

    print("\n" + "=" * 74)
    print("MAP CURVATURE EXPERIMENT — DECISION SUMMARY")
    print("=" * 74)

    for model, d in df.groupby("model"):
        print(f"\nMODEL: {model}")
        print("-" * 74)

        # 1. NumPy/JAX consistency
        print("\n[1] NumPy vs JAX model consistency")
        print(
            "objective error at MAP (max):   "
            f"{d.numpy_jax_objective_abs_error_map.max():.3e}"
        )
        print(
            "objective error at probe (max): "
            f"{d.numpy_jax_objective_abs_error_probe.max():.3e}"
        )
        print(
            "AD gradient norm at probe (median): "
            f"{d.ad_gradient_norm_probe.median():.3e}"
        )

        # 2. Gradient finite-difference stability away from MAP
        print("\n[2] FD gradient vs AD gradient at an interior probe")
        for label, col in [
            ("1e-3", "grad_h001_rel_error"),
            ("1e-4", "grad_h0001_rel_error"),
            ("1e-5", "grad_h00001_rel_error"),
            ("1e-6", "grad_h000001_rel_error"),
            ("1e-7", "grad_h0000001_rel_error"),
        ]:
            med, lo, hi = _median_range(d[col])
            print(
                f"h={label:<4}: median={med:.3e} "
                f"(range {lo:.3e} - {hi:.3e})"
            )

        # 3. GN vs observed AD Hessian
        print("\n[3] GN/score-outer-product curvature vs AD observed Hessian")
        med, lo, hi = _median_range(d.gn_ad_rel_fro_error)
        print(
            f"relative Frobenius error: median={med:.3f} "
            f"(range {lo:.3f} - {hi:.3f})"
        )

        # 4. FD Hessian validation
        print("\n[4] FD observed Hessian vs AD observed Hessian")
        for label, col in [
            ("1e-3", "fd_h001_rel_error"),
            ("1e-4", "fd_h0001_rel_error"),
            ("1e-5", "fd_h00001_rel_error"),
            ("1e-6", "fd_h000001_rel_error"),
        ]:
            med, lo, hi = _median_range(d[col])
            print(
                f"h={label:<4}: median={med:.3e} "
                f"(range {lo:.3e} - {hi:.3e})"
            )

        # 5. Original CBM evidence Hessian vs new central FD / AD
        print("\n[5] Original CBM evidence Hessian vs new observed Hessian")
        med, lo, hi = _median_range(d.original_ad_rel_fro_error)
        print(
            f"Original-CBM FD vs AD error: median={med:.3e} "
            f"(range {lo:.3e} - {hi:.3e})"
        )
        med2, lo2, hi2 = _median_range(d.original_fd_rel_fro_error)
        print(
            f"Original-CBM FD vs new central-FD error: median={med2:.3e} "
            f"(range {lo2:.3e} - {hi2:.3e})"
        )

        pd_rate = 100.0 * np.mean(d.original_is_pd.astype(bool))
        print(f"Original-CBM Hessian positive-definite rate: {pd_rate:.1f}%")

        orig_delta = d.laplace_original_minus_ad
        med3, lo3, hi3 = _median_range(orig_delta)
        print(
            f"logE_original - logE_AD: median={med3:+.6f} "
            f"(range {lo3:+.6f} - {hi3:+.6f})"
        )
        print(
            "median |original - AD logE|: "
            f"{np.nanmedian(np.abs(orig_delta)):.6f}"
        )

        orig_fd_delta = d.laplace_original_minus_fd
        print(
            "median |original - new central-FD logE|: "
            f"{np.nanmedian(np.abs(orig_fd_delta)):.6f}"
        )

        # 6. Curvature properties
        print("\n[6] Curvature at the same MAP")
        print(
            f"minimum eigenvalue, GN median: {d.gn_min_eig.median():.6g}"
        )
        print(
            f"minimum eigenvalue, AD median: {d.ad_min_eig.median():.6g}"
        )
        logdet_diff = d.gn_logdet - d.ad_logdet
        med, lo, hi = _median_range(logdet_diff)
        print(
            f"logdet(GN)-logdet(AD): median={med:+.6f} "
            f"(range {lo:+.6f} - {hi:+.6f})"
        )
        print(
            "median |logdet difference|: "
            f"{np.nanmedian(np.abs(logdet_diff)):.6f}"
        )

        # 6. Curvature-only Laplace consequence
        print("\n[7] Curvature-only Laplace consequence at the SAME MAP")
        delta = d.laplace_gn_minus_ad
        med, lo, hi = _median_range(delta)
        print(
            f"logE_GN - logE_AD: median={med:+.6f} "
            f"(range {lo:+.6f} - {hi:+.6f})"
        )
        print(
            "median |log-evidence difference|: "
            f"{np.nanmedian(np.abs(delta)):.6f}"
        )
        fd_delta = d.laplace_fd_minus_ad
        print(
            "FD(h=1e-4) - AD median |logE difference|: "
            f"{np.nanmedian(np.abs(fd_delta)):.3e}"
        )

        # 7. Optimization comparison
        print("\n[8] GN vs AD Newton polish")
        obj_diff = d.gn_minus_ad_polish_objective
        med, lo, hi = _median_range(obj_diff)
        print(
            f"GN objective - AD objective: median={med:.3e} "
            f"(range {lo:.3e} - {hi:.3e})"
        )
        print(
            f"GN optimization time median: {d.optimization_seconds.median():.4f} s"
        )
        print(
            f"AD optimization time median: {d.ad_polish_seconds.median():.4f} s"
        )
        if d.optimization_seconds.median() > 0:
            print(
                "AD/GN time ratio: "
                f"{d.ad_polish_seconds.median() / d.optimization_seconds.median():.2f}x"
            )
        print(
            f"GN polish steps median: {d.n_polish_steps.median():.1f}; "
            f"AD polish steps median: {d.ad_polish_n_steps.median():.1f}"
        )

        # 8. Data-driven decision hints, deliberately conservative.
        print("\n[9] What this model says about the toolbox")
        curvature_err = np.nanmedian(d.gn_ad_rel_fro_error)
        map_diff = np.nanmedian(np.abs(obj_diff))
        fd_ad = np.nanmedian(d.fd_h0001_rel_error)
        laplace_abs = np.nanmedian(np.abs(delta))

        if map_diff < 1e-8:
            print("- MAP: no detectable benefit from replacing GN polish with AD.")
        else:
            print("- MAP: AD and GN polishing reach measurably different objectives.")

        if fd_ad < 1e-4:
            print("- Observed Hessian: central FD is essentially identical to AD here.")
        elif fd_ad < 1e-2:
            print("- Observed Hessian: FD is close to AD, but not numerically identical.")
        else:
            print("- Observed Hessian: FD accuracy is questionable for this model.")

        orig_ad = np.nanmedian(d.original_ad_rel_fro_error)
        orig_loge = np.nanmedian(np.abs(d.laplace_original_minus_ad))
        orig_pd_rate = np.mean(d.original_is_pd.astype(bool))

        if orig_ad < 1e-4:
            print("- Original CBM Hessian: numerically equivalent to AD for this model.")
        else:
            print(
                "- Original CBM Hessian: differs measurably from the validated "
                "observed Hessian."
            )

        if orig_pd_rate < 1.0:
            print(
                f"- Original CBM PD check: failed in {(1.0-orig_pd_rate)*100:.1f}% "
                "of simulated datasets at this common MAP."
            )

        print(
            f"- Original-vs-AD curvature-only evidence discrepancy: "
            f"median |ΔlogE|={orig_loge:.4f}."
        )

        if curvature_err >= 0.05:
            print(
                "- GN curvature: it should not be treated as the exact observed "
                "posterior Hessian."
            )
        else:
            print("- GN curvature: close to the observed Hessian in this experiment.")

        if laplace_abs < 0.05:
            print("- Laplace impact: small for these datasets.")
        elif laplace_abs < 0.5:
            print(
                "- Laplace impact: modest per subject, but potentially cumulative "
                "in model comparison/HBI."
            )
        else:
            print("- Laplace impact: large enough to be immediately consequential.")

    print("\n" + "=" * 74)
    print("CROSS-MODEL DECISION")
    print("=" * 74)

    map_abs = np.nanmedian(np.abs(df.gn_minus_ad_polish_objective))
    fd_ad = np.nanmedian(df.fd_h0001_rel_error)
    gn_ad = np.nanmedian(df.gn_ad_rel_fro_error)
    original_ad = np.nanmedian(df.original_ad_rel_fro_error)
    lap_abs = np.nanmedian(np.abs(df.laplace_gn_minus_ad))
    original_lap_abs = np.nanmedian(np.abs(df.laplace_original_minus_ad))
    original_pd_rate = np.mean(df.original_is_pd.astype(bool))

    print(f"median |GN-AD MAP objective difference|: {map_abs:.3e}")
    print(f"median FD(h=1e-4) vs AD Hessian error: {fd_ad:.3e}")
    print(f"median GN vs AD Hessian error: {gn_ad:.3f}")
    print(f"median original-CBM FD vs AD Hessian error: {original_ad:.3e}")
    print(f"original-CBM Hessian PD rate: {100*original_pd_rate:.1f}%")
    print(f"median |GN-AD curvature-only logE difference|: {lap_abs:.4f}")
    print(
        "median |original-CBM - AD curvature-only logE difference|: "
        f"{original_lap_abs:.4f}"
    )

    print("\nSuggested interpretation:")
    if map_abs < 1e-8:
        print(
            "1. Keep the current GN polish as the MAP optimization method "
            "unless other stress tests show failures."
        )
    else:
        print(
            "1. Reconsider the optimization curvature because AD reaches "
            "different MAP objectives."
        )

    if fd_ad < 1e-4:
        print(
            "2. A separate central-FD observed Hessian at the final MAP is a "
            "strong default candidate; AD is an excellent validation/optional backend."
        )
    else:
        print(
            "2. Prefer AD for the post-MAP observed Hessian unless FD can be "
            "stabilized further."
        )

    if gn_ad >= 0.05:
        print(
            "3. Do not label J.T @ J + prior precision as the observed Hessian; "
            "keep optimization curvature and post-MAP curvature separate."
        )

    print(
        "4. Compare original-CBM nested forward FD directly with the validated "
        "central-FD/AD curvature. This isolates whether the evidence-Hessian "
        "estimator itself changes log evidence on identical data and MAPs."
    )
    print(
        "5. Use the reported curvature-only log-evidence differences to decide "
        "whether the production CBM evidence calculation should switch to the "
        "validated central-FD observed Hessian."
    )

    print("=" * 74)


if __name__ == "__main__":
    main()
