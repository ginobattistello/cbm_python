from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def summarize(df):
    groups=["model","param1_name","param2_name","param1","param2","n_trials","sigma","prior_scale"]
    rows=[]
    for keys,d in df.groupby(groups,dropna=False):
        model,p1n,p2n,p1,p2,T,sigma,ps=keys
        rows.append({
            "model":model,"param1_name":p1n,"param2_name":p2n,
            "param1":p1,"param2":p2,"n_trials":T,"sigma":sigma,"prior_scale":ps,
            "n":len(d),
            "original_pd_rate":d.original_pd.mean(),
            "fd_pd_rate":d.fd_pd.mean(),
            "ad_pd_rate":d.ad_pd.mean(),
            "original_numerical_failure_rate":d.original_numerical_failure.mean(),
            "fd_numerical_failure_rate":d.fd_numerical_failure.mean(),
            "structural_laplace_failure_rate":d.structural_laplace_failure.mean(),
            "median_original_ad_rel_error":np.nanmedian(d.original_ad_rel_error),
            "median_fd_ad_rel_error":np.nanmedian(d.fd_ad_rel_error),
            "median_original_min_eig":np.nanmedian(d.original_min_eig),
            "median_fd_min_eig":np.nanmedian(d.fd_min_eig),
            "median_ad_min_eig":np.nanmedian(d.ad_min_eig),
            "median_original_log10_condition":np.nanmedian(
                d.original_log10_condition.replace([np.inf,-np.inf],np.nan)
            ),
            "median_fd_log10_condition":np.nanmedian(
                d.fd_log10_condition.replace([np.inf,-np.inf],np.nan)
            ),
            "median_ad_log10_condition":np.nanmedian(
                d.ad_log10_condition.replace([np.inf,-np.inf],np.nan)
            ),
            "median_abs_delta_loge_original_ad":np.nanmedian(np.abs(d.delta_loge_original_ad)),
            "median_abs_delta_loge_fd_ad":np.nanmedian(np.abs(d.delta_loge_fd_ad)),
        })
    return pd.DataFrame(rows)

def save_heatmap_tables(summary,out):
    metrics=[
        "original_numerical_failure_rate",
        "fd_numerical_failure_rate",
        "structural_laplace_failure_rate",
        "median_original_ad_rel_error",
        "median_fd_ad_rel_error",
        "median_abs_delta_loge_original_ad",
        "median_abs_delta_loge_fd_ad",
        "median_ad_log10_condition",
    ]
    for model,d in summary.groupby("model"):
        for metric in metrics:
            d.pivot(index="param1",columns="param2",values=metric).to_csv(
                out/f"heatmap_{model}_{metric}.csv"
            )

def print_summary(summary):
    print("\n"+"="*84)
    print("MAP CURVATURE STRESS — PARAMETER GRID")
    print("="*84)
    for model,d in summary.groupby("model"):
        print(f"\nMODEL: {model}")
        print("-"*84)

        print("\nHighest original-CBM numerical failure rates")
        for _,r in d.sort_values("original_numerical_failure_rate",ascending=False).head(5).iterrows():
            print(
                f"  {r.param1_name}={r.param1:g}, {r.param2_name}={r.param2:g}: "
                f"{100*r.original_numerical_failure_rate:.1f}% "
                f"(AD PD={100*r.ad_pd_rate:.1f}%)"
            )

        print("\nLargest original-CBM Hessian errors")
        for _,r in d.sort_values("median_original_ad_rel_error",ascending=False).head(5).iterrows():
            print(
                f"  {r.param1_name}={r.param1:g}, {r.param2_name}={r.param2:g}: "
                f"error={r.median_original_ad_rel_error:.3e}, "
                f"|ΔlogE|={r.median_abs_delta_loge_original_ad:.4f}"
            )

        print("\nLargest original-CBM evidence differences")
        for _,r in d.sort_values("median_abs_delta_loge_original_ad",ascending=False).head(5).iterrows():
            print(
                f"  {r.param1_name}={r.param1:g}, {r.param2_name}={r.param2:g}: "
                f"|ΔlogE|={r.median_abs_delta_loge_original_ad:.4f}, "
                f"new FD={r.median_abs_delta_loge_fd_ad:.3e}"
            )

        w=d.n
        print("\nOverall rates")
        print(f"  original numerical failure: {100*np.average(d.original_numerical_failure_rate,weights=w):.2f}%")
        print(f"  new FD numerical failure:   {100*np.average(d.fd_numerical_failure_rate,weights=w):.2f}%")
        print(f"  structural AD failure:      {100*np.average(d.structural_laplace_failure_rate,weights=w):.2f}%")

    print("\nDefinitions:")
    print("  original numerical failure = original non-PD AND AD PD")
    print("  new FD numerical failure   = central-FD non-PD AND AD PD")
    print("  structural failure         = AD non-PD")
    print("="*84)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--path",default="results/map_curvature_stress/stress_raw.csv")
    ap.add_argument("--output-dir",default="results/map_curvature_stress")
    args=ap.parse_args()

    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)

    df=pd.read_csv(args.path)
    s=summarize(df)
    s.to_csv(out/"stress_grid_summary.csv",index=False)
    save_heatmap_tables(s,out)
    print_summary(s)
    print(f"\nSaved: {out/'stress_grid_summary.csv'}")
    print("Heatmap-ready CSV files were also created.")

if __name__=="__main__":
    main()
