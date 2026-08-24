from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_METRICS=[
    "original_numerical_failure_rate",
    "median_original_ad_rel_error",
    "median_abs_delta_loge_original_ad",
]

def plot_one(d,metric,out):
    model=d.model.iloc[0]
    p1=d.param1_name.iloc[0]
    p2=d.param2_name.iloc[0]
    pivot=d.pivot(index="param1",columns="param2",values=metric).sort_index().sort_index(axis=1)

    fig,ax=plt.subplots(figsize=(6.5,5))
    im=ax.imshow(pivot.values,origin="lower",aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{x:g}" for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{x:g}" for x in pivot.index])
    ax.set_xlabel(p2)
    ax.set_ylabel(p1)
    ax.set_title(f"{model}: {metric}")
    fig.colorbar(im,ax=ax)
    fig.tight_layout()
    fig.savefig(out/f"{model}_{metric}.png",dpi=180)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--path",default="results/map_curvature_stress/stress_grid_summary.csv")
    ap.add_argument("--output-dir",default="results/map_curvature_stress/figures")
    ap.add_argument("--metric",action="append")
    args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    s=pd.read_csv(args.path)
    metrics=args.metric or DEFAULT_METRICS
    for _,d in s.groupby("model"):
        for metric in metrics:
            plot_one(d,metric,out)
    print(f"Saved figures to {out}")

if __name__=="__main__":
    main()
