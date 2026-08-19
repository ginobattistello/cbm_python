"""Command-line entry point.

Run from the repository root:

    python -m cbm.dev.map_curvature.run_experiment

or:

    python -m cbm.dev.map_curvature.run_experiment --model binary --n-datasets 20
"""

from __future__ import annotations

import argparse

from .experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="MAP curvature experiment: GN vs FD vs autodiff."
    )

    parser.add_argument(
        "--model",
        choices=["binary", "categorical", "ces", "all"],
        default="all",
    )
    parser.add_argument("--n-datasets", type=int, default=20)
    parser.add_argument("--n-trials", type=int, default=250)
    parser.add_argument("--n-starts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="results/map_curvature")

    args = parser.parse_args()

    models = (
        ["binary", "categorical", "ces"]
        if args.model == "all"
        else [args.model]
    )

    run_experiment(
        models=models,
        n_datasets=args.n_datasets,
        n_trials=args.n_trials,
        n_starts=args.n_starts,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
