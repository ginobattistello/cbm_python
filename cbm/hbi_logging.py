"""Small logging helpers for HBI.

Keeping formatting outside ``hbi.py`` makes the inference loop easier to read.
"""

from __future__ import annotations

from datetime import datetime
import sys


def hbi_log(verbose: bool, file_handle, text: str) -> None:
    if verbose:
        sys.stdout.write(text)
        sys.stdout.flush()

    if file_handle is not None:
        file_handle.write(text)
        file_handle.flush()


def log_header(
    verbose,
    file_handle,
    n_models,
    n_subjects,
    map_files,
    is_null: bool,
):
    hbi_log(verbose, file_handle, f"{'=' * 70}\n")
    label = "Hierarchical Bayesian Inference"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hbi_log(
        verbose,
        file_handle,
        f"{label:<40s}{now:>30s}\n",
    )
    hbi_log(verbose, file_handle, f"{'=' * 70}\n")

    if is_null:
        hbi_log(verbose, file_handle, "Running in null mode\n")

    hbi_log(
        verbose,
        file_handle,
        f"Number of subjects: {n_subjects}\n",
    )
    hbi_log(
        verbose,
        file_handle,
        f"Number of models: {n_models}\n\n",
    )

    hbi_log(verbose, file_handle, "Initialized from:\n")
    for k in range(n_models):
        hbi_log(
            verbose,
            file_handle,
            f"  {map_files[k]} [model {k + 1}]\n",
        )
    hbi_log(verbose, file_handle, f"\n{'=' * 70}\n")


def log_iteration(
    verbose,
    file_handle,
    verbose_multi_model,
    multi_model_file,
    iteration,
    Nbar,
    n_subjects,
    progress_change,
    terminate,
    n_models,
):
    if iteration <= 1:
        return

    hbi_log(
        verbose_multi_model,
        multi_model_file,
        "\tmodel frequencies (percent)\n\t",
    )
    text = " ".join(
        f"model {k + 1}: {Nbar[k] / n_subjects * 100:2.1f}% |"
        for k in range(n_models)
    )
    hbi_log(
        verbose_multi_model,
        multi_model_file,
        f"{text}\n",
    )

    hbi_log(
        verbose,
        file_handle,
        f"{' ':40s}{f'dL: {progress_change.change_bound:7.2f}':>30s}\n",
    )

    dm = progress_change.change_model_freq / n_subjects * 100
    hbi_log(
        verbose_multi_model,
        multi_model_file,
        f"{' ':40s}{f'dm: {dm:7.2f}':>30s}\n",
    )

    hbi_log(
        verbose,
        file_handle,
        f"{' ':40s}"
        f"{f'dx: {progress_change.change_parameters:7.2f}':>30s}\n",
    )

    if terminate:
        hbi_log(
            verbose,
            file_handle,
            f"{' ':40s}{'Converged :]':>30s}\n",
        )


def log_final(verbose, file_handle, output):
    hbi_log(verbose, file_handle, "\nFinal summary\n")
    hbi_log(verbose, file_handle, "Model frequencies (percent)\n")

    frequencies = output.model_frequency
    text = "| ".join(
        f"model {i + 1}: {frequencies[i] * 100:4.1f}"
        for i in range(len(frequencies))
    )
    hbi_log(verbose, file_handle, f"\t{text}| \n")

    if output.exceedance_prob.size:
        xp = output.exceedance_prob
        text = "| ".join(
            f"model {i + 1}: {xp[i]:.3f}"
            for i in range(len(xp))
        )
        hbi_log(verbose, file_handle, "Exceedance probabilities\n")
        hbi_log(verbose, file_handle, f"\t{text}| \n")

    if output.protected_exceedance_prob.size:
        pxp = output.protected_exceedance_prob
        text = "| ".join(
            f"model {i + 1}: {pxp[i]:.3f}"
            for i in range(len(pxp))
        )
        hbi_log(
            verbose,
            file_handle,
            "Protected exceedance probabilities\n",
        )
        hbi_log(verbose, file_handle, f"\t{text}| \n")
