import sys
from datetime import datetime

def hbi_log(verbose: bool, fh, s: str) -> None:
    if verbose:
        sys.stdout.write(s)
        sys.stdout.flush()
    if fh is not None:
        fh.write(s)
        fh.flush()

def log_header(verbose, fid, K, N, fcbm_maps, isnull: bool):
    hbi_log(verbose, fid, f"{'=' * 70}\n")
    label = 'Hierarchical Bayesian Inference'
    hbi_log(verbose, fid, f"{label:<40s}{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):>30s}\n")
    hbi_log(verbose, fid, f"{'=' * 70}\n")
    if isnull:
        hbi_log(verbose, fid, "Running in null mode\n")
    hbi_log(verbose, fid, f"Number of samples: {N}\n")
    hbi_log(verbose, fid, f"Number of models: {K}\n")
    hbi_log(verbose, fid, "\n")
    hbi_log(verbose, fid, "Initialized from:\n")
    for k in range(K):
        hbi_log(verbose, fid, f"  {fcbm_maps[k]} [model {k + 1}]\n")
    hbi_log(verbose, fid, f"\n{'=' * 70}\n")

def log_iteration(verbose, fid, verbose_multiK, fid_multiK, it, Nbar, N, prog_change, terminate, K):
    # _hbi_log(verbose, fid, f"Iteration {it:02d}\n")
    if it > 1:
        hbi_log(verbose_multiK, fid_multiK, "\tmodel frequencies (percent)")
        ss = ""
        for k in range(K):
            ss += f"model {k + 1}: {Nbar[k] / N * 100:2.1f}| "
        hbi_log(verbose_multiK, fid_multiK, f"\n\t{ss}\n")
        dL_value = prog_change.change_bound
        hbi_log(verbose, fid, f"{' ':40s}{f'dL: {dL_value:7.2f}':>30s}\n")
        dm_value = prog_change.change_model_freq/N * 100
        hbi_log(verbose_multiK, fid_multiK, f"{' ':40s}{f'dm: {dm_value:7.2f}':>30s}\n")
        dx_value = prog_change.change_parameters
        hbi_log(verbose, fid, f"{' ':40s}{f'dx: {dx_value:7.2f}':>30s}\n")
        if terminate:
            hbi_log(verbose, fid, f"{' ':40s}{'Converged :]':>30s}\n")

def log_final(verbose, fid, output):
    hbi_log(verbose, fid, "\nFinal summary\n")
    mf = output.model_frequency
    hbi_log(verbose, fid, "Model frequencies (percent)\n")
    ss = "| ".join([f"model {i+1}: {mf[i]*100:4.1f}" for i in range(len(mf))])
    hbi_log(verbose, fid, f"\t{ss}| \n")
    if output.exceedance_prob.size > 0:
        xp = output.exceedance_prob
        hbi_log(verbose, fid, "Exceedance probabilities\n")
        ss = "| ".join([f"model {i+1}: {xp[i]:.3f}" for i in range(len(xp))])
        hbi_log(verbose, fid, f"\t{ss}| \n")
    if output.protected_exceedance_prob.size > 0:
        pxp = output.protected_exceedance_prob
        hbi_log(verbose, fid, "Protected exceedance probabilities\n")
        ss = "| ".join([f"model {i+1}: {pxp[i]:.3f}" for i in range(len(pxp))])
        hbi_log(verbose, fid, f"\t{ss}| \n")
