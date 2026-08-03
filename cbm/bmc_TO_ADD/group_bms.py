from itertools import product as iterproduct
from scipy.stats import dirichlet
import warnings


class GroupBMS:
    """
    Group-level Bayesian Model Selection.
    Reproduces VBA_groupBMC / VBA_groupBMC_btwConds.
    """

    def __init__(self, L, options=None, n_samples=1_000_000):
        if options is None:
            options = {}
        self._ns = n_samples
        self._opts = options

        if isinstance(L, (list, tuple)) and not isinstance(L, np.ndarray):
            self.result = self._btw_groups(list(L), options)
        elif L.ndim == 2:
            self.result = self._standard(L, options)
        elif L.ndim == 3:
            self.result = self._btw_conds(L, options)
        else:
            raise ValueError("L must be 2D, 3D, or a list of 2D arrays")

    def __getitem__(self, key):
        return self.result[key]

    def keys(self):
        return self.result.keys()

    # =========================================================
    # Standard BMS
    # =========================================================
    def _standard(self, L, options):
        n_sub, n_mod = L.shape

        if options.get("families") is None:
            a0 = np.ones(n_mod) / n_mod
            res = bms(L, alpha0=a0)
            return {
                "models": {
                    "a": res.posterior_parameters,
                    "ef": res.model_frequency,
                    "xp": res.exceedance_prob,
                    "bor": res.bor,
                    "pxp": res.protected_exceedance_prob,
                    "alpha0": a0,
                }
            }

        fams = options["families"]
        nf = len(fams)
        names = options.get("family_names", [f"f_{i}" for i in range(nf)])
        assert len(names) == nf

        a0 = np.zeros(n_mod)
        for f, idx in enumerate(fams):
            a0[np.array(idx)] = 1.0 / (nf * len(idx))

        res = bms(L, alpha0=a0)
        a = res.posterior_parameters
        a_fam = np.array([a[np.array(idx)].sum() for idx in fams])

        samp = dirichlet.rvs(a, size=self._ns)
        ff = np.zeros((self._ns, nf))
        for f, idx in enumerate(fams):
            ff[:, f] = samp[:, np.array(idx)].sum(axis=1)

        fam_ef = ff.mean(axis=0)
        w = ff.argmax(axis=1)
        fam_xp = np.array([(w == f).mean() for f in range(nf)])

        bor_c = np.clip(res.bor, 1e-16, 1 - 1e-16)
        lbf = np.log((1 - bor_c) / bor_c) - n_sub * np.log(n_mod / nf)
        fam_bor = 1 / (1 + np.exp(lbf))
        fam_pxp = fam_bor / nf + (1 - fam_bor) * fam_xp

        within = []
        for f, idx in enumerate(fams):
            idx = np.asarray(idx)
            a0w = np.ones(len(idx)) / len(idx)
            rw = bms(L[:, idx], alpha0=a0w)
            within.append(
                {
                    "name": names[f],
                    "models": idx,
                    "a": rw.posterior_parameters,
                    "ef": rw.model_frequency,
                    "xp": rw.exceedance_prob,
                    "bor": rw.bor,
                    "pxp": rw.protected_exceedance_prob,
                }
            )

        return {
            "models": {
                "a": a,
                "ef": res.model_frequency,
                "xp": res.exceedance_prob,
                "bor": res.bor,
                "pxp": res.protected_exceedance_prob,
                "alpha0": a0,
            },
            "families": {
                "names": names,
                "a": a_fam,
                "ef": fam_ef,
                "xp": fam_xp,
                "bor": fam_bor,
                "pxp": fam_pxp,
                "within": within,
            },
        }

    # =========================================================
    # Between-conditions BMS
    # =========================================================
    def _btw_conds(self, L, options):
        n_sub, n_mod, n_cond = L.shape
        fams = options.get("families")
        fam_names = options.get("family_names")

        if n_cond == 1:
            warnings.warn("Only 1 condition — falling back to standard BMS")
            return self._standard(L[:, :, 0], options)

        # Family assignment per model
        if fams is not None:
            cfam = np.zeros(n_mod, dtype=int)
            for f, idx in enumerate(fams):
                cfam[np.array(idx)] = f
        else:
            cfam = np.arange(n_mod)

        # All K^C tuples
        tuples = np.array(list(iterproduct(range(n_mod), repeat=n_cond)))
        nt = len(tuples)
        assert nt == n_mod**n_cond, (
            f"Expected {n_mod}^{n_cond}={n_mod**n_cond}, got {nt}"
        )

        if nt > 100_000:
            warnings.warn(f"{n_mod}^{n_cond} = {nt} tuples — may be slow")

        # Tuple log-evidence: sum across conditions
        Lt = sum(L[:, tuples[:, c], c] for c in range(n_cond))

        # Classify: equal vs not-equal
        is_eq = np.array([len(set(cfam[tuples[t]])) == 1 for t in range(nt)])
        eq_idx = np.where(is_eq)[0].tolist()
        neq_idx = np.where(~is_eq)[0].tolist()

        # Between-conditions BMS
        btw_opts = {
            "families": [eq_idx, neq_idx],
            "family_names": ["equal", "not_equal"],
        }
        btw = self._standard(Lt, btw_opts)

        # Per-condition BMS
        per_cond = [self._standard(L[:, :, c], options) for c in range(n_cond)]

        # Best tuple
        best_t = np.argmax(btw["models"]["ef"])
        best_models = tuples[best_t]
        best_families = cfam[best_models] if fams is not None else best_models
        best_info = {
            "tuple_idx": best_t,
            "models": best_models,
            "families": best_families,
            "family_names": [fam_names[f] for f in best_families]
            if fam_names
            else None,
            "is_equal": is_eq[best_t],
        }

        return {
            "xp": btw["families"]["xp"][0],
            "pxp": btw["families"]["pxp"][0],
            "bor": btw["models"]["bor"],
            "n_tuples": nt,
            "n_equal": len(eq_idx),
            "n_not_equal": len(neq_idx),
            "tuples": tuples,
            "best": best_info,
            "btw": btw,
            "per_cond": per_cond,
        }

    # =========================================================
    # Between-groups BMS
    # =========================================================
    def _btw_groups(self, Ls, options=None):
        """
        Between-groups BMS. Ls: list of G arrays, each (n_sub_g, n_mod).
        Reproduces VBA_groupBMC_btwGroups.
        """
        if options is None:
            options = self._opts or {}
        n_grp = len(Ls)
        if n_grp < 2:
            raise ValueError("btwGroups requires >= 2 groups")
        n_mod = Ls[0].shape[1]
        if any(L.shape[1] != n_mod for L in Ls):
            raise ValueError("All groups must have the same number of models")

        fams = options.get("families")
        fam_names = options.get("family_names")
        if fams is not None:
            cfam = np.zeros(n_mod, dtype=int)
            for f, idx in enumerate(fams):
                cfam[np.array(idx)] = f
        else:
            cfam = np.arange(n_mod)

        # H1: each group has its own model -> tuples over groups
        tuples = np.array(list(iterproduct(range(n_mod), repeat=n_grp)))
        nt = len(tuples)
        if nt > 100_000:
            warnings.warn(f"{n_mod}^{n_grp} = {nt} tuples — may be slow")

        # Subjects x tuples. A subject in group g only informs the g-th slot,
        # so its log-evidence for tuple t is L_g[:, tuples[t, g]].
        Lt = np.vstack([Ls[g][:, tuples[:, g]] for g in range(n_grp)])

        is_eq = np.array([len(set(cfam[tuples[t]])) == 1 for t in range(nt)])
        eq_idx = np.where(is_eq)[0].tolist()
        neq_idx = np.where(~is_eq)[0].tolist()

        btw = self._standard(
            Lt,
            {"families": [eq_idx, neq_idx], "family_names": ["equal", "not_equal"]},
        )

        per_group = [self._standard(L, options) for L in Ls]

        best_t = int(np.argmax(btw["models"]["ef"]))
        best_models = tuples[best_t]
        best_families = cfam[best_models]
        best_info = {
            "tuple_idx": best_t,
            "models": best_models,
            "families": best_families,
            "family_names": [fam_names[f] for f in best_families]
            if fam_names
            else None,
            "is_equal": bool(is_eq[best_t]),
        }

        return {
            "xp": btw["families"]["xp"][0],  # P(same model/family across groups)
            "pxp": btw["families"]["pxp"][0],
            "bor": btw["models"]["bor"],
            "n_groups": n_grp,
            "n_tuples": nt,
            "n_equal": len(eq_idx),
            "n_not_equal": len(neq_idx),
            "tuples": tuples,
            "group_sizes": [L.shape[0] for L in Ls],
            "best": best_info,
            "btw": btw,
            "per_group": per_group,
        }