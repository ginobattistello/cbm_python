# Original CBM → proposed fork: exhaustive file review

The **original CBM repository is the baseline**. All categories below are computed over tracked files.

## Summary

| Category | Count |
| --- | --- |
| Original tracked files | 20 |
| Fork tracked files | 51 |
| 1. New files | 36 |
| 2. Deleted original files | 5 |
| 3. Shared files | 15 |
| 4a. Shared + modified | 13 |
| 4b. Shared + unchanged | 2 |

Inventory source: original = `filesystem fallback`; fork = `git ls-files`.

## 1. New files added in the fork

- `.Rhistory`
- `.github/workflows/tests.yml`
- `.gitignore`
- `cbm/bms_group.py`
- `cbm/dev/manual_review/README.md`
- `cbm/dev/manual_review/compare_cbm.py`
- `cbm/dev/manual_review/pre_pr_checks.py`
- `cbm/dev/manual_review/review_core.sh`
- `cbm/display.py`
- `cbm/examples/01_individual_fit.py`
- `cbm/examples/02_parameter_recovery.py`
- `cbm/examples/03_bms.py`
- `cbm/examples/04_hbi.py`
- `cbm/examples/05_fixed_parameters.py`
- `cbm/examples/README.md`
- `cbm/examples/models.py`
- `cbm/examples/output/hbi_map_fixed_alpha.pkl`
- `cbm/examples/output/hbi_map_free.pkl`
- `cbm/examples/simulate.py`
- `cbm/hessian.py`
- `cbm/parameter_space.py`
- `cbm/reporting.py`
- `cbm_local.egg-info/PKG-INFO`
- `cbm_local.egg-info/SOURCES.txt`
- `cbm_local.egg-info/dependency_links.txt`
- `cbm_local.egg-info/top_level.txt`
- `tests/conftest.py`
- `tests/test_bms.py`
- `tests/test_display_reporting.py`
- `tests/test_hbi.py`
- `tests/test_hbi_updates.py`
- `tests/test_hessian.py`
- `tests/test_individual_fit.py`
- `tests/test_optimization.py`
- `tests/test_parameter_space.py`
- `tests/test_scientific_regression.py`

## 2. Original CBM files deleted in the fork

- `examples/README.md`
- `examples/exampla_individual_fit.py`
- `examples/example.py`
- `examples/example_RL.py`
- `examples/example_model_selection.py`

## 3. Shared files

| Shared file | Status | Where changed |
| --- | --- | --- |
| `LICENSE` | unchanged | — |
| `README.md` | modified | 1-27 → 1-50, 31-58 → 54-80 |
| `cbm/__init__.py` | modified | 1-3 → — |
| `cbm/hbi.py` | modified | 1-37 → 1-96, 39-343 → 98-978, 350-475 → 985-997, 490-505 → 1012-1031, 507-656 → 1033-1182; function `_hbi_prog`, function `_load_map_result`, function `_validate_hbi_map_files`, function `_validate_optional_model_list`, function `hbi_init`, function `hbi_main`, function `hbi_null`, function `hbi_run` |
| `cbm/hbi_bound.py` | unchanged | — |
| `cbm/hbi_config.py` | modified | 1-70 → 1-72, 76-126 → 78-109; class `HBIConfig`, function `_default_fname`, function `_valid_flog`, function `_valid_fname`, method `HBIConfig.__post_init__` |
| `cbm/hbi_exceedance.py` | modified | 1-8 → 1-14, 10-44 → 16-42, 46-140 → 44-138; function `_compute_exceedance`, function `_dirichlet_exceedance`, function `cbm_hbi_exceedance` |
| `cbm/hbi_logging.py` | modified | 1-59 → 1-153; function `hbi_log`, function `log_final`, function `log_header`, function `log_iteration` |
| `cbm/hbi_types.py` | modified | 1-13 → 1-35, 15-33 → 37-60, 36-47 → 63-76, 61-66 → 90-96, 69-74 → 99-105, 77-87 → 108-120, 90-102 → 123-146, 105-110 → 149-155, 119-124 → 164-170, 129-141 → 175-212; class `BoundQHZ`, class `BoundQM`, class `BoundQMutau`, class `BoundState`, class `BoundTerms`, class `DirichletDistribution`, class `ExceedanceResult`, class `GaussianDistribution`, class `GaussianGammaDistribution`, class `HBIInput`, class `HBIMath`, class `HBIOutput`, class `HBIProfile`, class `HBIResult`, class `IndividualPosterior`, class `ProgressChange`, class `ProgressState`, method `HBIResult.__repr__`, method `HBIResult.subject_table`, method `HBIResult.summary`, method `HBIResult.table` |
| `cbm/hbi_updates.py` | modified | 1-47 → 1-94, 49-83 → 96-154, 85-104 → 156-195, 110-115 → 201-207, 117-142 → 209-248, 145-164 → 251-272, 167-197 → 275-368, 199-222 → 370-421, 224-297 → 423-658, 302-331 → 663-702, 336-345 → 707-719; function `_validate_optional_model_list`, function `hbi_bound`, function `hbi_qHZ`, function `hbi_qhquad`, function `hbi_qm`, function `hbi_qmutau`, function `hbi_sumstats` |
| `cbm/individual_fit.py` | modified | 1-77 → 1-269, 79-369 → 271-1012; class `FitInput`, class `FitMath`, class `FitOutput`, class `FitProfile`, class `FitResult`, class `Prior`, function `_preflight_checks`, function `_probe_evolution`, function `_resolve_config`, function `_resolve_prior`, function `_validate_evolution_output`, function `individual_fit`, method `FitResult.__repr__`, method `FitResult.plot`, method `FitResult.se`, method `FitResult.summary`, method `FitResult.table`, method `Prior.__post_init__`, method `Prior.d_free`, method `Prior.fixed_mask`, method `Prior.free_mask`, method `Prior.free_mean`, method `Prior.free_precision` |
| `cbm/map_estimation.py` | modified | 1-99 → 1-426; function `_detect_trialwise_model`, function `_fixed_only_result`, function `_make_jax_neg_log_posterior`, function `_model_output`, function `_validate_prior`, function `log_posterior`, function `optimize_map` |
| `cbm/model_selection.py` | modified | 1-34 → 1-34, 37-123 → 37-108, 125-286 → 110-244; class `BMSResult`, function `_resolve_rng`, function `bms`, function `compute_bor`, function `compute_fe`, function `dirichlet_exceedance`, function `fe_null` |
| `cbm/optimization.py` | modified | 1-46 → 1-90, 54-108 → 98-209, 115-437 → 216-1078, 444-539 → 1085-1130; class `BFGSOptimizer`, class `Config`, class `ConvergenceStatus`, class `OptimizationResult`, class `PostFitDiagnostics`, function `_expand_bounds`, method `BFGSOptimizer.__init__`, method `BFGSOptimizer._autodiff_hessian`, method `BFGSOptimizer._central_fd_hessian`, method `BFGSOptimizer._central_gradient`, method `BFGSOptimizer._gauss_newton_curvature`, method `BFGSOptimizer._newton_polish`, method `BFGSOptimizer._single_optimization`, method `BFGSOptimizer.compute_hessian`, method `BFGSOptimizer.get_all_results`, method `BFGSOptimizer.get_history`, method `BFGSOptimizer.optimize`, method `Config.__post_init__`, method `OptimizationResult.F`, method `OptimizationResult.diagnostics`, method `OptimizationResult.neg_log_post` |
| `pyproject.toml` | modified | 3-19 → 3-42 |

## 4. Proposed edits within shared files

### `README.md`

| Original line(s) | Fork line(s) | Diff context |
| --- | --- | --- |
| 1-27 | 1-50 | — |
| 31-58 | 54-80 | — |

Full unified diff: `diffs/README.md.diff`

### `cbm/__init__.py`

| Original line(s) | Fork line(s) | Diff context |
| --- | --- | --- |
| 1-3 | — | — |

Full unified diff: `diffs/cbm____init__.py.diff`

### `cbm/hbi.py`

**Affected Python scopes:** function `_hbi_prog`, function `_load_map_result`, function `_validate_hbi_map_files`, function `_validate_optional_model_list`, function `hbi_init`, function `hbi_main`, function `hbi_null`, function `hbi_run`

| Original line(s) | Fork line(s) | Diff context |
| --- | --- | --- |
| 1-37 | 1-96 | — |
| 39-343 | 98-978 | — |
| 350-475 | 985-997 | — |
| 490-505 | 1012-1031 | — |
| 507-656 | 1033-1182 | — |

Full unified diff: `diffs/cbm__hbi.py.diff`

### `cbm/hbi_config.py`

**Affected Python scopes:** class `HBIConfig`, function `_default_fname`, function `_valid_flog`, function `_valid_fname`, method `HBIConfig.__post_init__`

| Original line(s) | Fork line(s) | Diff context |
| --- | --- | --- |
| 1-70 | 1-72 | — |
| 76-126 | 78-109 | — |

Full unified diff: `diffs/cbm__hbi_config.py.diff`

### `cbm/hbi_exceedance.py`

**Affected Python scopes:** function `_compute_exceedance`, function `_dirichlet_exceedance`, function `cbm_hbi_exceedance`

| Original line(s) | Fork line(s) | Diff context |
| --- | --- | --- |
| 1-8 | 1-14 | — |
| 10-44 | 16-42 | — |
| 46-140 | 44-138 | — |

Full unified diff: `diffs/cbm__hbi_exceedance.py.diff`

### `cbm/hbi_logging.py`

**Affected Python scopes:** function `hbi_log`, function `log_final`, function `log_header`, function `log_iteration`

| Original line(s) | Fork line(s) | Diff context |
| --- | --- | --- |
| 1-59 | 1-153 | — |

Full unified diff: `diffs/cbm__hbi_logging.py.diff`

### `cbm/hbi_types.py`

**Affected Python scopes:** class `BoundQHZ`, class `BoundQM`, class `BoundQMutau`, class `BoundState`, class `BoundTerms`, class `DirichletDistribution`, class `ExceedanceResult`, class `GaussianDistribution`, class `GaussianGammaDistribution`, class `HBIInput`, class `HBIMath`, class `HBIOutput`, class `HBIProfile`, class `HBIResult`, class `IndividualPosterior`, class `ProgressChange`, class `ProgressState`, method `HBIResult.__repr__`, method `HBIResult.subject_table`, method `HBIResult.summary`, method `HBIResult.table`

| Original line(s) | Fork line(s) | Diff context |
| --- | --- | --- |
| 1-13 | 1-35 | — |
| 15-33 | 37-60 | — |
| 36-47 | 63-76 | — |
| 61-66 | 90-96 | — |
| 69-74 | 99-105 | — |
| 77-87 | 108-120 | — |
| 90-102 | 123-146 | — |
| 105-110 | 149-155 | — |
| 119-124 | 164-170 | — |
| 129-141 | 175-212 | — |

Full unified diff: `diffs/cbm__hbi_types.py.diff`

### `cbm/hbi_updates.py`

**Affected Python scopes:** function `_validate_optional_model_list`, function `hbi_bound`, function `hbi_qHZ`, function `hbi_qhquad`, function `hbi_qm`, function `hbi_qmutau`, function `hbi_sumstats`

| Original line(s) | Fork line(s) | Diff context |
| --- | --- | --- |
| 1-47 | 1-94 | — |
| 49-83 | 96-154 | — |
| 85-104 | 156-195 | — |
| 110-115 | 201-207 | — |
| 117-142 | 209-248 | — |
| 145-164 | 251-272 | — |
| 167-197 | 275-368 | — |
| 199-222 | 370-421 | — |
| 224-297 | 423-658 | — |
| 302-331 | 663-702 | — |
| 336-345 | 707-719 | — |

Full unified diff: `diffs/cbm__hbi_updates.py.diff`

### `cbm/individual_fit.py`

**Affected Python scopes:** class `FitInput`, class `FitMath`, class `FitOutput`, class `FitProfile`, class `FitResult`, class `Prior`, function `_preflight_checks`, function `_probe_evolution`, function `_resolve_config`, function `_resolve_prior`, function `_validate_evolution_output`, function `individual_fit`, method `FitResult.__repr__`, method `FitResult.plot`, method `FitResult.se`, method `FitResult.summary`, method `FitResult.table`, method `Prior.__post_init__`, method `Prior.d_free`, method `Prior.fixed_mask`, method `Prior.free_mask`, method `Prior.free_mean`, method `Prior.free_precision`

| Original line(s) | Fork line(s) | Diff context |
| --- | --- | --- |
| 1-77 | 1-269 | — |
| 79-369 | 271-1012 | — |

Full unified diff: `diffs/cbm__individual_fit.py.diff`

### `cbm/map_estimation.py`

**Affected Python scopes:** function `_detect_trialwise_model`, function `_fixed_only_result`, function `_make_jax_neg_log_posterior`, function `_model_output`, function `_validate_prior`, function `log_posterior`, function `optimize_map`

| Original line(s) | Fork line(s) | Diff context |
| --- | --- | --- |
| 1-99 | 1-426 | — |

Full unified diff: `diffs/cbm__map_estimation.py.diff`

### `cbm/model_selection.py`

**Affected Python scopes:** class `BMSResult`, function `_resolve_rng`, function `bms`, function `compute_bor`, function `compute_fe`, function `dirichlet_exceedance`, function `fe_null`

| Original line(s) | Fork line(s) | Diff context |
| --- | --- | --- |
| 1-34 | 1-34 | — |
| 37-123 | 37-108 | — |
| 125-286 | 110-244 | — |

Full unified diff: `diffs/cbm__model_selection.py.diff`

### `cbm/optimization.py`

**Affected Python scopes:** class `BFGSOptimizer`, class `Config`, class `ConvergenceStatus`, class `OptimizationResult`, class `PostFitDiagnostics`, function `_expand_bounds`, method `BFGSOptimizer.__init__`, method `BFGSOptimizer._autodiff_hessian`, method `BFGSOptimizer._central_fd_hessian`, method `BFGSOptimizer._central_gradient`, method `BFGSOptimizer._gauss_newton_curvature`, method `BFGSOptimizer._newton_polish`, method `BFGSOptimizer._single_optimization`, method `BFGSOptimizer.compute_hessian`, method `BFGSOptimizer.get_all_results`, method `BFGSOptimizer.get_history`, method `BFGSOptimizer.optimize`, method `Config.__post_init__`, method `OptimizationResult.F`, method `OptimizationResult.diagnostics`, method `OptimizationResult.neg_log_post`

| Original line(s) | Fork line(s) | Diff context |
| --- | --- | --- |
| 1-46 | 1-90 | — |
| 54-108 | 98-209 | — |
| 115-437 | 216-1078 | — |
| 444-539 | 1085-1130 | — |

Full unified diff: `diffs/cbm__optimization.py.diff`

### `pyproject.toml`

| Original line(s) | Fork line(s) | Diff context |
| --- | --- | --- |
| 3-19 | 3-42 | — |

Full unified diff: `diffs/pyproject.toml.diff`


## Review order

For a PR review, inspect:

1. **Deleted files** first — confirm every deletion is intentional.
2. **New production files** — confirm each belongs upstream.
3. **Modified shared files** — follow the line ranges above and open the corresponding unified diff.
4. **Unchanged shared files** require no code review.
