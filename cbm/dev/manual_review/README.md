# Manual review before upstream PR

This folder is intended to make the fork reviewable by a maintainer.

## Recommended setup

Keep two local clones next to each other:

```text
workspace/
├── cbm_python/             # your fork
└── cbm_python_original/    # payampiray/cbm_python
```

In the original clone:

```bash
git remote -v
git pull
```

In the fork clone:

```bash
git remote -v
git pull
```

Then, from the fork repository root:

```bash
python dev/manual_review/pre_pr_checks.py

bash dev/manual_review/review_core.sh ../cbm_python_original
```

Open:

```text
dev/manual_review/output/REPORT.md
```

then inspect the generated diffs in:

```text
dev/manual_review/output/diffs/
```

## Review order

Review the changes in this order:

1. `cbm/optimization.py`
   - multi-start L-BFGS-B;
   - optional GN polish;
   - central-FD / AD observed Hessian;
   - no evidence-Hessian clipping;
   - flags, warnings and diagnostics.

2. `cbm/map_estimation.py`
   - scalar vs trialwise model return;
   - free/fixed parameter mapping;
   - JAX model path.

3. `cbm/individual_fit.py`
   - public API;
   - zero-variance fixed parameters;
   - evidence validity;
   - verbose/display behavior;
   - observation/evolution outputs.

4. `cbm/display.py`
   - verify display is diagnostics only;
   - verify it cannot alter numerical outputs.

5. `cbm/model_selection.py` and `cbm/group_bms.py`
   - BMS numerical changes separately from presentation.

6. HBI files
   - confirm the updated MAP/evidence contract is propagated correctly.

7. `pyproject.toml`, README and examples
   - review only after numerical behavior is agreed.

## PR principle

The PR should make it easy to distinguish:

- numerical changes;
- API changes;
- diagnostics/plotting changes;
- examples/documentation;
- repository cleanup.

Avoid mixing legacy copies or generated files into the upstream diff.
