# Exhaustive manual review

This comparison uses the **original CBM repository as the baseline**.

It reports:

1. every tracked file that is **new in the fork**;
2. every tracked file from original CBM that is **deleted in the fork**;
3. **all shared files**, including unchanged files;
4. for every modified shared file:
   - original line range;
   - proposed fork line range;
   - diff hunk context;
   - affected top-level Python functions/classes/methods when detectable;
   - the complete unified diff.

## Local layout

Recommended:

```text
Documents/
├── cbm_python/              # your fork
└── cbm_python_original/     # payampiray/cbm_python
```

Make sure both are up to date:

```bash
cd ../cbm_python_original
git pull

cd ../cbm_python
git pull
```

## Run

From the **fork repository root**:

```bash
bash cbm/dev/manual_review/review_all.sh ../cbm_python_original
```

Then open:

```text
cbm/dev/manual_review/output/REPORT.md
```

## Output

```text
output/
├── REPORT.md
├── inventory.json
├── shared_file_edits.json
└── diffs/
    ├── cbm__optimization.py.diff
    ├── cbm__individual_fit.py.diff
    └── ...
```

`git ls-files` is used for both clones whenever possible. This is important:
the report describes source-controlled files rather than local caches,
`egg-info`, test outputs, or other untracked artifacts.

## Interpretation

The categories are directional:

```text
ORIGINAL -> FORK
```

Therefore:

- **new** = exists in fork but not original;
- **deleted** = exists in original but not fork;
- **shared** = exists at the same path in both;
- **modified shared** = shared path, different contents.

The script also detects exact-content deleted/new pairs as possible file
renames or moves.
