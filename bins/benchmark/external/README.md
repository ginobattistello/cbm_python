# External benchmark arms (not part of the cbm package)

`cbm_original/` — the pristine pre-fork CBM, extracted verbatim from commit
`e72193f` ("Initial commit"), which DEV.md §7 identifies as the only place the
unmodified upstream code exists. Do not edit: it is a comparison baseline.
Extracted 2026-08-12 with `git show e72193f:cbm/<file>`.
Verified pristine: 538-line `optimization.py`, zero `MODIFICATION` markers.

`VBA-toolbox/` — MBB-team/VBA-toolbox, shallow clone (2026-08-12), gitignored.
Re-create with:
    git clone --depth 1 https://github.com/MBB-team/VBA-toolbox.git
Requires MATLAB (tested: R2025b, headless via `matlab -batch`).
