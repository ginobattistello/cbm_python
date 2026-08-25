#!/usr/bin/env python3
"""Fast static pre-PR checks for the current CBM fork.

This does not replace numerical tests. It catches review/packaging issues that
make an upstream PR harder to assess.
"""

from pathlib import Path
import sys

CBM_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CBM_ROOT.parent

failures = []
warnings = []


def require(condition, message):
    if not condition:
        failures.append(message)


def warn(condition, message):
    if not condition:
        warnings.append(message)


# Repository hygiene.
require((REPO_ROOT / "README.md").exists(), "root README.md is missing")
require(not (CBM_ROOT / ".Rhistory").exists(), ".Rhistory should not be committed")
require(
    not any(CBM_ROOT.glob("*.egg-info")),
    "generated *.egg-info directory should not be committed",
)

legacy = list((CBM_ROOT / "cbm").glob("*_legacy.py"))
warn(not legacy, "legacy Python files remain in cbm/: " + ", ".join(p.name for p in legacy))

# Packaging metadata.
pyproject = REPO_ROOT / "pyproject.toml"
if pyproject.exists():
    text = pyproject.read_text(encoding="utf-8")
    require('name = "cbm-local"' not in text, 'pyproject package name is still "cbm-local"')
    require("Proprietary" not in text, "pyproject license metadata conflicts with MIT LICENSE")
    require("https://example.com" not in text, "pyproject contains placeholder project URL")
else:
    failures.append("pyproject.toml is missing")

# Core numerical invariants.
optimization = (CBM_ROOT / "optimization.py").read_text(encoding="utf-8")
require("_central_fd_hessian" in optimization, "central-FD observed Hessian is missing")
require("jax.hessian" in optimization, "optional AD Hessian backend is missing")
require("laplace_valid" in optimization, "Laplace validity flag is missing")
require("hess_condition_number" in optimization, "Hessian conditioning diagnostic is missing")
require("hess_raw_min_eig" in optimization, "raw minimum Hessian eigenvalue diagnostic is missing")
require("J.T @ J" in optimization, "GN optimization curvature is not documented")
require("hess_n_clipped=0" in optimization, "final-Hessian no-clipping invariant not explicit")

# Deprecated scipy output options should not be forwarded.
require('"disp"' not in optimization, "deprecated scipy L-BFGS-B disp option remains")
require('"iprint"' not in optimization, "deprecated scipy L-BFGS-B iprint option remains")

# Reproducibility is currently a warning until random_state is implemented.
warn(
    "random_state" in optimization or "rng" in optimization,
    "MAP random initializations have no explicit random_state/rng API",
)

model_selection = (CBM_ROOT / "model_selection.py").read_text(encoding="utf-8")
warn(
    "random_state" in model_selection or "rng" in model_selection,
    "BMS exceedance sampling has no explicit random_state/rng API",
)

# Tests.
warn((CBM_ROOT / "tests").exists(), "no automated tests/ directory")

print("CBM pre-PR static checks")
print("=" * 60)

if failures:
    print("\nFAIL")
    for item in failures:
        print(" -", item)

if warnings:
    print("\nWARN")
    for item in warnings:
        print(" -", item)

if not failures and not warnings:
    print("\nPASS")

sys.exit(1 if failures else 0)
