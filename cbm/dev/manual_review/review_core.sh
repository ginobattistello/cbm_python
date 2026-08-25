#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./dev/manual_review/review_core.sh ../cbm_python_original
#
# Run from the fork repository root.

ORIGINAL="${1:?Pass path to original cbm_python clone}"
FORK="$(pwd)"
OUT="dev/manual_review/output"

mkdir -p "$OUT/raw_diffs"

FILES=(
  cbm/__init__.py
  cbm/optimization.py
  cbm/map_estimation.py
  cbm/individual_fit.py
  cbm/model_selection.py
  cbm/group_bms.py
  cbm/hbi.py
  cbm/hbi_bound.py
  cbm/hbi_config.py
  cbm/hbi_exceedance.py
  cbm/hbi_logging.py
  cbm/hbi_types.py
  cbm/hbi_updates.py
  pyproject.toml
  README.md
)

for rel in "${FILES[@]}"; do
  safe="${rel//\//__}"
  git diff --no-index -- \
    "$ORIGINAL/$rel" \
    "$FORK/$rel" \
    > "$OUT/raw_diffs/${safe}.diff" || true
done

python dev/manual_review/compare_cbm.py \
  --fork "$FORK" \
  --original "$ORIGINAL" \
  --out "$OUT"

echo
echo "Manual review generated."
echo "Start with: $OUT/REPORT.md"
