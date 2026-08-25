#!/usr/bin/env bash
set -euo pipefail

# Exhaustive original-CBM -> fork review.
#
# Usage, from the FORK repository root:
#
#   bash cbm/dev/manual_review/review_all.sh ../cbm_python_original
#
# The original repository is the baseline.

ORIGINAL="${1:?Pass the path to the original cbm_python clone}"
FORK="$(pwd)"
OUT="cbm/dev/manual_review/output"

rm -rf "$OUT"

python cbm/dev/manual_review/compare_cbm.py \
    --fork "$FORK" \
    --original "$ORIGINAL" \
    --out "$OUT"

echo
echo "Generated:"
echo "  $OUT/REPORT.md"
echo "  $OUT/inventory.json"
echo "  $OUT/shared_file_edits.json"
echo "  $OUT/diffs/"
