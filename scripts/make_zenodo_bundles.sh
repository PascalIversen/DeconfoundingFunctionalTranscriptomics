#!/usr/bin/env bash
# Build the two Zenodo tarballs from a fully-staged repo.
#   bash scripts/make_zenodo_bundles.sh raw     # -> ../zenodo_raw_data/raw_data.tar.gz
#   bash scripts/make_zenodo_bundles.sh preds   # -> ../zenodo_predictions/predictions.tar.gz
#   bash scripts/make_zenodo_bundles.sh all
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$(cd "$REPO/.." && pwd)"

build_raw() {
  echo ">> raw_data.tar.gz"
  tar -czf "$STAGE/zenodo_raw_data/raw_data.tar.gz" -C "$REPO" \
    data/DepMap/CRISPRGeneEffect.csv \
    data/DepMap/OmicsExpressionProteinCodingGenesTPMLogp1.csv \
    data/CCLE/gene_expression.csv \
    data/CTRPv2/CTRPv2.csv
}

build_preds() {
  echo ">> predictions.tar.gz"
  tar -czf "$STAGE/zenodo_predictions/predictions.tar.gz" -C "$REPO" \
    $(cd "$REPO" && ls -d results/*_preds) \
    $(cd "$REPO" && ls results/*.csv) \
    fig02_ci_distribution/results/ctrpv2_attr_eta2.csv \
    fig02_ci_distribution/results/depmap_attr_eta2.csv
}

case "${1:-all}" in
  raw)   build_raw ;;
  preds) build_preds ;;
  all)   build_raw; build_preds ;;
  *) echo "usage: $0 [raw|preds|all]"; exit 1 ;;
esac
echo ">> done. Upload the tarball(s) to Zenodo, then set the URLs in scripts/fetch_data.sh."
