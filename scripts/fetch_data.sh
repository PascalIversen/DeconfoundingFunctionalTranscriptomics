#!/usr/bin/env bash
# Fetch the raw-data and prediction bundles from Zenodo into this repo.
#
#   Deposit 1 (raw data, ~1.3 GB) -> data/{DepMap,CCLE,CTRPv2}/
#   Deposit 2 (predictions + precomputed result CSVs, ~2.5-3.5 GB) -> results/ and figXX/results/
#
# Set the two record URLs below once the Zenodo deposits are published, then run:
#   bash scripts/fetch_data.sh            # both bundles
#   bash scripts/fetch_data.sh raw        # raw data only
#   bash scripts/fetch_data.sh preds      # predictions only
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RAW_URL="${ZENODO_RAW_URL:-https://zenodo.org/records/21171970/files/raw_data.tar.gz?download=1}"
PREDS_URL="${ZENODO_PREDS_URL:-https://zenodo.org/records/21171970/files/predictions.tar.gz?download=1}"

fetch() {  # url dest.tar.gz extract_dir
  local url="$1" tgz="$2" dest="$3"
  [ "$url" = TODO* ] && { echo "!! $url not set: edit scripts/fetch_data.sh or export ZENODO_*_URL"; return 1; }
  echo ">> downloading $url"
  curl -L --fail -o "$tgz" "$url"
  echo ">> extracting into $dest"
  mkdir -p "$dest"; tar -xzf "$tgz" -C "$dest"; rm -f "$tgz"
}

what="${1:-all}"
case "$what" in
  raw)   fetch "$RAW_URL"   "$HERE/_raw.tar.gz"   "$HERE/data" ;;
  preds) fetch "$PREDS_URL" "$HERE/_preds.tar.gz" "$HERE" ;;
  all)   fetch "$RAW_URL"   "$HERE/_raw.tar.gz"   "$HERE/data"
         fetch "$PREDS_URL" "$HERE/_preds.tar.gz" "$HERE" ;;
  *) echo "usage: $0 [raw|preds|all]"; exit 1 ;;
esac
echo ">> done."
