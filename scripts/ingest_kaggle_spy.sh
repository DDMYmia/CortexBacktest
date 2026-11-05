#!/bin/sh
# Ingest the Kaggle SPY EOD dataset (2010-2023) into the option-chain store.
#
#   kagglehub downloads to ~/.cache/kagglehub/datasets/dudesurfin/
#   spy-options-eod-volatility-surface-2010-2023/versions/2
#
# 16:00 America/New_York, not a fixed UTC hour: the sample spans fourteen years
# of daylight-saving changes. Structural cleaning only — the store keeps zero
# bids, wide spreads and IV outliers so each study applies its own filter; a
# store built at one filter strength cannot be widened later.
set -eu
SOURCE=${1:?usage: ingest_kaggle_spy.sh <dataset dir> [store root]}
ROOT=${2:-data/options}
for file in "$SOURCE"/spy_eod_*.parquet; do
    echo "=== $file"
    .venv/bin/python -u scripts/fetch_option_chains.py file "$file" \
        --source kaggle_spy_eod --root "$ROOT" \
        --zone America/New_York --quote-time 16:00 \
        --structural-only --dates-per-chunk 20
done
