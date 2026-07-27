#!/usr/bin/env bash
# Retrain coherent-rank ensemble for >=0.60 round scores (R2 onward).
set -euo pipefail
cd "$(dirname "$0")/.."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy || true
source miner_env/bin/activate
export PYTHONPATH=.

echo "==> Download latest benchmark"
python deploy/download_benchmark.py --dates 30 --refresh

echo "==> Train stacked + hybrid v22"
python deploy/train_stacked.py --dates 30 --holdout-dates 8 --output models/stacked.joblib --refresh-cache
python deploy/train_hybrid.py --dates 30 --holdout-dates 8 --output models/hybrid.joblib --refresh-cache

echo "==> Tune coherent rank ensemble (floor 0.60)"
python deploy/tune_ensemble.py --dates 30 --holdout-dates 8 --refresh-cache

echo "Done. Restart: pm2 restart poker44_miner --update-env"
