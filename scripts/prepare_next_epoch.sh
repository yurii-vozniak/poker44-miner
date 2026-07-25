#!/usr/bin/env bash
# Retrain and tune for next competition epoch (target: >=0.55 every round).
set -euo pipefail
cd "$(dirname "$0")/.."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy || true
source miner_env/bin/activate
export PYTHONPATH=.

echo "==> Download latest benchmark releases"
python deploy/download_benchmark.py --dates 41 --refresh

echo "==> Train hybrid v21 (batched-reward selection)"
python deploy/train_hybrid.py --dates 41 --holdout-dates 10 --output models/hybrid.joblib --refresh-cache

echo "==> Tune live batch postprocess"
python deploy/tune_hybrid_live.py --dates 41 --holdout-dates 10

echo "==> Copy hybrid to production model path"
cp -f models/hybrid.joblib models/production.joblib

echo "Done. Restart miner: pm2 restart poker44_miner --update-env"
