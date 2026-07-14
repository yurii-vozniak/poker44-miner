#!/usr/bin/env bash
# Retrain when a new benchmark sourceDate is available, then restart the miner.
set -euo pipefail

REPO_DIR="/root/workspaces/projects/poker44"
STATE_FILE="${REPO_DIR}/data/benchmark/last_trained_source_date.txt"
LOG_FILE="${REPO_DIR}/data/benchmark/auto_retrain.log"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true

mkdir -p "${REPO_DIR}/data/benchmark"
# shellcheck disable=SC1091
source "${REPO_DIR}/miner_env/bin/activate"
export PYTHONPATH="${REPO_DIR}"
cd "${REPO_DIR}"

latest="$(
python - <<'PY'
from deploy.benchmark_client import BenchmarkClient
print(BenchmarkClient().latest_source_date())
PY
)"

last=""
if [ -f "${STATE_FILE}" ]; then
  last="$(<"${STATE_FILE}")"
fi

{
  echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] latest=${latest} last=${last:-none}"
} >> "${LOG_FILE}"

if [ "${latest}" = "${last}" ]; then
  echo "No new benchmark release (${latest})."
  exit 0
fi

echo "New benchmark release detected: ${latest}"
python deploy/train_hybrid.py --dates 90 --holdout-dates 5 --refresh-cache
bash scripts/poker44-miner validate
bash scripts/poker44-miner restart
printf '%s' "${latest}" > "${STATE_FILE}"
echo "Retrain complete for ${latest}" | tee -a "${LOG_FILE}"
