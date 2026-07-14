#!/usr/bin/env bash
# Install hourly cron job for pre-epoch retraining.
set -euo pipefail

REPO_DIR="/root/workspaces/projects/poker44"
SCRIPT="${REPO_DIR}/scripts/retrain_before_epoch.sh"
CRON_TAG="# poker44-miner-epoch-retrain"
CRON_LINE="15 * * * * ${SCRIPT} >> ${REPO_DIR}/data/benchmark/epoch_retrain.log 2>&1 ${CRON_TAG}"

chmod +x "${SCRIPT}"
mkdir -p "${REPO_DIR}/data/benchmark"

tmp="$(mktemp)"
crontab -l 2>/dev/null | grep -v "${CRON_TAG}" > "${tmp}" || true
echo "${CRON_LINE}" >> "${tmp}"
crontab "${tmp}"
rm -f "${tmp}"

echo "Installed cron entry:"
echo "  ${CRON_LINE}"
