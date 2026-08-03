#!/usr/bin/env bash
# Install v3 monitor/retrain cron (every 30 minutes).
set -euo pipefail

REPO_DIR="/root/workspaces/projects/poker44"
SCRIPT="${REPO_DIR}/scripts/v3_monitor_and_retrain.sh"
CRON_TAG="# poker44-v3-monitor"
CRON_LINE="*/30 * * * * ${SCRIPT} >> ${REPO_DIR}/data/benchmark/v3_monitor.log 2>&1 ${CRON_TAG}"

chmod +x "${SCRIPT}" "${REPO_DIR}/scripts/v3_health_check.sh"
mkdir -p "${REPO_DIR}/data/benchmark"

tmp="$(mktemp)"
crontab -l 2>/dev/null | grep -v "${CRON_TAG}" > "${tmp}" || true
echo "${CRON_LINE}" >> "${tmp}"
crontab "${tmp}"
rm -f "${tmp}"

echo "Installed cron entry:"
echo "  ${CRON_LINE}"
