#!/usr/bin/env bash
# Health check for Poker44 v3 micro-session miner (UID 79).
set -euo pipefail

REPO_DIR="/root/workspaces/projects/poker44"
cd "${REPO_DIR}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy || true
# shellcheck disable=SC1091
source miner_env/bin/activate
export PYTHONPATH="${REPO_DIR}"

echo "=== PM2 ==="
pm2 list 2>/dev/null | grep -E "poker44_miner|name" || echo "PM2 not running"

echo "=== Axon port 8092 ==="
ss -tlnp 2>/dev/null | grep 8092 || echo "NOT LISTENING"

echo "=== Model env ==="
grep '^POKER44_MODEL' .env 2>/dev/null || echo ".env missing model settings"

echo "=== Recent micro-session scores ==="
pm2 logs poker44_miner --lines 5000 --nostream 2>/dev/null | grep 'Scored .* micro-session' | tail -5 || echo "No micro-session scores yet"

echo "=== Deploy decision ==="
if [ -f data/benchmark/v3_deploy_state.json ]; then
  cat data/benchmark/v3_deploy_state.json
else
  python deploy/select_deploy_model.py
fi

echo "=== Dashboard status ==="
python deploy/v3_status.py --uid "${POKER44_UID:-79}"
