#!/bin/bash
set -euo pipefail
cd "/root/workspaces/projects/poker44"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy || true
if [ -f "/root/workspaces/projects/poker44/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "/root/workspaces/projects/poker44/.env"
  set +a
fi
source miner_env/bin/activate
export PYTHONPATH="/root/workspaces/projects/poker44"
export BT_NO_PARSE_CLI_ARGS=0
export POKER44_MODEL_FACTORY="${POKER44_MODEL_FACTORY:-deploy.micro_session_model:create_model}"
export POKER44_MODEL_PATH="${POKER44_MODEL_PATH:-./models/micro_session_v1.joblib}"
export POKER44_MODEL_VERSION="${POKER44_MODEL_VERSION:-micro-v1}"
exec python "./neurons/miner.py" \
  --netuid 126 \
  --wallet.name yaroslav-coldkey \
  --wallet.hotkey yaroslav-poker44-hotkey \
  --wallet.path /root/.bittensor/wallets \
  --subtensor.network finney \
  --axon.port 8092 \
  --logging.info --blacklist.force_validator_permit
