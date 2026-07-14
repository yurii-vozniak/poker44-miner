#!/bin/bash
set -euo pipefail
cd "/root/workspaces/projects/poker44"
if [ -f "/root/workspaces/projects/poker44/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "/root/workspaces/projects/poker44/.env"
  set +a
fi
source miner_env/bin/activate
export PYTHONPATH="/root/workspaces/projects/poker44"
export BT_NO_PARSE_CLI_ARGS=0
export POKER44_MODEL_PATH="./models/hybrid.joblib"
exec python "./neurons/miner_hybrid.py" \
  --netuid 126 \
  --wallet.name yaroslav-coldkey \
  --wallet.hotkey yaroslav-poker44-hotkey \
  --wallet.path /root/.bittensor/wallets \
  --subtensor.network finney \
  --axon.port 8092 \
  --logging.info --blacklist.force_validator_permit
