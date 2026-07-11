#!/bin/bash
# Start the Poker44 baseline miner under PM2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_DIR}"

if [ -f "${PROJECT_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/.env"
  set +a
fi

NETUID="${NETUID:-126}"
WALLET_NAME="${WALLET_NAME:-yaroslav-coldkey}"
HOTKEY="${HOTKEY:-yaroslav-poker44-hotkey}"
WALLET_PATH="${WALLET_PATH:-/root/.bittensor/wallets}"
NETWORK="${NETWORK:-finney}"
MINER_SCRIPT="${MINER_SCRIPT:-./neurons/miner_baseline.py}"
PM2_NAME="${PM2_NAME:-poker44_miner}"
AXON_PORT="${AXON_PORT:-8092}"
ALLOWED_VALIDATOR_HOTKEYS="${ALLOWED_VALIDATOR_HOTKEYS:-}"
MODEL_PATH="${POKER44_MODEL_PATH:-./models/hybrid.joblib}"
LAUNCHER="${PROJECT_DIR}/scripts/miner/run/poker44_miner_launcher.sh"

if [ ! -d "miner_env" ]; then
  echo "Error: miner_env not found. Run ./scripts/miner/setup.sh first."
  exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
  echo "Error: model not found at ${MODEL_PATH}"
  echo "Run: poker44-miner train-hybrid"
  exit 1
fi

if [ ! -f "${MINER_SCRIPT}" ]; then
  echo "Error: Miner script not found at ${MINER_SCRIPT}"
  exit 1
fi

if ! command -v pm2 >/dev/null 2>&1; then
  echo "Error: PM2 is not installed. Install with: npm install -g pm2"
  exit 1
fi

ALLOWLIST_ARGS=""
if [ -n "${ALLOWED_VALIDATOR_HOTKEYS}" ]; then
  read -r -a VALIDATOR_HOTKEY_ARRAY <<< "${ALLOWED_VALIDATOR_HOTKEYS}"
  for hk in "${VALIDATOR_HOTKEY_ARRAY[@]}"; do
    ALLOWLIST_ARGS+=" --blacklist.allowed_validator_hotkeys ${hk}"
  done
else
  ALLOWLIST_ARGS=" --blacklist.force_validator_permit"
fi

cat > "${LAUNCHER}" <<EOF
#!/bin/bash
set -euo pipefail
cd "${PROJECT_DIR}"
if [ -f "${PROJECT_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/.env"
  set +a
fi
source miner_env/bin/activate
export PYTHONPATH="${PROJECT_DIR}"
export BT_NO_PARSE_CLI_ARGS=0
export POKER44_MODEL_PATH="${MODEL_PATH}"
exec python "${MINER_SCRIPT}" \\
  --netuid ${NETUID} \\
  --wallet.name ${WALLET_NAME} \\
  --wallet.hotkey ${HOTKEY} \\
  --wallet.path ${WALLET_PATH} \\
  --subtensor.network ${NETWORK} \\
  --axon.port ${AXON_PORT} \\
  --logging.info${ALLOWLIST_ARGS}
EOF
chmod +x "${LAUNCHER}"

pm2 delete "${PM2_NAME}" 2>/dev/null || true
pm2 start "${LAUNCHER}" --name "${PM2_NAME}"
pm2 save

echo "Miner started: ${PM2_NAME}"
echo "View logs: pm2 logs ${PM2_NAME}"
echo "Config: netuid=${NETUID} network=${NETWORK} wallet=${WALLET_NAME} hotkey=${HOTKEY} axon_port=${AXON_PORT}"
