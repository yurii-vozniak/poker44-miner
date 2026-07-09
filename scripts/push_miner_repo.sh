#!/usr/bin/env bash
# Push the local miner commit to yurii-vozniak/poker44-miner using ~/.github_token.
set -euo pipefail

REPO_DIR="/root/workspaces/projects/poker44"
TOKEN_FILE="${HOME}/.github_token"

if [ ! -f "${TOKEN_FILE}" ]; then
  echo "Missing ${TOKEN_FILE}. Run: bash ${REPO_DIR}/scripts/save_github_token.sh" >&2
  exit 1
fi

cd "${REPO_DIR}"
TOKEN="$(<"${TOKEN_FILE}")"
if [ -z "${TOKEN}" ]; then
  echo "Token file is empty." >&2
  exit 1
fi

git push "https://yurii-vozniak:${TOKEN}@github.com/yurii-vozniak/poker44-miner.git" main
echo "Push complete."
