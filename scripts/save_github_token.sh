#!/usr/bin/env bash
# Save a GitHub personal access token outside the repo (for git push only).
set -euo pipefail

TARGET="${HOME}/.github_token"

if [ -f "${TARGET}" ]; then
  echo "Token file already exists at ${TARGET}"
  read -r -p "Overwrite? [y/N] " answer
  if [[ ! "${answer}" =~ ^[Yy]$ ]]; then
    exit 0
  fi
fi

read -r -s -p "Paste GitHub token (input hidden): " token
echo
if [ -z "${token}" ]; then
  echo "No token entered." >&2
  exit 1
fi

umask 077
printf '%s' "${token}" > "${TARGET}"
chmod 600 "${TARGET}"
unset token

echo "Saved token to ${TARGET} (mode 600)."
echo "This file is outside the repo and will not be committed."
