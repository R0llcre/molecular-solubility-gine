#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <ssh-key> <user@host>"
  echo "Example: $0 ~/.ssh/id_ed25519 user@<vm-ip>"
  exit 2
fi

KEY="$1"
TARGET="$2"
REMOTE_DIR="${REMOTE_DIR:-~/ChemPath-GINE}"
SSH_PORT="${SSH_PORT:-22}"
SSH_OPTS="-i ${KEY} -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"

rsync -az --delete \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'runs/' \
  --exclude 'cache/' \
  --exclude 'tmp_dataset_check/' \
  -e "ssh ${SSH_OPTS}" \
  ./ "${TARGET}:${REMOTE_DIR}/"

ssh ${SSH_OPTS} "${TARGET}" \
  "cd ${REMOTE_DIR} && bash scripts/setup_remote.sh"
