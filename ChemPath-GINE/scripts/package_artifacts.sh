#!/usr/bin/env bash
set -euo pipefail

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-packaged_artifacts}"
PKG_DIR="${OUT_DIR}/ChemPath-GINE-sota-${STAMP}"
mkdir -p "${PKG_DIR}"

python scripts/summarize_sota_results.py --root sota_runs --out reports/sota_system || true

rsync -a \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'cache/' \
  --exclude 'runs/' \
  --exclude 'ablation_runs/' \
  --exclude 'baseline_runs/' \
  --exclude 'tabular_runs/' \
  --exclude 'stack_runs/' \
  --exclude 'graph_oof_runs/' \
  --exclude 'pretrained_smiles_runs/' \
  --exclude 'distill_runs/' \
  --exclude 'feature_runs/' \
  --exclude 'packaged_artifacts/' \
  ./ "${PKG_DIR}/"

if command -v git >/dev/null 2>&1; then
  git status --short > "${PKG_DIR}/GIT_STATUS.txt" || true
  git log --oneline -20 > "${PKG_DIR}/GIT_LOG.txt" || true
fi

tar -czf "${OUT_DIR}/ChemPath-GINE-sota-${STAMP}.tar.gz" -C "${OUT_DIR}" "ChemPath-GINE-sota-${STAMP}"
echo "${OUT_DIR}/ChemPath-GINE-sota-${STAMP}.tar.gz"
