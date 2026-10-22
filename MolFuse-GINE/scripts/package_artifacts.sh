#!/usr/bin/env bash
set -euo pipefail

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-packaged_artifacts}"
PKG_DIR="${OUT_DIR}/MolFuse-GINE-sota-${STAMP}"
INCLUDE_RESULTS="${INCLUDE_RESULTS:-0}"
mkdir -p "${PKG_DIR}"

"${PYTHON:-python3}" scripts/summarize_sota_results.py --root sota_runs_v2 --out reports/sota_search || true

RESULT_EXCLUDES=()
if [[ "${INCLUDE_RESULTS}" != "1" ]]; then
  RESULT_EXCLUDES+=(--exclude 'sota_runs*/')
  RESULT_EXCLUDES+=(--exclude 'reports/split_audit/**/split_files/')
  RESULT_EXCLUDES+=(--exclude 'runs/')
fi

RSYNC_ARGS=(
  -a
  --exclude '__pycache__/'
  --exclude '.pytest_cache/'
  --exclude 'cache/'
  --exclude 'ablation_runs/'
  --exclude 'baseline_runs/'
  --exclude 'tabular_runs/'
  --exclude 'stack_runs/'
  --exclude 'graph_oof_runs/'
  --exclude 'pretrained_smiles_runs/'
  --exclude 'pretrained_smiles_oof_runs/'
  --exclude 'distill_runs/'
  --exclude 'feature_runs/'
  --exclude '*.pt'
  --exclude '*.pth'
  --exclude '*.npz'
  --exclude '*.joblib'
  --exclude '*.pkl'
  --exclude 'packaged_artifacts/'
)
if [[ ${#RESULT_EXCLUDES[@]} -gt 0 ]]; then
  RSYNC_ARGS+=("${RESULT_EXCLUDES[@]}")
fi
RSYNC_ARGS+=(./ "${PKG_DIR}/")

rsync "${RSYNC_ARGS[@]}"

if command -v git >/dev/null 2>&1; then
  git status --short > "${PKG_DIR}/GIT_STATUS.txt" || true
  git log --oneline -20 > "${PKG_DIR}/GIT_LOG.txt" || true
fi

tar -czf "${OUT_DIR}/MolFuse-GINE-sota-${STAMP}.tar.gz" -C "${OUT_DIR}" "MolFuse-GINE-sota-${STAMP}"
echo "${OUT_DIR}/MolFuse-GINE-sota-${STAMP}.tar.gz"
