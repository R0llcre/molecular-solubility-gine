#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-1,2,3,4,5}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-DeepChem/ChemBERTa-77M-MLM}"
RANDOM_SMILES="${RANDOM_SMILES:-16}"
TABULAR_MODELS="${TABULAR_MODELS:-ridge,xgb,lightgbm,catboost}"
EXTRA_TABULAR_FEATURES="${EXTRA_TABULAR_FEATURES:-rdkit2d_selected32_morgan2_maccs_fg_brics512,rdkit2d_selected32_morgan2_maccs_fg_conf3d3}"
GRAPH_OOF_VARIANTS="${GRAPH_OOF_VARIANTS:-gine_desc_concat}"

CONFIGS=(
  "configs/esol_gine_desc16.yaml"
  "configs/esol_gine_desc16_scaffold.yaml"
  "configs/lipo_gine_desc16.yaml"
  "configs/lipo_gine_desc16_scaffold.yaml"
  "configs/freesolv_gine_desc16.yaml"
  "configs/freesolv_gine_desc16_scaffold.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "${cfg}" .yaml)"

  echo "[stage2] fragment/3d tabular experts ${name}"
  python scripts/run_tabular_baselines.py \
    --config "${cfg}" \
    --feature-sets "${EXTRA_TABULAR_FEATURES}" \
    --models "${TABULAR_MODELS}" \
    --seeds "${SEEDS}" \
    --out "sota_runs/${name}/tabular_extra"

  echo "[stage2] pretrained SMILES expert ${name}"
  python scripts/run_pretrained_smiles.py \
    --config "${cfg}" \
    --model-name "${PRETRAINED_MODEL}" \
    --head-models "ridge,elasticnet,xgb" \
    --random-smiles "${RANDOM_SMILES}" \
    --seeds "${SEEDS}" \
    --out "sota_runs/${name}/pretrained_smiles"

  echo "[stage2] graph OOF expert ${name}"
  python scripts/run_graph_oof_experts.py \
    --config "${cfg}" \
    --variants "${GRAPH_OOF_VARIANTS}" \
    --seeds "${SEEDS}" \
    --folds 5 \
    --out "sota_runs/${name}/graph_oof"
done
