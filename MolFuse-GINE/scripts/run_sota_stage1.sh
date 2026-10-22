#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-1,2,3,4,5}"
GRAPH_VARIANTS="${GRAPH_VARIANTS:-gine,gine_desc,gine_desc_concat,gine_desc_residual_add,gine_desc_film_graph,gine_desc_lowrank_bilinear}"
TABULAR_MODELS="${TABULAR_MODELS:-ridge,elasticnet,svr,rf,xgb,lightgbm,catboost}"
TABULAR_FEATURES="${TABULAR_FEATURES:-desc16,morgan2,desc16_morgan2,desc16_morgan2_maccs,rdkit2d_selected32_morgan2_maccs,rdkit2d_selected32_morgan2_maccs_fg}"
STACK_EXPERTS="${STACK_EXPERTS:-desc16:ridge,desc16_morgan2:xgb,desc16_morgan2_maccs:xgb,rdkit2d_selected32_morgan2_maccs:ridge}"

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
  echo "[stage1] graph ablation ${name}"
  python scripts/run_ablation.py \
    --config "${cfg}" \
    --variants "${GRAPH_VARIANTS}" \
    --seeds "${SEEDS}" \
    --out "sota_runs/${name}/graph_ablation"

  echo "[stage1] tabular experts ${name}"
  python scripts/run_tabular_baselines.py \
    --config "${cfg}" \
    --feature-sets "${TABULAR_FEATURES}" \
    --models "${TABULAR_MODELS}" \
    --seeds "${SEEDS}" \
    --out "sota_runs/${name}/tabular"

  echo "[stage1] tabular OOF stack ${name}"
  python scripts/run_oof_stack.py \
    --config "${cfg}" \
    --experts "${STACK_EXPERTS}" \
    --seeds "${SEEDS}" \
    --folds 5 \
    --meta-model ridge \
    --out "sota_runs/${name}/tabular_stack"
done
