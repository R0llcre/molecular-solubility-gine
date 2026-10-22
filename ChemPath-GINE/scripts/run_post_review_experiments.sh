#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-1,2,3,4,5}"
PYTHON="${PYTHON:-python}"
read -r -a PYTHON_CMD <<< "${PYTHON}"

FUSION_VARIANTS="${FUSION_VARIANTS:-gine,gine_desc,gine_desc_concat,gine_desc_residual_add,gine_desc_film_graph,gine_desc_lowrank_bilinear}"
CHEMPATH_DIAG_VARIANTS="${CHEMPATH_DIAG_VARIANTS:-chempath_desc_full,chempath_desc_zero_attr_same_params,chempath_desc_shuffle_bucket}"
FREESOLV_VARIANTS="${FREESOLV_VARIANTS:-gine,gine_desc,gine_desc_lowrank_bilinear}"
DESC_BASELINES="${DESC_BASELINES:-ridge,elasticnet,svr,rf,xgb}"

run_ablation() {
  local config="$1"
  local variants="$2"
  local out="$3"
  echo "[run_ablation] config=${config} variants=${variants} out=${out}"
  "${PYTHON_CMD[@]}" scripts/run_ablation.py \
    --config "${config}" \
    --variants "${variants}" \
    --seeds "${SEEDS}" \
    --out "${out}"
}

run_baselines() {
  local config="$1"
  local out="$2"
  echo "[run_descriptor_baselines] config=${config} models=${DESC_BASELINES} out=${out}"
  "${PYTHON_CMD[@]}" scripts/run_descriptor_baselines.py \
    --config "${config}" \
    --models "${DESC_BASELINES}" \
    --seeds "${SEEDS}" \
    --out "${out}"
}

run_ablation configs/esol_fusion_ablation_random.yaml "${FUSION_VARIANTS}" ablation_runs/esol_fusion_random_5seed
run_ablation configs/esol_fusion_ablation_scaffold.yaml "${FUSION_VARIANTS}" ablation_runs/esol_fusion_scaffold_5seed
run_ablation configs/lipo_fusion_ablation_random.yaml "${FUSION_VARIANTS}" ablation_runs/lipo_fusion_random_5seed
run_ablation configs/lipo_fusion_ablation_scaffold.yaml "${FUSION_VARIANTS}" ablation_runs/lipo_fusion_scaffold_5seed

run_ablation configs/esol_chempath_diagnostic_random.yaml "${CHEMPATH_DIAG_VARIANTS}" ablation_runs/esol_chempath_diag_random_5seed
run_ablation configs/esol_chempath_diagnostic_scaffold.yaml "${CHEMPATH_DIAG_VARIANTS}" ablation_runs/esol_chempath_diag_scaffold_5seed
run_ablation configs/lipo_chempath_diagnostic_random.yaml "${CHEMPATH_DIAG_VARIANTS}" ablation_runs/lipo_chempath_diag_random_5seed
run_ablation configs/lipo_chempath_diagnostic_scaffold.yaml "${CHEMPATH_DIAG_VARIANTS}" ablation_runs/lipo_chempath_diag_scaffold_5seed

run_ablation configs/freesolv_ablation_random.yaml "${FREESOLV_VARIANTS}" ablation_runs/freesolv_random_5seed
run_ablation configs/freesolv_ablation_scaffold.yaml "${FREESOLV_VARIANTS}" ablation_runs/freesolv_scaffold_5seed

run_baselines configs/esol_fusion_ablation_random.yaml baseline_runs/esol_random_desc_baselines_5seed
run_baselines configs/esol_fusion_ablation_scaffold.yaml baseline_runs/esol_scaffold_desc_baselines_5seed
run_baselines configs/lipo_fusion_ablation_random.yaml baseline_runs/lipo_random_desc_baselines_5seed
run_baselines configs/lipo_fusion_ablation_scaffold.yaml baseline_runs/lipo_scaffold_desc_baselines_5seed
run_baselines configs/freesolv_ablation_random.yaml baseline_runs/freesolv_random_desc_baselines_5seed
run_baselines configs/freesolv_ablation_scaffold.yaml baseline_runs/freesolv_scaffold_desc_baselines_5seed

echo "[done] post-review experiments finished"
