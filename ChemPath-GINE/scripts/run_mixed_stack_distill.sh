#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-1,2,3,4,5}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.5}"
DISTILL_VARIANT="${DISTILL_VARIANT:-gine_desc_concat}"

CONFIGS=(
  "configs/esol_gine_desc16.yaml"
  "configs/esol_gine_desc16_scaffold.yaml"
  "configs/lipo_gine_desc16.yaml"
  "configs/lipo_gine_desc16_scaffold.yaml"
  "configs/freesolv_gine_desc16.yaml"
  "configs/freesolv_gine_desc16_scaffold.yaml"
)

IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "${cfg}" .yaml)"
  for seed in "${SEED_ARRAY[@]}"; do
    echo "[mixed] ${name} seed=${seed}"
    python scripts/run_prediction_stack.py \
      --expert-dirs "sota_runs/${name}/tabular_stack/runs/seed${seed}_ridge,sota_runs/${name}/graph_oof/runs/seed${seed}_gine_desc_concat" \
      --meta-model ridge \
      --seed "${seed}" \
      --out "sota_runs/${name}/mixed_stack/runs/seed${seed}_ridge"

    echo "[distill] ${name} seed=${seed}"
    python scripts/run_distill.py \
      --base-config "${cfg}" \
      --split-dir "sota_runs/${name}/tabular_stack/splits/seed${seed}" \
      --teacher-run "sota_runs/${name}/mixed_stack/runs/seed${seed}_ridge" \
      --distill-weight "${DISTILL_WEIGHT}" \
      --variant "${DISTILL_VARIANT}" \
      --seed "${seed}" \
      --out "sota_runs/${name}/distill/seed${seed}"
  done
done
