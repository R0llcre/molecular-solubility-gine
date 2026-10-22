#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-1,2,3,4,5}"
ROOT="${ROOT:-sota_runs_v2}"
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
    python scripts/run_mixed_stack.py \
      --expert-dirs "${ROOT}/${name}/tabular_stack/runs/seed${seed}_ridge,${ROOT}/${name}/graph_oof/runs/seed${seed}_gine_desc_concat" \
      --meta-model ridge \
      --seed "${seed}" \
      --stack-name "stage_shell_mixed_stack" \
      --out "${ROOT}/${name}/mixed_stack/runs/seed${seed}_ridge"

    echo "[distill] ${name} seed=${seed}"
    python scripts/run_distill.py \
      --base-config "${cfg}" \
      --split-dir "${ROOT}/${name}/tabular_stack/splits/seed${seed}" \
      --teacher-run "${ROOT}/${name}/mixed_stack/runs/seed${seed}_ridge" \
      --distill-weight "${DISTILL_WEIGHT}" \
      --variant "${DISTILL_VARIANT}" \
      --seed "${seed}" \
      --out "${ROOT}/${name}/distill/seed${seed}"
  done
done
