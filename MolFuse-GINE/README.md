# MolFuse-GINE

MolFuse-GINE is a SOTA search system for small-molecule regression tasks. The
main target is not a single attention module. The system compares and stacks
multiple molecular views under a strict train-only preprocessing protocol.

```text
graph experts
+ descriptor and fingerprint tabular experts
+ pretrained SMILES embedding experts
+ fragment / functional-group experts
+ optional 3D conformer experts
+ optional ChemPath legacy expert
+ OOF stacking
+ distilled single student
```

The project starts from the finding that `GINE + physicochemical descriptors`
is more reliable than ChemPath path attention on ESOL and Lipophilicity.
ChemPath is kept as an optional diagnostic expert, not as the main claim.

## Current Direction

Primary model family:

```text
PhysChem-GINE / MolFuse-GINE
= graph encoder + train-only descriptors + fingerprints + expert stacking
```

Primary system output:

```text
best system:
  OOF-stacked expert ensemble

best single model:
  descriptor-enhanced GINE student distilled from the stack
```

## Repository Layout

```text
molfuse_gine/
  baselines/       tabular model builders
  data/            clean, split, scaler-safe datasets
  distill/         teacher prediction attachment
  features/        descriptors, fingerprints, fragments, 3D, SMILES embeddings
  models/          GINE and legacy ChemPath variants
  selection/       train-only feature selection
  stacking/        OOF tabular and mixed expert stacking
  training/        graph training, prediction, metrics

scripts/
  run_sota_search.py          YAML-driven search runner
  run_tabular_baselines.py    descriptor/fingerprint/fragment/3D experts
  run_oof_stack.py            tabular OOF stack
  run_graph_oof_experts.py    graph OOF experts
  run_pretrained_smiles.py    frozen SMILES embedding baselines
  run_pretrained_smiles_oof.py frozen SMILES embedding OOF experts
  run_mixed_stack.py          formal mixed stack over prediction dirs
  analyze_expert_pool.py      seed-wise deltas, win rates, diagnostics
  run_distill.py              stacked-teacher distillation

configs/search/
  round1_mixed_stack.yaml
  round2_pretrained_fragment_3d.yaml
```

## Install

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

For PyTorch Geometric, use wheels matching your Torch/CUDA version when needed.

## Data

Included processed datasets:

```text
data/processed/esol_with_desc.csv       target: Solubility
data/processed/lipo_with_desc.csv       target: Lipophilicity
data/processed/freesolv_with_desc.csv   target: FreeSolv
```

The data pipeline keeps test discipline explicit:

```text
clean/canonicalize SMILES before split
fit scalers only on train
select descriptors only on train
fit OOF experts only inside train folds
use test only for final evaluation
```

## Quick Checks

```bash
python -m compileall -q molfuse_gine scripts tests
python -m pytest -q
```

## Single Graph Baseline

```bash
python -m molfuse_gine.training.train --config configs/default.yaml
```

The default config uses `gine_desc` with the 16-descriptor PhysChem set.

## SOTA Search

Dry-run the first search round:

```bash
python scripts/run_sota_search.py \
  --plan configs/search/round1_mixed_stack.yaml \
  --dry-run
```

Run it for real:

```bash
python scripts/run_sota_search.py \
  --plan configs/search/round1_mixed_stack.yaml
```

Run the extra fragment/3D/pretrained stage:

```bash
python scripts/run_sota_search.py \
  --plan configs/search/round2_pretrained_fragment_3d.yaml
```

## Formal Mixed Stack

After individual experts produce `expert_oof_train.csv`,
`expert_val_predictions.csv`, and `expert_test_predictions.csv`, stack them:

```bash
python scripts/run_mixed_stack.py \
  --expert-dirs sota_runs_v2/esol_gine_desc16/tabular_stack/runs/seed1_ridge,sota_runs_v2/esol_gine_desc16/graph_oof/runs/seed1_gine_desc_concat \
  --meta-model ridge \
  --seed 1 \
  --out sota_runs_v2/esol_gine_desc16/mixed_stack/runs/seed1_ridge
```

Outputs:

```text
expert_oof_train.csv
expert_val_predictions.csv
expert_test_predictions.csv
mixed_stack_summary.json
mixed_stack_summary.csv
meta_coefficients.csv
meta_coefficients_ranked.csv
expert_correlation.csv
```

Collect leaderboards and diagnostics:

```bash
python scripts/summarize_sota_results.py \
  --root sota_runs_v2 \
  --out reports/sota_search

python scripts/analyze_expert_pool.py \
  --root sota_runs_v2 \
  --out reports/sota_search_diagnostics \
  --baseline gine_desc
```

## Roadmap

1. Formalize mixed stack and diagnostics.
2. Strengthen tabular/fingerprint experts with selected64/128 + Morgan/MACCS.
3. Add graph diversity: PNA-desc, D-MPNN-desc, AttentiveFP-desc.
4. Add stronger pretrained SMILES embeddings with randomized SMILES pooling.
5. Add fragment/functional-group and 3D conformer experts.
6. Distill the best stacked teacher into a deployable PhysChem-GINE student.

See [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) for the strict
evaluation contract.
