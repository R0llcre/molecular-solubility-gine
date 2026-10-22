# MolFuse-GINE Roadmap

## Round 1: Formal Mixed Stack

Goal: verify whether graph + tabular/fingerprint experts improve over the best
single expert under OOF.

Experts:

```text
gine_desc
gine_desc_concat
desc16_morgan2_maccs:ridge
desc16_morgan2_maccs:xgb
desc16_morgan2_maccs:lightgbm
rdkit2d_selected64_morgan2_maccs:ridge
rdkit2d_selected64_morgan2_maccs:xgb
```

Datasets:

```text
ESOL random/scaffold
Lipo random/scaffold
FreeSolv random/scaffold
```

## Round 2: Stronger Tabular Experts

Add:

```text
selected64 / selected128 descriptors
Morgan2 / Morgan3 / MACCS
BRICS fragments
functional group counts
3D conformer descriptors
```

## Round 3: Graph Backbone Diversity

Add:

```text
PNA-desc16
D-MPNN-desc16
AttentiveFP-desc16
GINE-desc16 + Huber
GINE-desc16 + attention pooling
```

First-pass implemented variants:

```text
pna_desc
dmpnn_desc
attentivefp_desc
```

## Round 3b: Split Audit and Expert Pruning

Before adding more experts, audit the split protocol and prune weak/noisy stack
inputs on train OOF predictions:

```bash
python scripts/audit_splits.py \
  --configs configs/esol_gine_desc16.yaml,configs/esol_gine_desc16_scaffold.yaml \
  --seeds 1,2,3,4,5 \
  --out reports/split_audit

python scripts/run_sota_search.py \
  --plan configs/search/round3_pruned_mixed_stack.yaml \
  --dry-run
```

Round 3 pruning writes:

```text
expert_selection_report.csv
leave_one_expert_out.csv
meta_coefficients_ranked.csv
expert_correlation.csv
```

Selection is based on train OOF RMSE plus prediction-correlation diversity.
Test remains final reporting only.

## Round 4: Pretrained Embeddings

Add frozen embedding experts:

```text
ChemBERTa family
MolFormer family
SELFIES encoder
ChemFM, if available
```

Use randomized SMILES mean/std pooling and small supervised heads.

OOF entrypoint:

```bash
python scripts/run_pretrained_smiles_oof.py --config configs/esol_gine_desc16.yaml
```

Pretrained experts can now concatenate frozen SMILES embeddings with train-only
processed tabular feature sets:

```bash
python scripts/run_pretrained_smiles_oof.py \
  --config configs/esol_gine_desc16.yaml \
  --feature-sets embedding,desc16_morgan2,rdkit2d_descsel64_morgan2_maccs
```

## Round 5: Distillation

Use the best OOF stack as a teacher:

```text
loss = MSE(y_true, y_student) + lambda * MSE(y_teacher, y_student)
```

Report both:

```text
best stacked system
best distilled single model
```
