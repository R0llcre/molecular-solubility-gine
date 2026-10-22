# Experiment Protocol

This repository is for SOTA exploration, so every result must preserve strict
test discipline.

## Splits

Each dataset run uses an outer split:

```text
train / val / test
```

For scaffold split, record whether the scaffold partition changes by seed. If
the partition is deterministic, report the runs as:

```text
same scaffold partition + different training seeds
```

Audit every new split protocol with:

```bash
python scripts/audit_splits.py \
  --configs configs/esol_gine_desc16.yaml,configs/esol_gine_desc16_scaffold.yaml \
  --seeds 1,2,3,4,5 \
  --out reports/split_audit
```

The audit writes target distribution, scaffold size distribution, acyclic
fraction, and seed-to-seed test overlap tables under `reports/split_audit/`.

## Feature Discipline

For every outer split:

```text
SMILES cleaning:
  before split

descriptor/fingerprint scaler:
  fit train only

feature selection:
  fit train only

PCA or learned tabular transforms:
  fit train only

target scaler:
  fit train only
```

No target-dependent operation may see val/test except for validation selection
or final reporting.

## OOF Stacking

For each expert:

```text
1. Split the outer train into K inner folds.
2. Fit on K-1 folds.
3. Predict the held-out fold to form OOF train predictions.
4. Predict outer val/test from each fold model.
5. Average fold predictions for outer val/test.
```

The meta learner trains on OOF train predictions only:

```text
X_meta_train = expert OOF predictions on outer train
y_meta_train = outer train target
```

Val can be used to choose meta-model families. Test must not be used for model
selection.

## Expert Pruning

Pruned stacks may remove weak or redundant experts, but the selection rule must
use only train OOF predictions:

```text
1. Rank experts by train OOF RMSE.
2. Keep top-k experts.
3. Add diverse experts whose correlation with selected experts is below the
   configured threshold.
4. Fit meta learner on selected OOF columns.
```

Pruned stack diagnostics must include:

```text
expert_selection_report.csv
leave_one_expert_out.csv
meta_coefficients_ranked.csv
expert_correlation.csv
```

## Required Stack Outputs

Each level-0 expert directory should write:

```text
expert_oof_train.csv
expert_val_predictions.csv
expert_test_predictions.csv
summary.json or summary.csv
```

Mixed and pruned stack directories should additionally write:

```text
meta_coefficients.csv
expert_correlation.csv
mixed_stack_summary.json/csv or pruned_stack_summary.json/csv
```

`source_index` must be preserved so errors can be traced back to the original
CSV.

## Reporting

Always report:

```text
RMSE mean +/- std
MAE mean +/- std
seed-wise paired delta vs current best
win rate
expert correlation matrix
stacker coefficients
top residual molecules
```

Do not claim an expert contributes causally unless it improves over the best
single expert or receives stable nonzero stack weight under OOF evaluation.

Generate seed-level diagnostics with:

```bash
python scripts/analyze_expert_pool.py --root sota_runs_v2 --baseline gine_desc
```
