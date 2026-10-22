# ChemPath-GINE

ChemPath-GINE is a molecular property prediction project with a controlled
comparison between descriptor-enhanced GINE and exploratory chemically typed
path attention.

```text
local edge-aware GINE message passing
  + physicochemical descriptor fusion
  + train-only descriptor/fingerprint/fragment tabular experts
  + OOF stacking for strict multi-view evaluation
  + optional all-shortest-path chemically typed attention bias
```

The current experimental readout is conservative: `gine_desc` is the most robust
model across ESOL/Lipophilicity random and scaffold splits; ChemPath attention is
kept as an exploratory module because it has not yet consistently passed
zero/shuffle-path negative controls.

The core ablation targets are:

```text
gine            ordinary GINE baseline
gine_attn       GINE + vanilla dense self-attention
gine_spd        GINE + shortest-path-distance attention bias
chempath_gine   GINE + all-shortest-path chemically typed path attention
gine_desc       GINE + descriptor-conditioned graph fusion
chempath_desc   ChemPath-GINE + descriptor-conditioned fusion/gating
```

ChemPath attention uses a structured bias decomposition:

```text
B_path =
  distance embedding
  + shortest-path count projection
  + bond/path count projection
  + relation embedding (donor/acceptor, same-ring, motif-touch, disconnected)
```

The bias is multiplied by a small learnable scale initialized near zero, so the
attention layer starts close to ordinary self-attention and learns how much
chemical path information to use.

## Install

Use a Python 3.10 environment with PyTorch, PyG, RDKit, pandas, numpy, scipy,
scikit-learn, and PyYAML. If you use pip for PyG extensions, make sure the
wheel index matches your Torch/CUDA version.

## Data

Input CSV must contain:

```text
SMILES
target column, for example Solubility
optional descriptor columns
```

Descriptor columns are opt-in and explicitly listed in each config to avoid
accidentally leaking numeric columns. Invalid SMILES are canonicalized/dropped
before split and scaler fitting; `source_index` is kept for prediction tracing.

Downloaded MoleculeNet/DeepChem datasets are included under `data/`:

```text
data/processed/esol.csv   target: Solubility
data/processed/lipo.csv   target: Lipophilicity
configs/esol.yaml
configs/lipo.yaml
configs/esol_gine_desc.yaml
configs/lipo_gine_desc.yaml
```

## Train

```bash
python -m chempath_gine.training.train --config configs/default.yaml
```

`configs/default.yaml` now points to `data/processed/esol_with_desc.csv` and
uses `gine_desc` with the 16-descriptor PhysChem set, the strongest current
default direction. Ready-made variant configs are
available:

```text
configs/gine.yaml
configs/gine_attn.yaml
configs/gine_spd.yaml
configs/chempath_gine.yaml
configs/chempath_desc.yaml
configs/esol_gine_desc.yaml
configs/lipo_gine_desc.yaml
configs/esol_gine_desc16.yaml
configs/lipo_gine_desc16.yaml
configs/freesolv_gine_desc16.yaml
```

The script creates a timestamped run directory with:

```text
runs/<run_name>/splits/*.csv
runs/<run_name>/scalers/*.npz
runs/<run_name>/best.pt
runs/<run_name>/test_predictions.csv
```

## Predict

```bash
python -m chempath_gine.training.predict \
  --checkpoint runs/<run_name>/best.pt \
  --csv new_molecules.csv \
  --output predictions.csv
```

The prediction command loads the training config and scaler values from the
checkpoint, cleans/canonicalizes input SMILES, preserves `source_index`, and
writes `y_pred`.

## Key Files

```text
chempath_gine/features/chem_path.py        all-shortest-path chemical features
chempath_gine/models/chem_path_attention.py structured path-bias attention
chempath_gine/models/chem_path_gine.py      GINE baseline and ChemPath-GINE
chempath_gine/data/dataset.py               scaler-safe cached dataset
```

## Suggested Ablation

```bash
python scripts/run_ablation.py \
  --config configs/esol_fusion_ablation_random.yaml \
  --variants gine,gine_desc,gine_desc_concat,gine_desc_residual_add,gine_desc_film_graph,gine_desc_lowrank_bilinear \
  --seeds 1,2,3,4,5 \
  --out ablation_runs/esol_fusion_random_5seed
```

| Variant | Edge Attr | Attention | SPD Bias | Chem Path Bias | Descriptor Fusion |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gine` | yes | no | no | no | no |
| `gine_attn` | yes | yes | no | no | no |
| `gine_spd` | yes | yes | yes | no | no |
| `chempath_gine` | yes | yes | yes | yes | no |
| `chempath_desc` | yes | yes | yes | yes | yes |

The most important empirical comparison is now `gine` versus `gine_desc` and
descriptor-fusion variants. ChemPath-specific comparisons should include
`chempath_desc_full`, `chempath_desc_zero_attr_same_params`, and
`chempath_desc_shuffle_bucket` to test whether path features are actually used.

For scaffold split, the default splitter is deterministic. A 5-seed scaffold run
uses the same scaffold partition with five training random seeds, not five
independent scaffold partitions.

## SOTA-System Track

ChemPath attention is no longer treated as the primary route to better scores.
The current SOTA-oriented track is:

```text
PhysChem-GINE / MolFuse-GINE
  = GINE + descriptor fusion
  + train-only selected RDKit descriptors
  + Morgan/MACCS/RDKit/AtomPair/Torsion fingerprints
  + optional frozen pretrained SMILES embeddings
  + optional conformer-derived 3D descriptors
  + BRICS and functional-group tabular experts
  + OOF stacking
  + optional teacher-student distillation
```

Build an inspection feature matrix:

```bash
python scripts/build_feature_bank.py \
  --input data/processed/esol_with_desc.csv \
  --output feature_runs/esol_desc16_morgan2_maccs.csv \
  --feature-set desc16_morgan2_maccs
```

Run tabular descriptor/fingerprint baselines under the same split protocol:

```bash
python scripts/run_tabular_baselines.py \
  --config configs/esol_gine_desc16.yaml \
  --feature-sets desc16,morgan2,desc16_morgan2,desc16_morgan2_maccs,rdkit2d_selected32_morgan2_maccs \
  --models ridge,elasticnet,svr,rf,xgb \
  --seeds 1,2,3,4,5 \
  --out tabular_runs/esol_random
```

Run a first strict OOF stack over tabular experts:

```bash
python scripts/run_oof_stack.py \
  --config configs/esol_gine_desc16.yaml \
  --experts desc16:ridge,desc16_morgan2:xgb,desc16_morgan2_maccs:xgb,rdkit2d_selected32_morgan2_maccs:ridge \
  --seeds 1,2,3,4,5 \
  --folds 5 \
  --meta-model ridge \
  --out stack_runs/esol_random
```

All descriptor selection, missing-value imputation, scaling, BRICS vocabularies,
and OOF meta-features are fit from the train fold only. Test predictions are
written once per held-out split.

Add BRICS/functional-group/3D features to a tabular run by including feature-set
tokens:

```text
rdkit2d_selected32_morgan2_maccs_fg
desc16_morgan2_maccs_brics256
desc16_morgan2_maccs_conf3d10
```

Run graph experts with OOF predictions, then combine graph and tabular experts:

```bash
python scripts/run_graph_oof_experts.py \
  --config configs/esol_gine_desc16.yaml \
  --variants gine_desc \
  --seeds 1 \
  --folds 5 \
  --out graph_oof_runs/esol_random

python scripts/run_prediction_stack.py \
  --expert-dirs stack_runs/esol_random/runs/seed1_ridge,graph_oof_runs/esol_random/runs/seed1_gine_desc \
  --meta-model ridge \
  --seed 1 \
  --out stack_runs/esol_random/mixed_seed1
```

Distill a stacked teacher into a single `gine_desc` student:

```bash
python scripts/run_distill.py \
  --base-config configs/esol_gine_desc16.yaml \
  --split-dir stack_runs/esol_random/splits/seed1 \
  --teacher-run stack_runs/esol_random/runs/seed1_ridge \
  --distill-weight 0.5 \
  --variant gine_desc \
  --out distill_runs/esol_random_seed1
```

Run a frozen pretrained SMILES expert when `torch` and `transformers` are
available:

```bash
python scripts/run_pretrained_smiles.py \
  --config configs/esol_gine_desc16.yaml \
  --model-name DeepChem/ChemBERTa-77M-MLM \
  --head-models ridge,elasticnet,xgb \
  --random-smiles 16 \
  --seeds 1,2,3,4,5 \
  --out pretrained_smiles_runs/esol_random
```

Run the first-stage SOTA sweep on a configured machine:

```bash
SEEDS=1,2,3,4,5 bash scripts/run_sota_stage1.sh
```
