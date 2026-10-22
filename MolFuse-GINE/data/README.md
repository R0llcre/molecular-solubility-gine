# Downloaded Datasets

Raw files:

```text
data/raw/esol_delaney_processed.csv
data/raw/lipophilicity.csv
data/raw/freesolv_sampl.csv
```

Training-ready normalized files:

```text
data/processed/esol.csv   columns: SMILES, Solubility
data/processed/esol_with_desc.csv   columns: SMILES, Solubility, RDKit descriptors
data/processed/lipo.csv   columns: SMILES, Lipophilicity
data/processed/lipo_with_desc.csv   columns: SMILES, Lipophilicity, RDKit descriptors
data/processed/freesolv.csv   columns: SMILES, FreeSolv
data/processed/freesolv_with_desc.csv   columns: SMILES, FreeSolv, RDKit descriptors
```

Sources:

```text
ESOL: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv
Lipophilicity: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv
FreeSolv: https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/molnet_publish/FreeSolv.zip
```

Configs:

```text
configs/esol.yaml
configs/esol_gine_desc.yaml
configs/lipo.yaml
configs/lipo_gine_desc.yaml
configs/freesolv_gine_desc.yaml
```
