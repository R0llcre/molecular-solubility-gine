# ChemPath-GINE Post-Review Summary

All neural entries use 5 seeds. Checkpoints are selected by validation loss only; test metrics are evaluated once from the best validation checkpoint. Descriptor and target scalers are fit on the train split only. Scaffold partitions are deterministic, so scaffold 5-seed results vary training randomness rather than scaffold membership.

## Main Result

The robust finding is still descriptor-enhanced GINE, not ChemPath path bias as a standalone contribution.

- ESOL: `gine_desc` is best among the tested neural fusion variants on both random and scaffold splits.
- Lipo: descriptor fusion helps on scaffold; random split differences among fusion variants are tiny.
- FreeSolv: `gine_desc` beats graph-only GINE and all descriptor-only baselines on both random and scaffold splits.
- ChemPath diagnostics: strict zero-attr and bucket-shuffle controls do not show a robust causal advantage for path bias. Lipo random has a small positive signal; ESOL random and Lipo scaffold do not.

## Best Neural Results

| Dataset | Split | Best neural model | RMSE mean ± std | Comparison |
|---|---|---:|---:|---|
| ESOL | random | `gine_desc` | 0.5594 ± 0.0463 | `gine` 0.6710 ± 0.0440 |
| ESOL | scaffold | `gine_desc` | 0.9295 ± 0.0241 | `gine` 1.1633 ± 0.0419 |
| Lipo | random | `gine_desc_film_graph` | 0.6000 ± 0.0235 | `gine_desc` 0.6019 ± 0.0286 |
| Lipo | scaffold | `gine_desc_residual_add` | 0.6948 ± 0.0136 | `gine_desc` 0.6959 ± 0.0118 |
| FreeSolv | random | `gine_desc` | 1.2605 ± 0.2839 | `gine` 1.3639 ± 0.1949 |
| FreeSolv | scaffold | `gine_desc` | 2.3249 ± 0.1168 | `gine` 2.6674 ± 0.1737 |

## Fusion Ablation

| Dataset | Split | `gine` | `gine_desc` | best alternate fusion |
|---|---:|---:|---:|---:|
| ESOL | random | 0.6710 ± 0.0440 | **0.5594 ± 0.0463** | lowrank 0.5666 ± 0.0485 |
| ESOL | scaffold | 1.1633 ± 0.0419 | **0.9295 ± 0.0241** | concat 0.9392 ± 0.0114 |
| Lipo | random | 0.6071 ± 0.0215 | 0.6019 ± 0.0286 | **film_graph 0.6000 ± 0.0235** |
| Lipo | scaffold | 0.7304 ± 0.0184 | 0.6959 ± 0.0118 | **residual_add 0.6948 ± 0.0136** |
| FreeSolv | random | 1.3639 ± 0.1949 | **1.2605 ± 0.2839** | lowrank 1.3335 ± 0.2449 |
| FreeSolv | scaffold | 2.6674 ± 0.1737 | **2.3249 ± 0.1168** | lowrank 2.5078 ± 0.0475 |

## ChemPath Strict Diagnostics

| Dataset | Split | full | zero attr same params | bucket shuffle | readout |
|---|---:|---:|---:|---:|---|
| ESOL | random | 0.5874 ± 0.0369 | 0.5791 ± 0.0502 | **0.5727 ± 0.0547** | negative controls beat full |
| ESOL | scaffold | **0.9372 ± 0.0361** | 0.9479 ± 0.0260 | 0.9458 ± 0.0181 | small path signal, still below `gine_desc` |
| Lipo | random | **0.5995 ± 0.0252** | 0.6032 ± 0.0198 | 0.6005 ± 0.0305 | weak positive signal |
| Lipo | scaffold | 0.7004 ± 0.0099 | 0.7090 ± 0.0076 | **0.6972 ± 0.0163** | shuffle beats full |

Diagnostics show that learned path usage is often small. For example, ESOL path-bias absolute means are near 0.01 in several strict runs; Lipo random full increases path-bias magnitude, but the accuracy gain is marginal.

## Descriptor-Only Baselines

| Dataset | Split | best descriptor-only | RMSE mean ± std | best graph+descriptor |
|---|---|---:|---:|---:|
| ESOL | random | XGBoost | 0.7997 ± 0.0431 | `gine_desc` 0.5594 ± 0.0463 |
| ESOL | scaffold | XGBoost | 1.2088 ± 0.0040 | `gine_desc` 0.9295 ± 0.0241 |
| Lipo | random | RF | 0.9431 ± 0.0193 | `gine_desc_film_graph` 0.6000 ± 0.0235 |
| Lipo | scaffold | RF | 1.0195 ± 0.0016 | `gine_desc_residual_add` 0.6948 ± 0.0136 |
| FreeSolv | random | XGBoost | 1.6057 ± 0.3428 | `gine_desc` 1.2605 ± 0.2839 |
| FreeSolv | scaffold | RF | 2.6849 ± 0.0148 | `gine_desc` 2.3249 ± 0.1168 |

## Conclusion

The safest project claim is:

> Descriptor-enhanced GINE consistently improves molecular regression over graph-only and descriptor-only baselines across ESOL, Lipophilicity, and FreeSolv. ChemPath attention is implemented and tested with stricter controls, but the current evidence does not support path bias as a robust causal performance source.

This supports a report/paper framing around graph-descriptor fusion, with ChemPath as an exploratory path-bias module and negative-control analysis.
