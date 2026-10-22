from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from molfuse_gine.baselines.tabular import _make_raw_features, parse_feature_spec
from molfuse_gine.features.fragment_bank import build_brics_vocab


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build an inspection/cache feature matrix for a CSV. Training scripts still build "
            "descriptor selection, BRICS vocabularies, imputers, and scalers from train only."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--feature-set", default="desc16_morgan2_maccs")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input)
    spec = parse_feature_spec(args.feature_set)
    vocab = []
    if spec.brics_fragments > 0:
        vocab = build_brics_vocab(df[args.smiles_col], max_fragments=spec.brics_fragments)
    features = _make_raw_features(df, args.smiles_col, spec, vocab=vocab)
    out = pd.concat([df[[args.smiles_col]].reset_index(drop=True), features], axis=1)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} rows={len(out)} cols={out.shape[1]}")


if __name__ == "__main__":
    main()
