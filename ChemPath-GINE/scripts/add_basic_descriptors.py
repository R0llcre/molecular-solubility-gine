from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def _minimum_degree(mol: Chem.Mol) -> int:
    if mol.GetNumAtoms() == 0:
        return 0
    return min(atom.GetDegree() for atom in mol.GetAtoms())


def descriptor_row(smiles: str) -> dict[str, float] | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return {
        "MinimumDegree": float(_minimum_degree(mol)),
        "MolecularWeight": float(Descriptors.MolWt(mol)),
        "MolLogP": float(Descriptors.MolLogP(mol)),
        "MolMR": float(Descriptors.MolMR(mol)),
        "HeavyAtomCount": float(Descriptors.HeavyAtomCount(mol)),
        "HBondDonors": float(rdMolDescriptors.CalcNumHBD(mol)),
        "HBondAcceptors": float(rdMolDescriptors.CalcNumHBA(mol)),
        "RingCount": float(rdMolDescriptors.CalcNumRings(mol)),
        "RotatableBonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "PolarSurfaceArea": float(rdMolDescriptors.CalcTPSA(mol)),
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "AromaticRings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "AliphaticRings": float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        "LabuteASA": float(rdMolDescriptors.CalcLabuteASA(mol)),
        "BalabanJ": float(Descriptors.BalabanJ(mol)),
        "BertzCT": float(Descriptors.BertzCT(mol)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smiles-col", default="SMILES")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input)
    rows = []
    for _, row in df.iterrows():
        desc = descriptor_row(row[args.smiles_col])
        if desc is None:
            continue
        out = row.to_dict()
        out.update(desc)
        rows.append(out)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
