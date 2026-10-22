from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


DESC6_NAMES = [
    "MinimumDegree",
    "MolecularWeight",
    "HBondDonors",
    "RingCount",
    "RotatableBonds",
    "PolarSurfaceArea",
]

DESC16_NAMES = [
    "MinimumDegree",
    "MolecularWeight",
    "MolLogP",
    "MolMR",
    "HeavyAtomCount",
    "HBondDonors",
    "HBondAcceptors",
    "RingCount",
    "RotatableBonds",
    "PolarSurfaceArea",
    "FractionCSP3",
    "AromaticRings",
    "AliphaticRings",
    "LabuteASA",
    "BalabanJ",
    "BertzCT",
]


def minimum_degree(mol: Chem.Mol) -> float:
    if mol.GetNumAtoms() == 0:
        return 0.0
    return float(min(atom.GetDegree() for atom in mol.GetAtoms()))


def hetero_atom_count(mol: Chem.Mol) -> float:
    return float(sum(atom.GetAtomicNum() not in (1, 6) for atom in mol.GetAtoms()))


def halogen_count(mol: Chem.Mol) -> float:
    return float(sum(atom.GetAtomicNum() in (9, 17, 35, 53) for atom in mol.GetAtoms()))


def formal_charge_abs_sum(mol: Chem.Mol) -> float:
    return float(sum(abs(atom.GetFormalCharge()) for atom in mol.GetAtoms()))


DESCRIPTOR_FUNCS: dict[str, Callable[[Chem.Mol], float]] = {
    "MinimumDegree": minimum_degree,
    "MolecularWeight": Descriptors.MolWt,
    "MolLogP": Crippen.MolLogP,
    "MolMR": Crippen.MolMR,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "HBondDonors": rdMolDescriptors.CalcNumHBD,
    "HBondAcceptors": rdMolDescriptors.CalcNumHBA,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds,
    "PolarSurfaceArea": rdMolDescriptors.CalcTPSA,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "AromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "AliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
    "SaturatedRings": rdMolDescriptors.CalcNumSaturatedRings,
    "AromaticHeterocycles": rdMolDescriptors.CalcNumAromaticHeterocycles,
    "AromaticCarbocycles": rdMolDescriptors.CalcNumAromaticCarbocycles,
    "AliphaticHeterocycles": rdMolDescriptors.CalcNumAliphaticHeterocycles,
    "AliphaticCarbocycles": rdMolDescriptors.CalcNumAliphaticCarbocycles,
    "SaturatedHeterocycles": rdMolDescriptors.CalcNumSaturatedHeterocycles,
    "SaturatedCarbocycles": rdMolDescriptors.CalcNumSaturatedCarbocycles,
    "LabuteASA": rdMolDescriptors.CalcLabuteASA,
    "BalabanJ": Descriptors.BalabanJ,
    "BertzCT": Descriptors.BertzCT,
    "Chi0": Descriptors.Chi0,
    "Chi0n": Descriptors.Chi0n,
    "Chi0v": Descriptors.Chi0v,
    "Chi1": Descriptors.Chi1,
    "Chi1n": Descriptors.Chi1n,
    "Chi1v": Descriptors.Chi1v,
    "Chi2n": Descriptors.Chi2n,
    "Chi2v": Descriptors.Chi2v,
    "Chi3n": Descriptors.Chi3n,
    "Chi3v": Descriptors.Chi3v,
    "Chi4n": Descriptors.Chi4n,
    "Chi4v": Descriptors.Chi4v,
    "Kappa1": Descriptors.Kappa1,
    "Kappa2": Descriptors.Kappa2,
    "Kappa3": Descriptors.Kappa3,
    "HallKierAlpha": Descriptors.HallKierAlpha,
    "Ipc": Descriptors.Ipc,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "NumRadicalElectrons": Descriptors.NumRadicalElectrons,
    "NHOHCount": Lipinski.NHOHCount,
    "NOCount": Lipinski.NOCount,
    "MolFractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "HeteroAtomCount": hetero_atom_count,
    "HalogenCount": halogen_count,
    "FormalChargeAbsSum": formal_charge_abs_sum,
    "MaxAbsEStateIndex": Descriptors.MaxAbsEStateIndex,
    "MaxEStateIndex": Descriptors.MaxEStateIndex,
    "MinAbsEStateIndex": Descriptors.MinAbsEStateIndex,
    "MinEStateIndex": Descriptors.MinEStateIndex,
}

RDKIT2D_NAMES = list(DESCRIPTOR_FUNCS)


def descriptor_names(name: str | Iterable[str]) -> list[str]:
    if isinstance(name, str):
        key = name.lower()
        if key == "desc6":
            return list(DESC6_NAMES)
        if key == "desc16":
            return list(DESC16_NAMES)
        if key in {"rdkit2d", "rdkit2d_full", "desc50"}:
            return list(RDKIT2D_NAMES)
        if not key:
            return []
        return [item.strip() for item in name.split(",") if item.strip()]
    return list(name)


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    if smiles is None:
        return None
    return Chem.MolFromSmiles(str(smiles))


def compute_descriptor_row(mol: Chem.Mol, names: Iterable[str]) -> dict[str, float]:
    row: dict[str, float] = {}
    for name in names:
        func = DESCRIPTOR_FUNCS.get(name)
        if func is None:
            raise KeyError(f"Unknown descriptor {name!r}")
        try:
            value = float(func(mol))
        except Exception:
            value = np.nan
        row[name] = value
    return row


def compute_descriptor_frame(
    smiles: Iterable[str],
    names: str | Iterable[str] = "desc16",
) -> pd.DataFrame:
    cols = descriptor_names(names)
    rows = []
    for smi in smiles:
        mol = mol_from_smiles(str(smi))
        if mol is None:
            rows.append({name: np.nan for name in cols})
        else:
            rows.append(compute_descriptor_row(mol, cols))
    return pd.DataFrame(rows, columns=cols)
