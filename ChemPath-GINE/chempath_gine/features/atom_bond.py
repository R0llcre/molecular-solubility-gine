from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdchem

from chempath_gine.features.motifs import MOTIF_SMARTS


HYBRIDIZATION_TO_IDX = {
    rdchem.HybridizationType.UNSPECIFIED: 0,
    rdchem.HybridizationType.S: 1,
    rdchem.HybridizationType.SP: 2,
    rdchem.HybridizationType.SP2: 3,
    rdchem.HybridizationType.SP3: 4,
    rdchem.HybridizationType.SP3D: 5,
    rdchem.HybridizationType.SP3D2: 6,
}

BOND_TYPE_TO_IDX = {
    rdchem.BondType.SINGLE: 0,
    rdchem.BondType.DOUBLE: 1,
    rdchem.BondType.TRIPLE: 2,
    rdchem.BondType.AROMATIC: 3,
}
SELF_LOOP_BOND_TYPE_IDX = 4

BOND_STEREO_TO_IDX = {
    rdchem.BondStereo.STEREONONE: 0,
    rdchem.BondStereo.STEREOANY: 1,
    rdchem.BondStereo.STEREOZ: 2,
    rdchem.BondStereo.STEREOE: 3,
    rdchem.BondStereo.STEREOCIS: 4,
    rdchem.BondStereo.STEREOTRANS: 5,
}

def ring_atoms(mol: Chem.Mol) -> set[int]:
    return {idx for ring in mol.GetRingInfo().AtomRings() for idx in ring}


def functional_group_atoms(mol: Chem.Mol) -> set[int]:
    atoms: set[int] = set()
    for smarts in MOTIF_SMARTS.values():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt):
            atoms.update(match)
    return atoms


def is_hbond_donor(atom: rdchem.Atom) -> bool:
    return atom.GetAtomicNum() in (7, 8) and atom.GetTotalNumHs(includeNeighbors=True) > 0


def is_hbond_acceptor(atom: rdchem.Atom) -> bool:
    if atom.GetFormalCharge() > 0:
        return False
    atomic_num = atom.GetAtomicNum()
    if atomic_num in (8, 16):
        return True
    if atomic_num != 7:
        return False
    if atom.GetIsAromatic():
        return atom.GetTotalNumHs(includeNeighbors=True) == 0
    for bond in atom.GetBonds():
        nbr = bond.GetOtherAtom(atom)
        if nbr.GetAtomicNum() == 6:
            for nbr_bond in nbr.GetBonds():
                other = nbr_bond.GetOtherAtom(nbr)
                if other.GetAtomicNum() == 8 and nbr_bond.GetBondType() == rdchem.BondType.DOUBLE:
                    return False
    return atom.GetTotalNumHs(includeNeighbors=True) == 0


def _atom_cat(atom: rdchem.Atom) -> list[int]:
    formal_charge = max(-5, min(5, atom.GetFormalCharge())) + 5
    return [
        max(0, min(118, atom.GetAtomicNum())),
        max(0, min(10, atom.GetDegree())),
        formal_charge,
        int(atom.GetChiralTag()),
        HYBRIDIZATION_TO_IDX.get(atom.GetHybridization(), 0),
        max(0, min(8, atom.GetTotalNumHs(includeNeighbors=True))),
        max(0, min(8, atom.GetTotalValence())),
    ]


def _atom_float(atom: rdchem.Atom, ring: set[int], fg: set[int]) -> list[float]:
    idx = atom.GetIdx()
    return [
        float(atom.GetIsAromatic()),
        float(idx in ring),
        float(idx in fg),
        float(is_hbond_donor(atom)),
        float(is_hbond_acceptor(atom)),
    ]


def _bond_cat(bond: rdchem.Bond) -> list[int]:
    return [
        BOND_TYPE_TO_IDX.get(bond.GetBondType(), 4),
        BOND_STEREO_TO_IDX.get(bond.GetStereo(), 0),
    ]


def _bond_float(bond: rdchem.Bond) -> list[float]:
    return [
        float(bond.GetIsConjugated()),
        float(bond.GetIsAromatic()),
        float(bond.IsInRing()),
    ]


def featurize_mol(mol: Chem.Mol):
    import torch

    ring = ring_atoms(mol)
    fg = functional_group_atoms(mol)

    atom_cat = torch.tensor([_atom_cat(a) for a in mol.GetAtoms()], dtype=torch.long)
    atom_float = torch.tensor([_atom_float(a, ring, fg) for a in mol.GetAtoms()], dtype=torch.float32)

    edges: list[list[int]] = []
    edge_cat: list[list[int]] = []
    edge_float: list[list[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        cat = _bond_cat(bond)
        flt = _bond_float(bond)
        edges.extend([[i, j], [j, i]])
        edge_cat.extend([cat, cat])
        edge_float.extend([flt, flt])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_cat_t = torch.tensor(edge_cat, dtype=torch.long)
        edge_float_t = torch.tensor(edge_float, dtype=torch.float32)
    else:
        # Use a dedicated self-loop sentinel for isolated atoms. This keeps
        # GINE/MPS away from empty edge tensors without pretending there is a
        # real single bond.
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_cat_t = torch.tensor([[SELF_LOOP_BOND_TYPE_IDX, 0]], dtype=torch.long)
        edge_float_t = torch.zeros((1, 3), dtype=torch.float32)

    return {
        "atom_cat": atom_cat,
        "atom_float": atom_float,
        "edge_index": edge_index,
        "edge_cat": edge_cat_t,
        "edge_float": edge_float_t,
        "ring_atoms": ring,
        "fg_atoms": fg,
    }
