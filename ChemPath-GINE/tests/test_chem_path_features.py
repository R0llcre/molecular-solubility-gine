from rdkit import Chem

from chempath_gine.features.atom_bond import functional_group_atoms, is_hbond_acceptor
from chempath_gine.features.chem_path import CHEM_PATH_FIELDS, compute_chem_path_features


def idx(name: str) -> int:
    return CHEM_PATH_FIELDS.index(name)


def test_chem_path_distinguishes_bond_and_hetero_path():
    mol = Chem.MolFromSmiles("CCO")
    path = compute_chem_path_features(mol, max_distance=8)

    assert path.shape == (3, 3, len(CHEM_PATH_FIELDS))
    assert path[0, 2, idx("distance")] == 2
    assert path[0, 2, idx("single_count")] == 2
    assert path[0, 2, idx("hetero_atom_count")] == 1
    assert path[0, 2, idx("donor_acceptor_touch")] == 1
    assert (path == path.transpose(1, 0, 2)).all()


def test_chem_path_marks_aromatic_ring_path():
    mol = Chem.MolFromSmiles("c1ccccc1")
    path = compute_chem_path_features(mol, max_distance=8)

    assert path[0, 3, idx("same_ring")] == 1
    assert path[0, 3, idx("aromatic_count")] > 0
    assert path[0, 3, idx("ring_bond_count")] > 0
    assert path[0, 3, idx("num_shortest_paths")] > path[0, 1, idx("num_shortest_paths")]


def test_chem_path_marks_disconnected_pairs():
    mol = Chem.MolFromSmiles("CC.O")
    path = compute_chem_path_features(mol, max_distance=8)

    assert path[0, 2, idx("distance")] == 9
    assert path[0, 2, idx("disconnected")] == 1


def test_motif_library_covers_halogen_and_amide_acceptor_is_conservative():
    mol = Chem.MolFromSmiles("CCCl")
    assert functional_group_atoms(mol)

    amide = Chem.MolFromSmiles("CC(=O)N")
    nitrogen = amide.GetAtomWithIdx(3)
    assert not is_hbond_acceptor(nitrogen)
