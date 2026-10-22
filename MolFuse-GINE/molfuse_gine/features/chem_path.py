from __future__ import annotations

from collections import deque

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem

from .atom_bond import (
    BOND_TYPE_TO_IDX,
    functional_group_atoms,
    is_hbond_acceptor,
    is_hbond_donor,
)


CHEM_PATH_FIELDS = [
    "distance",
    "num_shortest_paths",
    "single_count",
    "double_count",
    "triple_count",
    "aromatic_count",
    "conjugated_count",
    "ring_bond_count",
    "hetero_atom_count",
    "aromatic_atom_count",
    "donor_acceptor_touch",
    "same_ring",
    "fg_touch",
    "disconnected",
    "single_count_max",
    "double_count_max",
    "triple_count_max",
    "aromatic_count_max",
    "conjugated_count_max",
    "ring_bond_count_max",
    "hetero_atom_count_max",
    "aromatic_atom_count_max",
]
CHEM_PATH_DIM = len(CHEM_PATH_FIELDS)


def _adjacency(mol: Chem.Mol):
    adj: list[list[tuple[int, rdchem.Bond]]] = [[] for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i].append((j, bond))
        adj[j].append((i, bond))
    return adj


def _same_ring_matrix(mol: Chem.Mol) -> np.ndarray:
    n = mol.GetNumAtoms()
    same = np.zeros((n, n), dtype=np.float32)
    for ring in mol.GetRingInfo().AtomRings():
        for i in ring:
            for j in ring:
                same[i, j] = 1.0
    return same


def _shortest_predecessors(adj, source: int):
    n = len(adj)
    predecessors: list[list[tuple[int, rdchem.Bond]]] = [[] for _ in range(n)]
    dist = [-1] * n
    dist[source] = 0
    queue = deque([source])
    while queue:
        cur = queue.popleft()
        for nxt, bond in adj[cur]:
            cand = dist[cur] + 1
            if dist[nxt] < 0:
                dist[nxt] = cand
                predecessors[nxt].append((cur, bond))
                queue.append(nxt)
            elif dist[nxt] == cand:
                predecessors[nxt].append((cur, bond))
    return dist, predecessors


def _enumerate_shortest_paths(
    source: int,
    target: int,
    predecessors: list[list[tuple[int, rdchem.Bond]]],
    max_paths: int,
):
    if source == target:
        return [([source], [])]

    paths = []

    def visit(cur: int, nodes_rev: list[int], bonds_rev: list[rdchem.Bond]):
        if len(paths) >= max_paths:
            return
        if cur == source:
            paths.append((list(reversed(nodes_rev)), list(reversed(bonds_rev))))
            return
        for prev, bond in predecessors[cur]:
            visit(prev, nodes_rev + [prev], bonds_rev + [bond])

    visit(target, [target], [])
    return paths


def _path_stats(mol, nodes, bonds, fg_atoms, donor_acceptor):
    counts = np.zeros((8,), dtype=np.float32)
    flags = np.zeros((2,), dtype=np.float32)
    for node_idx in nodes:
        atom = mol.GetAtomWithIdx(node_idx)
        if atom.GetAtomicNum() not in (1, 6):
            counts[6] += 1.0
        if atom.GetIsAromatic():
            counts[7] += 1.0
        if node_idx in donor_acceptor:
            flags[0] = 1.0
        if node_idx in fg_atoms:
            flags[1] = 1.0

    for bond in bonds:
        bond_type = BOND_TYPE_TO_IDX.get(bond.GetBondType(), 4)
        if bond_type <= 3:
            counts[bond_type] += 1.0
        if bond.GetIsConjugated():
            counts[4] += 1.0
        if bond.IsInRing():
            counts[5] += 1.0
    return counts, flags


def compute_chem_path_features(
    mol: Chem.Mol,
    max_distance: int = 12,
    max_shortest_paths: int = 64,
) -> np.ndarray:
    """Return [N, N, CHEM_PATH_DIM] aggregated chemically typed shortest-path features."""
    n = mol.GetNumAtoms()
    path = np.zeros((n, n, CHEM_PATH_DIM), dtype=np.float32)
    if n == 0:
        return path

    adj = _adjacency(mol)
    same_ring = _same_ring_matrix(mol)
    fg_atoms = functional_group_atoms(mol)
    donor_acceptor = {
        a.GetIdx()
        for a in mol.GetAtoms()
        if is_hbond_donor(a) or is_hbond_acceptor(a)
    }

    for source in range(n):
        dist, predecessors = _shortest_predecessors(adj, source)
        for target in range(n):
            d = dist[target]
            if d < 0:
                path[source, target, 0] = max_distance + 1
                path[source, target, CHEM_PATH_FIELDS.index("disconnected")] = 1.0
                continue

            shortest_paths = _enumerate_shortest_paths(
                source, target, predecessors, max_paths=max_shortest_paths
            )
            clipped_d = min(d, max_distance)
            path[source, target, 0] = clipped_d
            path[source, target, CHEM_PATH_FIELDS.index("num_shortest_paths")] = np.log1p(
                len(shortest_paths)
            )
            path[source, target, CHEM_PATH_FIELDS.index("same_ring")] = same_ring[source, target]

            if not shortest_paths:
                continue

            count_rows = []
            flag_rows = []
            for nodes, bonds in shortest_paths:
                counts, flags = _path_stats(mol, nodes, bonds, fg_atoms, donor_acceptor)
                count_rows.append(counts)
                flag_rows.append(flags)

            count_mat = np.stack(count_rows, axis=0)
            flag_mat = np.stack(flag_rows, axis=0)
            mean_counts = count_mat.mean(axis=0)
            max_counts = count_mat.max(axis=0)
            any_flags = flag_mat.max(axis=0)

            for offset, value in enumerate(mean_counts):
                path[source, target, 2 + offset] = value
            path[source, target, CHEM_PATH_FIELDS.index("donor_acceptor_touch")] = any_flags[0]
            path[source, target, CHEM_PATH_FIELDS.index("fg_touch")] = any_flags[1]

            max_start = CHEM_PATH_FIELDS.index("single_count_max")
            for offset, value in enumerate(max_counts):
                path[source, target, max_start + offset] = value

    return path
