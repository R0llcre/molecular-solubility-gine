from .clean import clean_molecules
from .split import make_splits

__all__ = ["MoleculeDataset", "collate_molecules", "make_splits", "clean_molecules"]


def __getattr__(name: str):
    if name in {"MoleculeDataset", "collate_molecules"}:
        from .dataset import MoleculeDataset, collate_molecules

        return {"MoleculeDataset": MoleculeDataset, "collate_molecules": collate_molecules}[name]
    raise AttributeError(name)
