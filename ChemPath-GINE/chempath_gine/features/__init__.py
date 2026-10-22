from .descriptor_bank import DESC16_NAMES, DESC6_NAMES, RDKIT2D_NAMES, compute_descriptor_frame

__all__ = [
    "featurize_mol",
    "compute_chem_path_features",
    "CHEM_PATH_DIM",
    "DESC6_NAMES",
    "DESC16_NAMES",
    "RDKIT2D_NAMES",
    "compute_descriptor_frame",
]


def __getattr__(name: str):
    if name == "featurize_mol":
        from .atom_bond import featurize_mol

        return featurize_mol
    if name in {"CHEM_PATH_DIM", "compute_chem_path_features"}:
        from .chem_path import CHEM_PATH_DIM, compute_chem_path_features

        return {"CHEM_PATH_DIM": CHEM_PATH_DIM, "compute_chem_path_features": compute_chem_path_features}[
            name
        ]
    raise AttributeError(name)
