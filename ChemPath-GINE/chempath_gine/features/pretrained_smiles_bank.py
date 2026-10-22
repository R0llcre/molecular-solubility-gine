from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem


@dataclass(frozen=True)
class SmilesEmbeddingConfig:
    model_name: str = "DeepChem/ChemBERTa-77M-MLM"
    batch_size: int = 32
    max_length: int = 256
    device: str = "auto"
    n_random: int = 0
    seed: int = 0


def smiles_views(smiles: str, n_random: int = 0, seed: int = 0) -> list[str]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return []
    views = [Chem.MolToSmiles(mol, canonical=True)]
    if n_random <= 0:
        return views
    rng = np.random.default_rng(seed)
    for _ in range(int(n_random)):
        # RDKit random SMILES depends on its internal RNG; atom renumbering gives
        # deterministic per-call diversity controlled by this local RNG.
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        renum = Chem.RenumberAtoms(mol, [int(i) for i in order])
        views.append(Chem.MolToSmiles(renum, canonical=False, doRandom=True))
    return views


def _select_device(device: str):
    import torch

    key = (device or "auto").lower()
    if key == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(key)


def _mean_pool(last_hidden, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (last_hidden * mask).sum(dim=1) / denom


def embed_smiles_frame(
    smiles: Iterable[str],
    config: SmilesEmbeddingConfig,
) -> pd.DataFrame:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install torch and transformers to build pretrained SMILES embeddings") from exc

    smiles_list = list(smiles)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModel.from_pretrained(config.model_name)
    device = _select_device(config.device)
    model.to(device)
    model.eval()

    rows: list[np.ndarray] = []
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    for mol_idx, smi in enumerate(smiles_list):
        views = smiles_views(smi, n_random=config.n_random, seed=config.seed + mol_idx)
        if not views:
            rows.append(np.zeros((hidden_size * 2 if config.n_random > 0 else hidden_size,), dtype=np.float32))
            continue
        view_embs = []
        for start in range(0, len(views), config.batch_size):
            batch = views[start : start + config.batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                out = model(**encoded)
                pooled = _mean_pool(out.last_hidden_state, encoded["attention_mask"])
            view_embs.append(pooled.detach().cpu().numpy().astype(np.float32))
        emb = np.concatenate(view_embs, axis=0)
        mean = emb.mean(axis=0)
        if config.n_random > 0:
            std = emb.std(axis=0)
            rows.append(np.concatenate([mean, std]).astype(np.float32))
        else:
            rows.append(mean.astype(np.float32))

    arr = np.vstack(rows)
    columns = [f"smiles_emb_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=columns)
