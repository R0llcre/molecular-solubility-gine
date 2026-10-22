from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from molfuse_gine.data import MoleculeDataset, clean_molecules, collate_molecules
from molfuse_gine.data.dataset import FEATURE_VERSION
from molfuse_gine.data.scaler import load_scaler, save_scaler
from molfuse_gine.models import ChemPathGINE
from molfuse_gine.training.train import _ns, select_device


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_model_from_checkpoint(cfg, dataset: MoleculeDataset) -> ChemPathGINE:
    return ChemPathGINE(
        hidden_dim=int(cfg.model.hidden_dim),
        num_gnn_layers=int(cfg.model.num_gnn_layers),
        num_attention_layers=int(cfg.model.num_attention_layers),
        heads=int(cfg.model.heads),
        dropout=float(cfg.model.dropout),
        path_dim=dataset.path_dim,
        descriptor_dim=dataset.descriptor_dim,
        max_path_distance=int(cfg.model.max_path_distance),
        variant=cfg.model.variant,
        use_descriptor_fusion=bool(getattr(cfg.model, "use_descriptor_fusion", False)),
        use_descriptor_path_gate=bool(getattr(cfg.model, "use_descriptor_path_gate", False)),
    )


def _scaler_key_from_path(y_scaler_path: Path) -> str:
    name = y_scaler_path.name
    if name.endswith(".y.npz"):
        return name[: -len(".y.npz")]
    return y_scaler_path.stem


def _checkpoint_scaler_paths(ckpt: dict, work_dir: Path) -> tuple[Path, Path, str]:
    if "y_scaler" in ckpt and "desc_scaler" in ckpt:
        scaler_dir = work_dir / "checkpoint_scalers"
        scaler_key = "checkpoint"
        desc_scaler_path = scaler_dir / f"{scaler_key}.desc.npz"
        y_scaler_path = scaler_dir / f"{scaler_key}.y.npz"
        save_scaler(desc_scaler_path, ckpt["desc_scaler"])
        save_scaler(y_scaler_path, ckpt["y_scaler"])
        return desc_scaler_path, y_scaler_path, scaler_key

    y_scaler_path = Path(ckpt["y_scaler_path"]).resolve()
    desc_scaler_path = Path(ckpt["desc_scaler_path"]).resolve()
    scaler_key = _scaler_key_from_path(y_scaler_path)
    if desc_scaler_path.parent != y_scaler_path.parent:
        raise ValueError("Descriptor and target scalers must live in the same directory")
    return desc_scaler_path, y_scaler_path, scaler_key


@torch.no_grad()
def run_prediction(
    checkpoint: str | Path,
    csv_path: str | Path,
    output: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
):
    checkpoint = Path(checkpoint)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output.parent / f"{output.stem}_predict_cache"
    work_dir.mkdir(parents=True, exist_ok=True)

    ckpt = _load_checkpoint(checkpoint)
    if ckpt.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"Checkpoint feature_version={ckpt.get('feature_version')} does not match "
            f"current FEATURE_VERSION={FEATURE_VERSION}. Re-train or use matching code."
        )

    raw_cfg = ckpt["config"]
    cfg = _ns(raw_cfg)
    desc_scaler_path, y_scaler_path, scaler_key = _checkpoint_scaler_paths(ckpt, work_dir)

    clean_path = clean_molecules(csv_path, work_dir / "clean.csv", cfg.data.smiles_col)
    dataset = MoleculeDataset(
        clean_path,
        mode="predict",
        smiles_col=cfg.data.smiles_col,
        target_col=None,
        descriptor_cols=ckpt.get("descriptor_cols", cfg.data.descriptor_cols),
        scaler_dir=y_scaler_path.parent,
        scaler_key=scaler_key,
        cache_dir=work_dir / "cache",
        max_path_distance=int(cfg.model.max_path_distance),
        skip_path_features=bool(
            ckpt.get(
                "skip_path_features",
                not ChemPathGINE.variant_uses_real_path(cfg.model.variant),
            )
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_molecules,
    )

    device = select_device(getattr(cfg.run, "device", "auto"))
    print(f"device={device}")
    model = _build_model_from_checkpoint(cfg, dataset).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pred_scaled, row_idx, smiles = [], [], []
    for batch in loader:
        batch = batch.to(device)
        pred_scaled.append(model(batch).detach().cpu().numpy())
        row_idx.extend(batch.row_idx.view(-1).detach().cpu().tolist())
        smiles.extend(batch.smiles)

    if not pred_scaled:
        raise ValueError("No valid molecules to predict")

    pred_scaled = np.concatenate(pred_scaled).reshape(-1, 1)
    y_scaler = load_scaler(y_scaler_path)
    pred = (pred_scaled * y_scaler["scale"] + y_scaler["mean"]).reshape(-1)

    rows = []
    for i, smi, p in zip(row_idx, smiles, pred):
        src = dataset.df.iloc[int(i)].to_dict()
        src[cfg.data.smiles_col] = smi
        src["y_pred"] = float(p)
        rows.append(src)
    pd.DataFrame(rows).to_csv(output, index=False)

    meta = {
        "checkpoint": str(checkpoint),
        "input_csv": str(csv_path),
        "output_csv": str(output),
        "num_predictions": len(rows),
    }
    with output.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args(argv)
    meta = run_prediction(
        checkpoint=args.checkpoint,
        csv_path=args.csv,
        output=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    main()
