from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from chempath_gine.data import MoleculeDataset, collate_molecules, make_splits
from chempath_gine.data.dataset import FEATURE_VERSION
from chempath_gine.data.scaler import load_scaler
from chempath_gine.models import ChemPathGINE
from chempath_gine.training.metrics import regression_metrics


def _ns(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_ns(v) for v in value]
    return value


def load_config(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw, _ns(raw)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(preferred: str = "auto") -> torch.device:
    preferred = (preferred or "auto").lower()
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preferred == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is False")
    if preferred == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Requested MPS, but torch.backends.mps.is_available() is False")
    return torch.device(preferred)


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_molecules,
    )


def inverse_y(y_scaled: np.ndarray, scaler_path: Path) -> np.ndarray:
    scaler = load_scaler(scaler_path)
    return y_scaled.reshape(-1, 1) * scaler["scale"] + scaler["mean"]


def load_checkpoint(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    accum_steps: int,
    max_grad_norm: float,
    distill_weight: float = 0.0,
) -> float:
    model.train()
    accum_steps = max(1, int(accum_steps))
    total = 0.0
    steps = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), 1):
        batch = batch.to(device)
        pred = model(batch)
        loss = criterion(pred, batch.y.view(-1))
        if distill_weight > 0 and hasattr(batch, "teacher"):
            loss = loss + float(distill_weight) * criterion(pred, batch.teacher.view(-1))
        if torch.isnan(loss) or torch.isinf(loss):
            raise FloatingPointError("NaN/Inf loss during training")
        (loss / accum_steps).backward()
        if step % accum_steps == 0:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total += float(loss.item())
        steps += 1
    if steps > 0 and steps % accum_steps != 0:
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return total / max(1, steps)


@torch.no_grad()
def evaluate(model, loader, criterion, device, y_scaler_path: Path):
    model.eval()
    losses = []
    pred_scaled, true_scaled, row_idx, smiles = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(-1)
        losses.append(float(criterion(pred, y).item()))
        pred_scaled.append(pred.detach().cpu().numpy())
        true_scaled.append(y.detach().cpu().numpy())
        row_idx.extend(batch.row_idx.view(-1).detach().cpu().tolist())
        smiles.extend(batch.smiles)

    if not pred_scaled:
        raise ValueError("Empty evaluation loader after molecule filtering")

    pred_scaled = np.concatenate(pred_scaled)
    true_scaled = np.concatenate(true_scaled)
    pred = inverse_y(pred_scaled, y_scaler_path).reshape(-1)
    true = inverse_y(true_scaled, y_scaler_path).reshape(-1)
    metrics = regression_metrics(pred, true)
    return float(np.mean(losses)), metrics, pred, true, row_idx, smiles


def build_datasets(cfg, run_dir: Path):
    if all(hasattr(cfg.data, name) for name in ("train_csv", "val_csv", "test_csv")):
        split_paths = {
            "train": Path(cfg.data.train_csv),
            "val": Path(cfg.data.val_csv),
            "test": Path(cfg.data.test_csv),
        }
    else:
        split_paths = make_splits(
            csv_path=cfg.data.csv,
            out_dir=run_dir / "splits",
            smiles_col=cfg.data.smiles_col,
            split=cfg.data.split,
            train_ratio=float(cfg.data.train_ratio),
            val_ratio=float(cfg.data.val_ratio),
            seed=int(cfg.run.seed),
        )
    scaler_dir = run_dir / "scalers"
    cache_dir = Path(cfg.data.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = run_dir / cache_dir
    skip_path_features = getattr(cfg.data, "skip_path_features", None)
    if skip_path_features is None:
        skip_path_features = not ChemPathGINE.variant_uses_real_path(cfg.model.variant)
    common = dict(
        smiles_col=cfg.data.smiles_col,
        target_col=cfg.data.target_col,
        descriptor_cols=cfg.data.descriptor_cols,
        teacher_col=getattr(cfg.data, "teacher_col", None),
        scaler_dir=scaler_dir,
        scaler_key="split",
        cache_dir=cache_dir,
        max_path_distance=int(cfg.model.max_path_distance),
        skip_path_features=bool(skip_path_features),
    )
    train_ds = MoleculeDataset(split_paths["train"], mode="train", **common)
    val_ds = MoleculeDataset(split_paths["val"], mode="val", **common)
    test_ds = MoleculeDataset(split_paths["test"], mode="test", **common)
    return train_ds, val_ds, test_ds


def build_model(cfg, train_ds: MoleculeDataset) -> ChemPathGINE:
    return ChemPathGINE(
        hidden_dim=int(cfg.model.hidden_dim),
        num_gnn_layers=int(cfg.model.num_gnn_layers),
        num_attention_layers=int(cfg.model.num_attention_layers),
        heads=int(cfg.model.heads),
        dropout=float(cfg.model.dropout),
        path_dim=train_ds.path_dim,
        descriptor_dim=train_ds.descriptor_dim,
        max_path_distance=int(cfg.model.max_path_distance),
        variant=cfg.model.variant,
        use_descriptor_fusion=bool(getattr(cfg.model, "use_descriptor_fusion", False)),
        use_descriptor_path_gate=bool(getattr(cfg.model, "use_descriptor_path_gate", False)),
    )


@torch.no_grad()
def model_diagnostics(model, loader, device) -> dict:
    if not hasattr(model, "diagnostics"):
        return {}
    try:
        batch = next(iter(loader)).to(device)
    except StopIteration:
        return {}
    was_training = model.training
    model.eval()
    diag = model.diagnostics(batch)
    if was_training:
        model.train()
    return diag


def save_predictions(path: Path, dataset: MoleculeDataset, pred, true, row_idx, smiles):
    rows = []
    for i, p, t, smi in zip(row_idx, pred, true, smiles):
        src = dataset.df.iloc[int(i)].to_dict()
        src[dataset.smiles_col] = smi
        src["y_true"] = float(t)
        src["y_pred"] = float(p)
        src["absolute_error"] = abs(float(t) - float(p))
        rows.append(src)
    pd.DataFrame(rows).to_csv(path, index=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    raw_cfg, cfg = load_config(args.config)
    set_seed(int(cfg.run.seed))
    device = select_device(getattr(cfg.run, "device", "auto"))
    print(f"device={device}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.run.output_dir) / f"{cfg.run.name}_{cfg.model.variant}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw_cfg, f, sort_keys=False)

    train_ds, val_ds, test_ds = build_datasets(cfg, run_dir)
    loaders = {
        "train": make_loader(train_ds, cfg.train.batch_size, True, cfg.train.num_workers),
        "val": make_loader(val_ds, cfg.train.batch_size, False, cfg.train.num_workers),
        "test": make_loader(test_ds, cfg.train.batch_size, False, cfg.train.num_workers),
    }

    model = build_model(cfg, train_ds).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, int(cfg.train.patience) // 3)
    )
    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    stale = 0
    best_path = run_dir / "best.pt"
    history = []
    for epoch in range(1, int(cfg.train.epochs) + 1):
        try:
            train_loss = train_epoch(
                model,
                loaders["train"],
                optimizer,
                criterion,
                device,
                int(cfg.train.accum_steps),
                float(getattr(cfg.train, "max_grad_norm", 5.0)),
                float(getattr(cfg.train, "distill_weight", 0.0)),
            )
        except FloatingPointError as exc:
            if not best_path.exists():
                raise
            history.append({"epoch": epoch, "stopped": "floating_point_error", "error": str(exc)})
            print(f"stopping at epoch={epoch} due to {exc}; evaluating best checkpoint")
            break
        val_loss, val_metrics, *_ = evaluate(
            model, loaders["val"], criterion, device, train_ds.y_scaler_path
        )
        scheduler.step(val_loss)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics}
        history.append(record)
        print(
            f"epoch={epoch:03d} train={train_loss:.4f} val={val_loss:.4f} "
            f"rmse={val_metrics['rmse']:.4f} mae={val_metrics['mae']:.4f} r2={val_metrics['r2']:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            diagnostics = model_diagnostics(model, loaders["val"], device)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": copy.deepcopy(raw_cfg),
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "val_metrics": copy.deepcopy(val_metrics),
                    "descriptor_cols": train_ds.descriptor_cols,
                    "target_col": train_ds.target_col,
                    "desc_scaler_path": str(train_ds.desc_scaler_path),
                    "y_scaler_path": str(train_ds.y_scaler_path),
                    "desc_scaler": load_scaler(train_ds.desc_scaler_path),
                    "y_scaler": load_scaler(train_ds.y_scaler_path),
                    "model_variant": cfg.model.variant,
                    "feature_version": FEATURE_VERSION,
                    "model_diagnostics": diagnostics,
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= int(cfg.train.patience):
                print(f"early stopping at epoch={epoch}")
                break

    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    ckpt = load_checkpoint(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_metrics, pred, true, row_idx, smiles = evaluate(
        model, loaders["test"], criterion, device, train_ds.y_scaler_path
    )
    save_predictions(run_dir / "test_predictions.csv", test_ds, pred, true, row_idx, smiles)
    summary = {
        "test_loss": test_loss,
        **test_metrics,
        "best_epoch": int(ckpt["epoch"]),
        "best_val_loss": float(ckpt["best_val_loss"]),
        "best_val_metrics": ckpt["val_metrics"],
        "model_diagnostics": ckpt.get("model_diagnostics", {}),
        "run_dir": str(run_dir),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
