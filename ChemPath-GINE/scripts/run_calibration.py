"""Train gine_desc with a Gaussian NLL head and evaluate calibration.

Modes
-----
mse       unchanged original entry point (chempath_gine.training.train.main),
          then a constant sigma = validation RMSE is attached as the naive
          homoscedastic uncertainty baseline.
nll       GaussianGINE trained with Gaussian NLL (mu, log sigma^2).
beta_nll  GaussianGINE trained with beta-NLL (beta=0.5), a sensitivity check.
faithful  GaussianGINE where the mean is trained with MSE exactly like the
          baseline and the variance head with Gaussian NLL on stop-gradient
          copies of both the mean and the shared representation ("faithful"
          heteroscedastic regression, Stirn et al. 2023).
          Added after the scaffold validation split showed plain NLL
          early-stopping before the mean converged. Because the shared
          checkpoint is chosen by val MSE, the variance head is then fitted
          to convergence in a second stage (mean path frozen, eval mode,
          early stopping on val NLL); see fit_variance_head().

Checkpoints are selected on validation loss only (MSE for `mse` and
`faithful`, Gaussian NLL for `nll`/`beta_nll`); the test split is evaluated once from the best
checkpoint. Splits, scalers and descriptor columns come from the same config
and code path as the archived 5-seed ablation runs.

Example
-------
python scripts/run_calibration.py \
  --config configs/esol_fusion_ablation_random.yaml \
  --mode nll --seeds 1 --out calibration_runs/esol_random
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chempath_gine.data.scaler import load_scaler
from chempath_gine.training.train import (
    _ns,
    build_datasets,
    build_model,
    load_checkpoint,
    make_loader,
    main as train_main,
    select_device,
    set_seed,
)
from chempath_gine.extensions import apply_model_extensions
from chempath_gine.uncertainty import (
    GaussianGINE,
    beta_nll_loss,
    calibration_metrics,
    coverage_curve,
    fit_variance_scale,
    gaussian_nll_loss,
)

MODES = ("mse", "nll", "beta_nll", "faithful")


def _parse_list(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_tag(mode: str, seed: int, member: int) -> str:
    return f"{mode}_seed{seed}" if member == 0 else f"{mode}_seed{seed}_m{member}"


def _seed_config(base: dict, mode: str, seed: int, out_dir: Path, device: str, member: int = 0) -> dict:
    cfg = copy.deepcopy(base)
    split_name = str(cfg["data"].get("split", "")).lower()
    cfg["run"]["seed"] = int(seed)
    if member > 0:
        # Deep-ensemble member: same split (split seed = seed), different initialisation/shuffle.
        cfg["run"]["init_seed"] = 1000 * int(seed) + int(member)
    cfg["run"]["device"] = device
    cfg["run"]["output_dir"] = str((out_dir / "runs").resolve())
    cfg["run"]["name"] = _run_tag(mode, seed, member)
    cfg["model"]["variant"] = cfg["model"].get("variant", "gine_desc")
    cache_key = "scaffold" if split_name == "scaffold" else f"seed{seed}"
    cfg["data"]["cache_dir"] = str((out_dir / "shared_cache" / mode / cache_key).resolve())
    cfg["data"]["csv"] = str((ROOT / cfg["data"]["csv"]).resolve()) if not Path(
        cfg["data"]["csv"]
    ).is_absolute() else cfg["data"]["csv"]
    return cfg


def _y_scaler(train_ds):
    scaler = load_scaler(train_ds.y_scaler_path)
    return float(scaler["mean"][0]), float(scaler["scale"][0])


@torch.no_grad()
def predict_point(model, loader, device, y_mean, y_scale):
    model.eval()
    mu, y, row_idx, smiles = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        mu.append(out.detach().cpu().numpy().reshape(-1))
        y.append(batch.y.view(-1).cpu().numpy())
        row_idx.extend(batch.row_idx.view(-1).cpu().tolist())
        smiles.extend(batch.smiles)
    mu = np.concatenate(mu) * y_scale + y_mean
    y = np.concatenate(y) * y_scale + y_mean
    return y, mu, row_idx, smiles


@torch.no_grad()
def predict_gaussian(model, loader, device, y_mean, y_scale):
    model.eval()
    mu, logvar, y, row_idx, smiles = [], [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        m, lv = model(batch)
        mu.append(m.cpu().numpy().reshape(-1))
        logvar.append(lv.cpu().numpy().reshape(-1))
        y.append(batch.y.view(-1).cpu().numpy())
        row_idx.extend(batch.row_idx.view(-1).cpu().tolist())
        smiles.extend(batch.smiles)
    mu = np.concatenate(mu) * y_scale + y_mean
    sigma = np.exp(0.5 * np.concatenate(logvar).astype(np.float64)) * y_scale
    y = np.concatenate(y) * y_scale + y_mean
    return y, mu, sigma, row_idx, smiles


def _write_predictions(path: Path, dataset, y, mu, sigma, row_idx, smiles):
    rows = []
    for i, yt, m, s, smi in zip(row_idx, y, mu, sigma, smiles):
        src = dataset.df.iloc[int(i)]
        rows.append(
            {
                "source_index": int(src["source_index"]) if "source_index" in src else int(i),
                "SMILES": smi,
                "y_true": float(yt),
                "mu": float(m),
                "sigma": float(s),
                "abs_error": abs(float(yt) - float(m)),
                "z": (float(yt) - float(m)) / float(s),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _calibration_block(y_val, mu_val, s_val, y_test, mu_test, s_test, fit_scale: bool) -> dict:
    scale = fit_variance_scale(y_val, mu_val, s_val) if fit_scale else 1.0
    levels, emp = coverage_curve(y_test, mu_test, s_test)
    _, emp_recal = coverage_curve(y_test, mu_test, s_test * scale)
    return {
        "val": calibration_metrics(y_val, mu_val, s_val),
        "test": calibration_metrics(y_test, mu_test, s_test),
        "variance_scale": scale,
        "test_recalibrated": calibration_metrics(y_test, mu_test, s_test * scale),
        "coverage_curve_test": {
            "levels": [float(v) for v in levels],
            "empirical": [float(v) for v in emp],
            "empirical_recalibrated": [float(v) for v in emp_recal],
        },
    }


@torch.no_grad()
def _collect_head_features(model, loader, device):
    model.eval()
    zs, mus, ys = [], [], []
    for batch in loader:
        batch = batch.to(device)
        z, mu = model.head_features(batch)
        zs.append(z)
        mus.append(mu)
        ys.append(batch.y.view(-1))
    return torch.cat(zs), torch.cat(mus), torch.cat(ys)


def fit_variance_head(model, loaders, device, seed: int, lr: float = 1e-3, max_epochs: int = 500,
                      patience: int = 50, batch_size: int = 64) -> dict:
    """Stage 2 of `faithful`: fit only logvar_out on frozen (eval-mode) features.

    Train on train-split residuals of the frozen mean, select the epoch by
    validation Gaussian NLL. The mean predictions are untouched.
    """
    z_tr, mu_tr, y_tr = _collect_head_features(model, loaders["train"], device)
    z_va, mu_va, y_va = _collect_head_features(model, loaders["val"], device)
    head = model.logvar_out
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)

    def val_nll():
        with torch.no_grad():
            return float(gaussian_nll_loss(mu_va, model.logvar_from_features(z_va), y_va).mean())

    start = val_nll()
    best, best_state, best_epoch, stale = start, copy.deepcopy(head.state_dict()), 0, 0
    for epoch in range(1, max_epochs + 1):
        perm = torch.randperm(len(y_tr), generator=gen)
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            loss = gaussian_nll_loss(mu_tr[idx], model.logvar_from_features(z_tr[idx]), y_tr[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        cur = val_nll()
        if cur < best - 1e-6:
            best, best_state, best_epoch, stale = cur, copy.deepcopy(head.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    head.load_state_dict(best_state)
    return {"val_nll_scaled_before": start, "val_nll_scaled_after": best,
            "best_epoch": best_epoch, "epochs_run": epoch, "lr": lr}


def run_mse(cfg_raw: dict, cfg_path: Path, threads: int) -> dict:
    """Baseline: unchanged original training entry, then constant sigma from val RMSE."""
    summary = train_main(["--config", str(cfg_path)])
    run_dir = Path(summary["run_dir"])
    cfg = _ns(cfg_raw)
    device = select_device(cfg.run.device)
    # Rebuild the deterministic split/scaler/cache for this run dir to re-evaluate val.
    train_ds, val_ds, test_ds = build_datasets(cfg, run_dir)
    model = build_model(cfg, train_ds).to(device)
    ckpt = load_checkpoint(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    y_mean, y_scale = _y_scaler(train_ds)
    bs, nw = int(cfg.train.batch_size), int(cfg.train.num_workers)
    y_val, mu_val, ri_val, smi_val = predict_point(
        model, make_loader(val_ds, bs, False, nw), device, y_mean, y_scale
    )
    y_test, mu_test, ri_test, smi_test = predict_point(
        model, make_loader(test_ds, bs, False, nw), device, y_mean, y_scale
    )
    sigma_const = float(np.sqrt(np.mean((y_val - mu_val) ** 2)))
    s_val = np.full_like(mu_val, sigma_const)
    s_test = np.full_like(mu_test, sigma_const)
    _write_predictions(run_dir / "predictions_val.csv", val_ds, y_val, mu_val, s_val, ri_val, smi_val)
    _write_predictions(
        run_dir / "predictions_test.csv", test_ds, y_test, mu_test, s_test, ri_test, smi_test
    )
    # Sanity: the re-evaluated test predictions must match train.main's own file.
    orig = pd.read_csv(run_dir / "test_predictions.csv")
    max_dev = float(np.max(np.abs(np.sort(orig["y_pred"].values) - np.sort(mu_test))))
    block = _calibration_block(y_val, mu_val, s_val, y_test, mu_test, s_test, fit_scale=False)
    return {
        "mode": "mse",
        "seed": int(cfg.run.seed),
        "split": str(cfg.data.split),
        "variant": str(cfg.model.variant),
        "run_dir": str(run_dir),
        "best_epoch": int(ckpt["epoch"]),
        "best_val_loss": float(ckpt["best_val_loss"]),
        "sigma_const": sigma_const,
        "reeval_max_abs_dev_vs_train_main": max_dev,
        "train_main_test_rmse": float(summary["rmse"]),
        "threads": threads,
        **block,
    }


def run_gaussian(cfg_raw: dict, mode: str, beta: float, threads: int) -> dict:
    cfg = _ns(cfg_raw)
    set_seed(int(cfg.run.seed))
    device = select_device(cfg.run.device)
    run_dir = Path(cfg.run.output_dir) / f"{cfg.run.name}_{cfg.model.variant}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_raw, f, sort_keys=False)

    train_ds, val_ds, test_ds = build_datasets(cfg, run_dir)
    init_seed = getattr(cfg.run, "init_seed", None)
    if init_seed is not None:
        set_seed(int(init_seed))  # splits/scalers above depend only on run.seed
    bs, nw = int(cfg.train.batch_size), int(cfg.train.num_workers)
    loaders = {
        "train": make_loader(train_ds, bs, True, nw),
        "val": make_loader(val_ds, bs, False, nw),
        "test": make_loader(test_ds, bs, False, nw),
    }
    # Optional refit on train+val for a fixed number of epochs with a replayed LR
    # schedule (see --refit-from). Validation is then in-sample: no selection on it.
    refit_epochs = getattr(cfg.train, "refit_epochs", None)
    refit_lrs = list(getattr(cfg.train, "refit_lrs", []) or [])
    if refit_epochs is not None:
        from torch.utils.data import ConcatDataset

        loaders["train"] = make_loader(ConcatDataset([train_ds, val_ds]), bs, True, nw)
    base = apply_model_extensions(build_model(cfg, train_ds), cfg)
    model = GaussianGINE(base, detach_variance_input=(mode == "faithful")).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay)
    )
    patience = int(cfg.train.patience)
    # Optional tuning knobs (absent keys reproduce the original behaviour exactly).
    sched_name = str(getattr(cfg.train, "scheduler", "plateau"))
    early_stop = bool(getattr(cfg.train, "early_stop", sched_name == "plateau"))
    ema_decay = getattr(cfg.train, "ema_decay", None)
    ema_model = copy.deepcopy(model) if ema_decay is not None else None
    if sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(2, patience // 3)
        )
    elif sched_name == "cosine":
        total_steps = int(cfg.train.epochs) * len(loaders["train"])
        warmup_steps = int(getattr(cfg.train, "warmup_epochs", 5)) * len(loaders["train"])
        min_ratio = float(getattr(cfg.train, "min_lr", 1e-5)) / float(cfg.train.lr)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"unknown train.scheduler={sched_name!r}")
    max_grad_norm = float(getattr(cfg.train, "max_grad_norm", 5.0))

    def train_loss_fn(mu, lv, y):
        if mode == "beta_nll":
            return beta_nll_loss(mu, lv, y, beta=beta).mean()
        if mode == "faithful":
            return ((mu - y) ** 2).mean() + gaussian_nll_loss(mu.detach(), lv, y).mean()
        return gaussian_nll_loss(mu, lv, y).mean()

    best_val, stale, best_path, history = float("inf"), 0, run_dir / "best.pt", []
    y_mean, y_scale = _y_scaler(train_ds)
    n_epochs = int(refit_epochs) if refit_epochs is not None else int(cfg.train.epochs)
    for epoch in range(1, n_epochs + 1):
        if refit_lrs:
            for group in optimizer.param_groups:
                group["lr"] = float(refit_lrs[min(epoch, len(refit_lrs)) - 1])
        model.train()
        total, steps = 0.0, 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            mu, lv = model(batch)
            loss = train_loss_fn(mu, lv, batch.y.view(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if sched_name == "cosine":
                scheduler.step()
            if ema_model is not None:
                with torch.no_grad():
                    for p_ema, p_cur in zip(ema_model.parameters(), model.parameters()):
                        p_ema.mul_(float(ema_decay)).add_(p_cur.detach(), alpha=1.0 - float(ema_decay))
            total += float(loss.item())
            steps += 1

        eval_model = ema_model if ema_model is not None else model
        eval_model.eval()
        val_losses, val_sq = [], []
        with torch.no_grad():
            for batch in loaders["val"]:
                batch = batch.to(device)
                mu, lv = eval_model(batch)
                y = batch.y.view(-1)
                val_losses.append(gaussian_nll_loss(mu, lv, y).cpu().numpy())
                val_sq.append(((mu - y) ** 2).cpu().numpy())
        val_nll = float(np.mean(np.concatenate(val_losses)))
        val_mse_scaled = float(np.mean(np.concatenate(val_sq)))
        val_rmse = float(np.sqrt(val_mse_scaled)) * y_scale
        # Selection criterion: the objective that trains the mean.
        val_loss = val_mse_scaled if mode == "faithful" else val_nll
        if sched_name == "plateau" and refit_epochs is None:
            scheduler.step(val_loss)
        history.append(
            {"epoch": epoch, "train_loss": total / max(1, steps), "val_nll_scaled": val_nll,
             "val_rmse": val_rmse, "val_selection_loss": val_loss,
             "lr": optimizer.param_groups[0]["lr"]}
        )
        print(f"[{mode} seed={cfg.run.seed}] epoch={epoch:03d} train={total / max(1, steps):.4f} "
              f"val_nll={val_nll:.4f} val_rmse={val_rmse:.4f}", flush=True)
        if refit_epochs is not None or val_loss < best_val:
            best_val, stale = val_loss, 0
            torch.save(
                {
                    "model_state": eval_model.state_dict(),
                    "config": copy.deepcopy(cfg_raw),
                    "mode": mode,
                    "beta": beta,
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "logvar_min": model.logvar_min,
                    "logvar_max": model.logvar_max,
                    "detach_variance_input": model.detach_variance_input,
                    "descriptor_cols": train_ds.descriptor_cols,
                    "y_scaler": load_scaler(train_ds.y_scaler_path),
                    "desc_scaler": load_scaler(train_ds.desc_scaler_path),
                    "model_variant": cfg.model.variant,
                },
                best_path,
            )
        else:
            stale += 1
            if early_stop and refit_epochs is None and stale >= patience:
                print(f"early stopping at epoch={epoch}", flush=True)
                break

    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    ckpt = load_checkpoint(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    variance_stage = None
    if mode == "faithful" and refit_epochs is not None:
        variance_stage = {"skipped": "refit on train+val leaves no held-out data for the variance head"}
    elif mode == "faithful":
        variance_stage = fit_variance_head(model, loaders, device, seed=int(cfg.run.seed))
        print(f"[faithful seed={cfg.run.seed}] variance head: {variance_stage}", flush=True)
        ckpt["model_state"] = model.state_dict()
        ckpt["variance_stage"] = variance_stage
        torch.save(ckpt, best_path)
    y_val, mu_val, s_val, ri_val, smi_val = predict_gaussian(
        model, loaders["val"], device, y_mean, y_scale
    )
    y_test, mu_test, s_test, ri_test, smi_test = predict_gaussian(
        model, loaders["test"], device, y_mean, y_scale
    )
    _write_predictions(run_dir / "predictions_val.csv", val_ds, y_val, mu_val, s_val, ri_val, smi_val)
    _write_predictions(
        run_dir / "predictions_test.csv", test_ds, y_test, mu_test, s_test, ri_test, smi_test
    )
    block = _calibration_block(y_val, mu_val, s_val, y_test, mu_test, s_test, fit_scale=True)
    return {
        "mode": mode,
        "beta": beta if mode == "beta_nll" else 0.0,
        "seed": int(cfg.run.seed),
        "split": str(cfg.data.split),
        "variant": str(cfg.model.variant),
        "run_dir": str(run_dir),
        "best_epoch": int(ckpt["epoch"]),
        "best_val_loss": float(ckpt["best_val_loss"]),
        "epochs_run": len(history),
        "refit_trainval": refit_epochs is not None,
        "refit_epochs": int(refit_epochs) if refit_epochs is not None else None,
        "variance_stage": variance_stage,
        "threads": threads,
        **block,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--out", required=True)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=None, help="override for smoke tests")
    parser.add_argument("--refit-from", default=None,
                        help="ROOT:MODE of train-only runs; refit on train+val for their best_epoch "
                             "epochs, replaying their per-epoch learning rates")
    parser.add_argument("--members", default="0",
                        help="deep-ensemble member ids; 0 = the base run, m>0 uses init seed 1000*seed+m")
    args = parser.parse_args(argv)
    members = [int(m) for m in _parse_list(args.members)]
    if args.mode == "mse" and members != [0]:
        raise SystemExit("--members is only supported for the Gaussian modes")

    torch.set_num_threads(max(1, args.threads))
    with open(args.config, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    out_dir = Path(args.out).resolve()
    (out_dir / "configs").mkdir(parents=True, exist_ok=True)

    jobs = [(int(s), m) for s in _parse_list(args.seeds) for m in members]
    for seed, member in jobs:
        tag = _run_tag(args.mode, seed, member)
        marker = out_dir / "summaries" / f"{tag}.json"
        if marker.exists():
            stored = json.loads(marker.read_text()).get("threads")
            if stored != args.threads:
                raise SystemExit(
                    f"{marker} was produced with threads={stored}, requested {args.threads}; "
                    "results depend on the CPU thread count, delete the marker to rerun"
                )
            print(f"skip existing {marker}")
            continue
        cfg_raw = _seed_config(base, args.mode, seed, out_dir, args.device, member)
        if args.epochs is not None:
            cfg_raw["train"]["epochs"] = int(args.epochs)
        if args.refit_from:
            ref_root, ref_mode = args.refit_from.rsplit(":", 1)
            ref = json.loads((Path(ref_root) / "summaries" / f"{_run_tag(ref_mode, seed, member)}.json").read_text())
            ref_hist = json.loads((Path(ref["run_dir"]) / "history.json").read_text())
            best = int(ref["best_epoch"])
            # history[e]["lr"] is the LR after epoch e+1's scheduler step, i.e. the LR of epoch e+2.
            cfg_raw["train"]["refit_epochs"] = best
            cfg_raw["train"]["refit_lrs"] = [float(cfg_raw["train"]["lr"])] + [float(h["lr"]) for h in ref_hist[: best - 1]]
            cfg_raw["train"]["refit_reference"] = str(ref["run_dir"])
        cfg_path = out_dir / "configs" / f"{tag}.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_raw, f, sort_keys=False)
        if args.mode == "mse":
            result = run_mse(cfg_raw, cfg_path, args.threads)
        else:
            result = run_gaussian(cfg_raw, args.mode, args.beta, args.threads)
        result["config_path"] = str(cfg_path)
        result["member"] = member
        result["init_seed"] = cfg_raw["run"].get("init_seed", seed)
        marker.parent.mkdir(parents=True, exist_ok=True)
        with marker.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        v = result["val"]  # test metrics stay in the summary JSON only (no test numbers in logs)
        print(
            f"DONE mode={args.mode} seed={seed} member={member} split={result['split']} "
            f"val_rmse={v['rmse']:.4f} val_nll={v['nll']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
