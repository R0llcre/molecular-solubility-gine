"""Validation-first summary of tuning arms (faithful mode = MSE-trained mean).

Selection must use validation only. Test columns are hidden unless
--show-test is passed, which should happen once, after the configuration is
locked.

Layout: calibration_runs/perf/<arm>/esol_<split>/summaries/faithful_seed<k>.json;
the baseline arm is the existing calibration_runs/esol_<split>/ faithful runs.

Example
-------
python scripts/summarize_perf.py --perf-root calibration_runs/perf --out calibration_runs/perf/screen_val.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("random", "scaffold")


def _load(arm_root: Path, split: str, seeds, mode: str) -> dict[int, dict]:
    out = {}
    for s in seeds:
        p = arm_root / f"esol_{split}" / "summaries" / f"{mode}_seed{s}.json"
        if p.exists():
            out[s] = json.loads(p.read_text())
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf-root", default=str(ROOT / "calibration_runs/perf"))
    parser.add_argument("--baseline-root", default=str(ROOT / "calibration_runs"))
    parser.add_argument("--mode", default="faithful")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--arms", default="", help="comma list; default = every arm directory")
    parser.add_argument("--out", default=None)
    parser.add_argument("--show-test", action="store_true")
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    perf_root = Path(args.perf_root)
    arms = [a for a in args.arms.split(",") if a.strip()] or sorted(
        p.name for p in perf_root.iterdir() if p.is_dir() and (p / "esol_random").exists()
    )
    base = {sp: _load(Path(args.baseline_root), sp, seeds, args.mode) for sp in SPLITS}

    rows = []
    for arm in ["baseline"] + arms:
        row = {"arm": arm}
        complete = True
        for sp in SPLITS:
            runs = base[sp] if arm == "baseline" else _load(perf_root / arm, sp, seeds, args.mode)
            common = sorted(set(runs) & set(base[sp]))
            if len(common) < len(seeds):
                complete = False
            if not common:
                continue
            v = np.array([runs[s]["val"]["rmse"] for s in common])
            vb = np.array([base[sp][s]["val"]["rmse"] for s in common])
            d = v - vb
            row.update({
                f"{sp}_n": len(common),
                f"{sp}_val_rmse": v.mean(), f"{sp}_val_rmse_std": v.std(ddof=1) if len(v) > 1 else 0.0,
                f"{sp}_val_delta": d.mean(), f"{sp}_val_delta_std": d.std(ddof=1) if len(d) > 1 else 0.0,
                f"{sp}_seeds_better": int((d < 0).sum()),
                f"{sp}_best_epoch": float(np.mean([runs[s]["best_epoch"] for s in common])),
            })
            if args.show_test:
                t = np.array([runs[s]["test"]["rmse"] for s in common])
                tb = np.array([base[sp][s]["test"]["rmse"] for s in common])
                row.update({f"{sp}_test_rmse": t.mean(), f"{sp}_test_rmse_std": t.std(ddof=1) if len(t) > 1 else 0.0,
                            f"{sp}_test_delta": (t - tb).mean(), f"{sp}_test_seeds_better": int((t - tb < 0).sum())})
        row["complete"] = complete
        if "random_val_delta" in row and "scaffold_val_delta" in row:
            row["mean_val_delta"] = 0.5 * (row["random_val_delta"] + row["scaffold_val_delta"])
        rows.append(row)
    df = pd.DataFrame(rows)
    if "mean_val_delta" in df:
        df = df.sort_values("mean_val_delta", na_position="first")
    if args.out:
        df.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    cols = ["arm", "complete", "mean_val_delta"] + [
        f"{sp}_{c}" for sp in SPLITS for c in ("val_rmse", "val_delta", "seeds_better", "best_epoch")
    ]
    if args.show_test:
        cols += [f"{sp}_{c}" for sp in SPLITS for c in ("test_rmse", "test_delta", "test_seeds_better")]
    print(df[[c for c in cols if c in df.columns]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
