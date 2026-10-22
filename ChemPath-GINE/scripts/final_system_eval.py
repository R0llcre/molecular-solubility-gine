"""Evaluate the final system: refit-ensemble mean + train-only-ensemble calibrated intervals.

Point prediction: mu from the ensemble refit on train+val (more data, no held-out set).
Uncertainty: sigma from the same system trained on train only, calibrated on validation,
either Gaussian-scaled (s * sigma, s fitted on validation NLL) or normalized split
conformal (q_90 * sigma). Intervals are centred on the refit mu.

Inputs are ensemble roots in the run_calibration layout (summaries/<name>_seed<s>.json
with run_dir/predictions_test.csv), e.g. produced by build_mixed_ensembles.py.

Example
-------
python scripts/final_system_eval.py --refit ROOT:NAME --trainonly ROOT:NAME --seeds 1,2,3,4,5 --out FILE.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def _load(spec: str, seed: int):
    root, name = spec.rsplit(":", 1)
    summ = json.loads((Path(root) / "summaries" / f"{name}_seed{seed}.json").read_text())
    pred = pd.read_csv(Path(summ["run_dir"]) / "predictions_test.csv").sort_values("source_index").reset_index(drop=True)
    return summ, pred


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--refit", required=True)
    p.add_argument("--trainonly", required=True)
    p.add_argument("--seeds", default="1,2,3,4,5")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    z90 = float(norm.ppf(0.95))
    rows = []
    for s in [int(x) for x in a.seeds.split(",")]:
        s_ref, p_ref = _load(a.refit, s)
        s_tr, p_tr = _load(a.trainonly, s)
        assert np.array_equal(p_ref["source_index"].values, p_tr["source_index"].values)
        y, mu_ref, mu_tr, sig_tr = p_ref["y_true"].values, p_ref["mu"].values, p_tr["mu"].values, p_tr["sigma"].values
        scale = float(s_tr["variance_scale"])
        q90 = float(s_tr["conformal"]["q_90"])
        row = {"seed": s,
               "rmse_refit": float(np.sqrt(np.mean((y - mu_ref) ** 2))),
               "rmse_trainonly": float(np.sqrt(np.mean((y - mu_tr) ** 2)))}
        for tag, mu in (("refit_mu", mu_ref), ("trainonly_mu", mu_tr)):
            hw_g = z90 * scale * sig_tr
            hw_c = q90 * sig_tr
            row[f"{tag}_gauss_cov90"] = float(np.mean(np.abs(y - mu) <= hw_g))
            row[f"{tag}_conformal_cov90"] = float(np.mean(np.abs(y - mu) <= hw_c))
        row["gauss_width90"] = float(np.mean(2 * z90 * scale * sig_tr))
        row["conformal_width90"] = float(np.mean(2 * q90 * sig_tr))
        rows.append(row)
    df = pd.DataFrame(rows)
    summary = {c: [float(df[c].mean()), float(df[c].std(ddof=1))] for c in df.columns if c != "seed"}
    Path(a.out).write_text(json.dumps({"per_seed": rows, "mean_std": summary}, indent=2))
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))
    print({k: [round(v[0], 4), round(v[1], 4)] for k, v in summary.items()})


if __name__ == "__main__":
    main()
