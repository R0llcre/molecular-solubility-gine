"""Cross-architecture deep ensembles (e.g. GINE + D-MPNN members).

A system is a list of components ROOT:MODE:MEMBERS, where ROOT has
summaries/<MODE>_seed<s>[_m<k>].json (run_calibration layout) and MEMBERS is a
range like 0-4. For every outer seed s, all listed members (which share the
split of seed s) are combined with mixture moments (build_ensembles._combine)
and evaluated with build_ensembles.evaluate. Output runs follow the
run_calibration layout so analysis scripts can consume them.

Selection must use validation only: the printed table shows validation RMSE;
test metrics are written to the JSON/CSV files but never printed unless
--show-test is given (once, after the decision is locked).

Example
-------
python scripts/build_mixed_ensembles.py --split-roots random,scaffold \
  --system "gine10=calibration_runs/perf/desc7_logp/esol_{split}:faithful:0-9" \
  --system "gine5_dmpnn5=calibration_runs/perf/desc7_logp/esol_{split}:faithful:0-4;calibration_runs/perf/desc7_dmpnn/esol_{split}:faithful:0-4" \
  --out calibration_runs/r3/ensembles
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from build_ensembles import _combine, _flat, _load_member, evaluate  # noqa: E402


def _members(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _parse_system(text: str):
    name, comps = text.split("=", 1)
    parsed = []
    for comp in comps.split(";"):
        root, mode, members = comp.rsplit(":", 2)
        parsed.append((root, mode, _members(members)))
    return name, parsed


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", action="append", required=True)
    parser.add_argument("--split-roots", default="random,scaffold",
                        help="values substituted for {split} in component roots")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--out", required=True)
    parser.add_argument("--show-test", action="store_true")
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.out)
    rows = []
    for text in args.system:
        name, comps = _parse_system(text)
        for split_key in [s.strip() for s in args.split_roots.split(",") if s.strip()]:
            for s in seeds:
                mems = []
                for root, mode, members in comps:
                    r = ROOT / root.format(split=split_key)
                    for m in members:
                        tag = f"{mode}_seed{s}" if m == 0 else f"{mode}_seed{s}_m{m}"
                        mems.append(_load_member(r / "summaries" / f"{tag}.json"))
                split = mems[0]["summary"]["split"]
                val = _combine([m["val"] for m in mems])
                test = _combine([m["test"] for m in mems])
                ev = evaluate(val, test)
                run_dir = out / name / f"esol_{split_key}" / "runs" / f"{name}_seed{s}"
                (run_dir / "splits").mkdir(parents=True, exist_ok=True)
                src_splits = mems[0]["run_dir"] / "splits"
                for part in ("train", "val", "test"):
                    if (src_splits / f"{part}.csv").exists():
                        shutil.copy(src_splits / f"{part}.csv", run_dir / "splits" / f"{part}.csv")
                val.to_csv(run_dir / "predictions_val.csv", index=False)
                test.to_csv(run_dir / "predictions_test.csv", index=False)
                member_val = [float(np.sqrt(np.mean((m["val"]["y_true"] - m["val"]["mu"]) ** 2))) for m in mems]
                extra = {"n_members": len(mems), "member_val_rmse_mean": float(np.mean(member_val)),
                         "epistemic_var_share": float(np.mean(test["sigma_epistemic"] ** 2) / np.mean(test["sigma"] ** 2))}
                summary = {"mode": name, "split": split, "seed": s, "run_dir": str(run_dir),
                           "members": [str(m["run_dir"]) for m in mems], **extra, **ev}
                sdir = out / name / f"esol_{split_key}" / "summaries"
                sdir.mkdir(parents=True, exist_ok=True)
                (sdir / f"{name}_seed{s}.json").write_text(json.dumps(summary, indent=2))
                row = _flat(split_key, name, s, ev, extra)
                rows.append(row)
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "seed_level.csv", index=False)
    g = df.groupby(["split", "method"], sort=False)
    agg = g.mean(numeric_only=True).add_suffix("_mean").join(g.std(ddof=1, numeric_only=True).add_suffix("_std")).reset_index()
    agg.to_csv(out / "aggregate.csv", index=False)
    cols = ["split", "method", "n_members_mean", "val_rmse_mean", "val_rmse_std", "member_val_rmse_mean_mean",
            "epistemic_var_share_mean"]
    if args.show_test:
        cols += ["raw_rmse_mean", "raw_rmse_std", "recal_coverage_90_mean", "recal_width_90_mean",
                 "conformal_coverage_90_mean", "conformal_width_90_mean"]
    pd.set_option("display.width", 250)
    print(agg[[c for c in cols if c in agg.columns]].round(4).to_string(index=False))
    piv = agg.pivot(index="method", columns="split", values="val_rmse_mean")
    piv["sum"] = piv.sum(axis=1)
    print("\nvalidation RMSE (mean over seeds), sum over splits:")
    print(piv.sort_values("sum").round(4).to_string())


if __name__ == "__main__":
    main()
