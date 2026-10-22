"""One-time test reveal for the locked tuning decision.

Reads calibration_runs/locked/lock.json (written after selection on validation
only; note that run logs printed test RMSE until 2026-10-18, so the lock is a
procedural commitment, not an enforced seal) and reports test RMSE for exactly the rows declared
there, with paired per-seed differences. Writes reveal.json and reveal.md next
to the lock file and refuses to run twice unless --force is given.

Example
-------
python scripts/reveal_locked.py --lock calibration_runs/locked/lock.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("random", "scaffold")


def _summary(path: Path) -> dict:
    return json.loads(path.read_text())


def _rows(spec: dict, split: str, seeds) -> dict:
    """Return per-seed test/val RMSE (+ optional extras) for a declared row."""
    kind = spec["kind"]
    out = {"test": [], "val": [], "extra": {}}
    for s in seeds:
        if kind == "runs":
            d = _summary(ROOT / spec["root"].format(split=split) / "summaries" / f"{spec['mode']}_seed{s}.json")
            out["test"].append(d["test"]["rmse"])
            out["val"].append(d["val"]["rmse"])
        elif kind == "ensemble":
            d = _summary(ROOT / spec["root"].format(split=split) / "summaries" / f"{spec['mode']}_seed{s}.json")
            out["test"].append(d["test"]["rmse"])
            out["val"].append(d["val"]["rmse"])
            for key in ("member_rmse_mean",):
                out["extra"].setdefault(key, []).append(d[key])
            for blk, keys in (("test", ("nll", "ece", "coverage_90", "width_90")),
                              ("test_recalibrated", ("ece", "coverage_90", "width_90")),
                              ("conformal", ("ece", "coverage_90", "width_90"))):
                for k in keys:
                    out["extra"].setdefault(f"{blk}_{k}", []).append(d[blk][k])
        else:
            raise ValueError(kind)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default=str(ROOT / "calibration_runs/locked/lock.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    lock_path = Path(args.lock)
    lock = json.loads(lock_path.read_text())
    out_json = lock_path.parent / "reveal.json"
    if out_json.exists() and not args.force:
        raise SystemExit(f"{out_json} exists: the test split was already revealed once")
    seeds = lock["seeds"]

    results = {}
    for split in SPLITS:
        res = {}
        for name, spec in lock["rows"].items():
            r = _rows(spec, split, seeds)
            res[name] = {
                "label": spec["label"],
                "test_per_seed": r["test"], "val_per_seed": r["val"],
                "test_mean": float(np.mean(r["test"])), "test_std": float(np.std(r["test"], ddof=1)),
                "val_mean": float(np.mean(r["val"])),
                **{k: float(np.mean(v)) for k, v in r["extra"].items()},
            }
        for name, ref in lock["paired"]:
            a, b = np.array(res[name]["test_per_seed"]), np.array(res[ref]["test_per_seed"])
            d = a - b
            res[name].setdefault("paired", {})[ref] = {
                "diff_mean": float(d.mean()), "diff_std": float(d.std(ddof=1)),
                "n_seeds_better": int((d < 0).sum()), "paired_t_p": float(ttest_rel(a, b).pvalue),
            }
        # Hybrid row, using the frozen blend weight.
        hy = _summary(ROOT / lock["hybrid"]["reveal_json"].format(split=split)) if lock.get("hybrid") else None
        if hy:
            res["hybrid"] = hy[split]
        results[split] = res

    out_json.write_text(json.dumps(results, indent=2))
    lines = [f"# Locked reveal ({lock['locked_at']})", "", f"Frozen config: {lock['frozen_config']}", ""]
    for split in SPLITS:
        lines += [f"## {split}", "", "| row | test RMSE | val RMSE | paired Δ vs ref (seeds better, p) |", "|---|---:|---:|---|"]
        for name, r in results[split].items():
            if name == "hybrid":
                continue
            paired = "; ".join(f"vs {ref}: {p['diff_mean']:+.4f} ({p['n_seeds_better']}/5, p={p['paired_t_p']:.2f})"
                               for ref, p in r.get("paired", {}).items())
            lines.append(f"| {r['label']} | {r['test_mean']:.4f} ± {r['test_std']:.4f} | {r['val_mean']:.4f} | {paired} |")
        lines.append("")
    (lock_path.parent / "reveal.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
