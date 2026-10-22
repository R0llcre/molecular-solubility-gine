"""Paired molecule-level bootstrap comparison of test RMSE between two run sets.

Each side is given as ROOT:MODE, where ROOT is a run_calibration.py / build_ensembles.py
output root containing summaries/<MODE>_seed<k>.json, each pointing ("run_dir") at a
directory with predictions_test.csv (columns source_index, y_true, mu).

For every seed k the A and B test predictions are aligned on source_index (the molecule
sets and y_true must match), and the difference

    diff = RMSE_A - RMSE_B          (negative = A has the lower test RMSE)

is bootstrapped by resampling test molecules with replacement; the same resampled
molecules are used for A and B (paired bootstrap). Reported per seed: RMSE_A, RMSE_B,
diff, a percentile CI and the fraction of bootstrap replicates with diff < 0.

Pooled over seeds (point estimate = mean over seeds of the per-seed diff):
  * shared test set (all seeds evaluate the same molecules, e.g. a deterministic scaffold
    split): molecules are resampled ONCE per replicate and the per-seed diffs computed on
    that common resample are averaged. Seeds are therefore not treated as independent
    samples of molecules; the CI reflects test-molecule sampling noise only.
  * per-seed test sets (random splits, different molecules per seed): molecules are
    resampled independently within each seed and the per-seed diffs are averaged per
    replicate. Molecules that fall in the test set of several seeds are resampled
    independently in each seed (a mild anti-conservative approximation).
The mode is detected automatically (--pooling auto) and can be forced for diagnostics.

The CI is conditional on the trained models: it quantifies how much the RMSE gap could
move under a different draw of test molecules, not training-seed variability.

Example
-------
python scripts/bootstrap_compare.py \
  --a calibration_runs/esol_scaffold:nll --b calibration_runs/esol_scaffold:mse \
  --label-a "Gaussian NLL" --label-b "MSE" \
  --seeds 1,2,3,4,5 --n-boot 10000 --rng 20261018 --out compare_scaffold.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED_FILE = "predictions_test.csv"
PRED_COLS = ["source_index", "y_true", "mu"]
SUMMARY_RMSE_TOL = 1e-5


# ----------------------------------------------------------------------------- I/O


def read_bytes_robust(path: Path, retries: int = 8, delay: float = 1.0) -> bytes:
    """Read a file fully, retrying on timeouts / short reads (iCloud-evicted files)."""
    path = Path(path)
    last = "unknown error"
    for attempt in range(retries):
        try:
            size = os.stat(path).st_size
            with path.open("rb") as f:
                data = f.read()
            if size > 0 and len(data) == size:
                return data
            last = f"short read ({len(data)} of {size} bytes)"
        except FileNotFoundError:
            raise
        except (OSError, TimeoutError) as exc:  # e.g. ETIMEDOUT on an evicted file
            last = repr(exc)
        if attempt + 1 < retries:
            time.sleep(delay)
    raise OSError(f"could not read {path} after {retries} attempts: {last}")


def parse_spec(spec: str) -> tuple[Path, str]:
    """'ROOT:MODE' -> (resolved root, mode). Splits on the last colon."""
    if ":" not in spec:
        raise ValueError(f"expected ROOT:MODE, got {spec!r}")
    root_str, mode = spec.rsplit(":", 1)
    if not root_str or not mode:
        raise ValueError(f"expected ROOT:MODE, got {spec!r}")
    root = Path(root_str).expanduser()
    if not root.is_absolute() and not root.exists() and (ROOT / root).exists():
        root = ROOT / root
    if not root.exists():
        raise FileNotFoundError(f"run root not found: {root_str}")
    return root.resolve(), mode


def _resolve_run_dir(run_dir: str, summary_path: Path) -> Path:
    cand = Path(run_dir)
    tried = [cand]
    if cand.exists():
        return cand
    if not cand.is_absolute():
        tried.append(ROOT / cand)
    # the root may have been moved / synced elsewhere: <root>/runs/<basename>
    tried.append(summary_path.parent.parent / "runs" / cand.name)
    for c in tried[1:]:
        if c.exists():
            return c
    tried_str = [str(t) for t in tried]
    raise FileNotFoundError(f"run_dir from {summary_path} not found; tried {tried_str}")


def load_run(root: Path, mode: str, seed: int) -> dict:
    summary_path = root / "summaries" / f"{mode}_seed{seed}.json"
    summary = json.loads(read_bytes_robust(summary_path).decode("utf-8"))
    run_dir = _resolve_run_dir(summary["run_dir"], summary_path)
    df = pd.read_csv(io.BytesIO(read_bytes_robust(run_dir / PRED_FILE)))
    missing = [c for c in PRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{run_dir / PRED_FILE} lacks columns {missing}")
    df = df[PRED_COLS].copy()
    if df.isna().any().any():
        raise ValueError(f"{run_dir / PRED_FILE} contains NaN values")
    if not df["source_index"].is_unique:
        raise ValueError(f"{run_dir / PRED_FILE} has duplicate source_index values")
    test_block = summary.get("test")
    summary_rmse = test_block.get("rmse") if isinstance(test_block, dict) else None
    return {
        "summary_path": str(summary_path),
        "run_dir": str(run_dir),
        "df": df,
        "summary_rmse": None if summary_rmse is None else float(summary_rmse),
    }


def align_pair(
    df_a: pd.DataFrame, df_b: pd.DataFrame, y_atol: float = 1e-6, context: str = ""
) -> dict:
    """Align two prediction frames on source_index; molecule sets and y_true must agree."""
    a = df_a.sort_values("source_index").reset_index(drop=True)
    b = df_b.sort_values("source_index").reset_index(drop=True)
    ids_a = a["source_index"].to_numpy()
    ids_b = b["source_index"].to_numpy()
    if len(ids_a) != len(ids_b) or not np.array_equal(ids_a, ids_b):
        only_a = sorted(set(ids_a.tolist()) - set(ids_b.tolist()))
        only_b = sorted(set(ids_b.tolist()) - set(ids_a.tolist()))
        raise ValueError(
            f"{context}: test molecule sets differ (|A|={len(ids_a)}, |B|={len(ids_b)}, "
            f"only in A: {only_a[:10]}, only in B: {only_b[:10]})"
        )
    y_a = a["y_true"].to_numpy(dtype=np.float64)
    y_b = b["y_true"].to_numpy(dtype=np.float64)
    max_dy = float(np.max(np.abs(y_a - y_b))) if len(y_a) else 0.0
    if max_dy > y_atol:
        raise ValueError(f"{context}: y_true differs between A and B (max |dy| = {max_dy:.3g})")
    return {
        "source_index": ids_a,
        "y": y_a,
        "mu_a": a["mu"].to_numpy(dtype=np.float64),
        "mu_b": b["mu"].to_numpy(dtype=np.float64),
        "max_abs_dy": max_dy,
    }


# ------------------------------------------------------------------------ statistics


def rmse(y: np.ndarray, mu: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(mu)) ** 2)))


def boot_rmse_diff(
    sq_a: np.ndarray, sq_b: np.ndarray, idx: np.ndarray, chunk: int = 2000
) -> np.ndarray:
    """Bootstrap replicates of RMSE_A - RMSE_B for resample index matrix idx (n_boot x n)."""
    out = np.empty(idx.shape[0], dtype=np.float64)
    for start in range(0, idx.shape[0], chunk):
        sl = idx[start:start + chunk]
        out[start:start + chunk] = np.sqrt(sq_a[sl].mean(axis=1)) - np.sqrt(sq_b[sl].mean(axis=1))
    return out


def summarize_diffs(point: float, reps: np.ndarray, ci_level: float) -> dict:
    lo_q, hi_q = (1.0 - ci_level) / 2.0, 1.0 - (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(reps, [lo_q, hi_q])
    return {
        "diff": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_diff_lt_0": float(np.mean(reps < 0)),
        "p_diff_gt_0": float(np.mean(reps > 0)),
        "boot_se": float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.0,
        "boot_mean": float(np.mean(reps)),
    }


def _shared_test_set(pairs: list[dict]) -> bool:
    ref = pairs[0]
    for p in pairs[1:]:
        if len(p["source_index"]) != len(ref["source_index"]):
            return False
        if not np.array_equal(p["source_index"], ref["source_index"]):
            return False
        if not np.allclose(p["y"], ref["y"], rtol=0.0, atol=1e-6):
            return False
    return True


def compare(
    pairs: list[dict],
    n_boot: int = 10000,
    rng_seed: int = 20261018,
    ci_level: float = 0.95,
    pooling: str = "auto",
    return_replicates: bool = False,
) -> dict:
    """Paired bootstrap of RMSE_A - RMSE_B per seed and pooled over seeds.

    pairs: list of dicts with keys seed, source_index (sorted), y, mu_a, mu_b (aligned arrays).
    pooling: 'auto' | 'shared' | 'independent'.
    """
    if not pairs:
        raise ValueError("no seeds to compare")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be in (0, 1)")
    shared = _shared_test_set(pairs)
    if pooling == "auto":
        use_shared = shared
    elif pooling == "shared":
        if not shared:
            raise ValueError(
                "pooling='shared' requested but seeds have different test molecule sets"
            )
        use_shared = True
    elif pooling == "independent":
        use_shared = False
    else:
        raise ValueError(f"unknown pooling {pooling!r}")

    if use_shared:
        n = len(pairs[0]["y"])
        idx_shared = np.random.default_rng(rng_seed).integers(0, n, size=(n_boot, n))
        idx_list = [idx_shared] * len(pairs)
    else:
        children = np.random.SeedSequence(rng_seed).spawn(len(pairs))
        idx_list = [
            np.random.default_rng(c).integers(0, len(p["y"]), size=(n_boot, len(p["y"])))
            for c, p in zip(children, pairs)
        ]

    per_seed, reps_all = [], []
    for p, idx in zip(pairs, idx_list):
        sq_a = (p["y"] - p["mu_a"]) ** 2
        sq_b = (p["y"] - p["mu_b"]) ** 2
        r_a, r_b = rmse(p["y"], p["mu_a"]), rmse(p["y"], p["mu_b"])
        reps = boot_rmse_diff(sq_a, sq_b, idx)
        reps_all.append(reps)
        row = {"seed": p["seed"], "n": int(len(p["y"])), "rmse_a": r_a, "rmse_b": r_b}
        row.update(summarize_diffs(r_a - r_b, reps, ci_level))
        per_seed.append(row)

    reps_all = np.vstack(reps_all)  # (n_seeds, n_boot)
    pooled_reps = reps_all.mean(axis=0)
    diffs = np.array([r["diff"] for r in per_seed])
    pooled = {
        "n_seeds": len(per_seed),
        "mean_rmse_a": float(np.mean([r["rmse_a"] for r in per_seed])),
        "mean_rmse_b": float(np.mean([r["rmse_b"] for r in per_seed])),
        "n_seeds_a_lower": int(np.sum(diffs < 0)),
    }
    pooled.update(summarize_diffs(float(diffs.mean()), pooled_reps, ci_level))

    union = set()
    for p in pairs:
        union.update(np.asarray(p["source_index"]).tolist())
    counts: dict = {}
    for p in pairs:
        for s in np.asarray(p["source_index"]).tolist():
            counts[s] = counts.get(s, 0) + 1
    result = {
        "pooling": "shared_test_set" if use_shared else "per_seed_test_sets",
        "test_sets": {
            "identical_across_seeds": bool(shared),
            "n_per_seed": [int(len(p["y"])) for p in pairs],
            "n_union": int(len(union)),
            "n_in_multiple_seeds": int(sum(1 for c in counts.values() if c > 1)),
        },
        "per_seed": per_seed,
        "pooled": pooled,
    }
    if return_replicates:
        result["_replicates"] = {"per_seed": reps_all, "pooled": pooled_reps}
    return result


# ------------------------------------------------------------------------------- CLI


def _fmt(x: float) -> str:
    return f"{x:+.4f}"


def print_report(res: dict) -> None:
    a, b = res["a"]["label"], res["b"]["label"]
    lvl = int(round(100 * res["ci_level"]))
    print(f"A = {a}   B = {b}   diff = RMSE_A - RMSE_B (negative: A better)")
    print(f"pooling: {res['pooling']}   n_boot={res['n_boot']}   rng={res['rng']}")
    print(f"{'seed':>6} {'n':>4} {'RMSE_A':>8} {'RMSE_B':>8} {'diff':>8} "
          f"{f'{lvl}% CI':>20} {'P(d<0)':>7}")
    for r in res["per_seed"]:
        print(f"{r['seed']:>6} {r['n']:>4} {r['rmse_a']:>8.4f} {r['rmse_b']:>8.4f} "
              f"{_fmt(r['diff']):>8} "
              f"  [{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}] {r['p_diff_lt_0']:>7.3f}")
    p = res["pooled"]
    print(f"{'pooled':>6} {'':>4} {p['mean_rmse_a']:>8.4f} {p['mean_rmse_b']:>8.4f} "
          f"{_fmt(p['diff']):>8} "
          f"  [{_fmt(p['ci_low'])}, {_fmt(p['ci_high'])}] {p['p_diff_lt_0']:>7.3f}")
    for w in res.get("warnings", []):
        print(f"WARNING: {w}", file=sys.stderr)


def run(args: argparse.Namespace) -> dict:
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds in {args.seeds!r}")
    root_a, mode_a = parse_spec(args.a)
    root_b, mode_b = parse_spec(args.b)
    label_a = args.label_a or f"{root_a.name}:{mode_a}"
    label_b = args.label_b or f"{root_b.name}:{mode_b}"

    pairs, meta, warnings = [], [], []
    for s in seeds:
        ra, rb = load_run(root_a, mode_a, s), load_run(root_b, mode_b, s)
        if Path(ra["run_dir"]).resolve() == Path(rb["run_dir"]).resolve():
            warnings.append(f"seed {s}: A and B point at the same run_dir {ra['run_dir']}")
        al = align_pair(ra["df"], rb["df"], y_atol=args.y_atol, context=f"seed {s}")
        al["seed"] = s
        pairs.append(al)
        m = {"seed": s, "run_dir_a": ra["run_dir"], "run_dir_b": rb["run_dir"],
             "summary_rmse_a": ra["summary_rmse"], "summary_rmse_b": rb["summary_rmse"],
             "max_abs_dy": al["max_abs_dy"]}
        for side, key in (("a", "mu_a"), ("b", "mu_b")):
            srm = m[f"summary_rmse_{side}"]
            if srm is not None:
                own = rmse(al["y"], al[key])
                if abs(own - srm) > SUMMARY_RMSE_TOL:
                    warnings.append(f"seed {s}: recomputed RMSE_{side.upper()} {own:.6f} != "
                                    f"summary test rmse {srm:.6f}")
        meta.append(m)

    res = compare(pairs, n_boot=args.n_boot, rng_seed=args.rng, ci_level=args.ci,
                  pooling=args.pooling)
    for row, m in zip(res["per_seed"], meta):
        row.update(m)

    notes = [
        "diff = RMSE_A - RMSE_B on the test set; negative means A has the lower RMSE.",
        "Paired bootstrap over test molecules (same resample for A and B); percentile CI.",
        "Pooled point estimate = mean over seeds of the per-seed diff.",
    ]
    if res["pooling"] == "shared_test_set":
        notes.append("All seeds share one test set: molecules are resampled once per replicate and "
                     "the per-seed diffs on that resample are averaged (seeds are not treated as "
                     "independent molecule samples).")
    else:
        notes.append("Seeds have different test sets: molecules are resampled independently within "
                     "each seed and the per-seed diffs are averaged per replicate; molecules "
                     "present in several seeds' test sets are resampled independently per seed.")
    notes.append("The CI reflects test-molecule sampling noise given the trained models, not "
                 "training-seed variability.")

    out = {
        "tool": "scripts/bootstrap_compare.py",
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "a": {"label": label_a, "root": str(root_a), "mode": mode_a, "spec": args.a},
        "b": {"label": label_b, "root": str(root_b), "mode": mode_b, "spec": args.b},
        "seeds": seeds,
        "n_boot": int(args.n_boot),
        "rng": int(args.rng),
        "ci_level": float(args.ci),
        "diff_definition": "RMSE_A - RMSE_B (negative = A has lower test RMSE)",
        **res,
        "warnings": warnings,
        "notes": notes,
    }
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True,
                    help="ROOT:MODE for side A (summaries/<MODE>_seed<k>.json)")
    ap.add_argument("--b", required=True, help="ROOT:MODE for side B")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--rng", type=int, default=20261018)
    ap.add_argument("--ci", type=float, default=0.95, help="CI level (default 0.95)")
    ap.add_argument("--pooling", choices=["auto", "shared", "independent"], default="auto",
                    help="auto: shared resample iff all seeds have the same test molecules")
    ap.add_argument("--y-atol", type=float, default=1e-6,
                    help="max allowed |y_true_A - y_true_B| per molecule")
    ap.add_argument("--out", default=None, help="write the result JSON here")
    ap.add_argument("--quiet", action="store_true", help="do not print the table")
    return ap


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    t0 = time.perf_counter()
    res = run(args)
    res["elapsed_sec"] = round(time.perf_counter() - t0, 3)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        print_report(res)
        if args.out:
            print(f"wrote {args.out}")
    return res


if __name__ == "__main__":
    main()
