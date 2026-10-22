import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_compare.py"
_spec = importlib.util.spec_from_file_location("bootstrap_compare", _SCRIPT)
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)


N_BOOT = 2000


def _write_root(root: Path, mode: str, frames: dict) -> None:
    """frames: seed -> DataFrame(source_index, y_true, mu); writes the run_calibration layout."""
    (root / "summaries").mkdir(parents=True, exist_ok=True)
    for seed, df in frames.items():
        run_dir = root / "runs" / f"{mode}_seed{seed}_x"
        run_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(run_dir / "predictions_test.csv", index=False)
        rmse = float(np.sqrt(np.mean((df["y_true"] - df["mu"]) ** 2)))
        summary = {"mode": mode, "seed": seed, "run_dir": str(run_dir), "test": {"rmse": rmse}}
        (root / "summaries" / f"{mode}_seed{seed}.json").write_text(json.dumps(summary))


def _synthetic(shared: bool, n: int = 120, n_seeds: int = 3, n_pool: int = 400, seed: int = 0):
    """Return seed -> (ids, y, mu_a, err_a). Shared: same molecules every seed."""
    rng = np.random.default_rng(seed)
    y_pool = rng.normal(-3.0, 2.0, size=n_pool)
    out = {}
    fixed_ids = np.sort(rng.choice(n_pool, size=n, replace=False))
    for s in range(1, n_seeds + 1):
        ids = fixed_ids if shared else np.sort(rng.choice(n_pool, size=n, replace=False))
        y = y_pool[ids]
        err_a = rng.normal(0.0, 0.6, size=n)
        out[s] = (ids, y, y + err_a, err_a)
    return out


def _frames(data, which: str, worse_factor: float = 1.5, shuffle_seed=None):
    frames = {}
    for s, (ids, y, mu_a, err_a) in data.items():
        if which == "a":
            mu = mu_a
        elif which == "same":
            mu = mu_a.copy()
        else:  # strictly worse on every molecule: |err_B| > |err_A|
            mu = y + worse_factor * err_a + np.sign(err_a + 1e-12) * 0.05
        df = pd.DataFrame({"source_index": ids, "y_true": y, "mu": mu})
        if shuffle_seed is not None:
            df = df.sample(frac=1.0, random_state=shuffle_seed + s).reset_index(drop=True)
        frames[s] = df
    return frames


def _run_cli(tmp_path, a_frames, b_frames, extra=()):
    _write_root(tmp_path / "A", "m", a_frames)
    _write_root(tmp_path / "B", "m", b_frames)
    seeds = ",".join(str(s) for s in a_frames)
    out = tmp_path / "out.json"
    res = bc.main([
        "--a", f"{tmp_path / 'A'}:m", "--b", f"{tmp_path / 'B'}:m", "--seeds", seeds,
        "--n-boot", str(N_BOOT), "--rng", "123", "--out", str(out), "--quiet", *extra,
    ])
    on_disk = json.loads(out.read_text())
    assert on_disk["pooled"] == pytest.approx(res["pooled"])
    return res


@pytest.mark.parametrize("shared", [True, False])
def test_identical_predictions_give_zero_diff(tmp_path, shared):
    data = _synthetic(shared)
    res = _run_cli(tmp_path, _frames(data, "a"), _frames(data, "same"))
    assert res["pooling"] == ("shared_test_set" if shared else "per_seed_test_sets")
    for row in res["per_seed"] + [res["pooled"]]:
        assert row["diff"] == 0.0
        assert row["ci_low"] <= 0.0 <= row["ci_high"]
        assert row["p_diff_lt_0"] == 0.0
    assert res["warnings"] == []


@pytest.mark.parametrize("shared", [True, False])
def test_b_strictly_worse_gives_ci_below_zero(tmp_path, shared):
    data = _synthetic(shared)
    res = _run_cli(tmp_path, _frames(data, "a"), _frames(data, "worse"))
    for row in res["per_seed"]:
        assert row["rmse_a"] < row["rmse_b"]
        assert row["ci_high"] < 0.0
        assert row["p_diff_lt_0"] == 1.0
    p = res["pooled"]
    assert p["diff"] < 0.0 and p["ci_high"] < 0.0 and p["p_diff_lt_0"] == 1.0
    assert p["n_seeds_a_lower"] == len(data)
    # sanity: point estimate inside its CI
    assert p["ci_low"] <= p["diff"] <= p["ci_high"]


def test_row_order_does_not_matter(tmp_path):
    data = _synthetic(shared=False)
    r1 = _run_cli(tmp_path / "1", _frames(data, "a"), _frames(data, "worse"))
    r2 = _run_cli(tmp_path / "2", _frames(data, "a"), _frames(data, "worse", shuffle_seed=7))
    assert r1["pooled"]["diff"] == pytest.approx(r2["pooled"]["diff"])
    assert r1["pooled"]["ci_low"] == pytest.approx(r2["pooled"]["ci_low"])
    assert r1["pooled"]["ci_high"] == pytest.approx(r2["pooled"]["ci_high"])


def test_deterministic_given_rng(tmp_path):
    data = _synthetic(shared=True)
    r1 = _run_cli(tmp_path / "1", _frames(data, "a"), _frames(data, "worse"))
    r2 = _run_cli(tmp_path / "2", _frames(data, "a"), _frames(data, "worse"))
    assert r1["pooled"] == r2["pooled"]
    keys = ["diff", "ci_low", "ci_high", "p_diff_lt_0"]
    rows1 = [[r[k] for k in keys] for r in r1["per_seed"]]
    rows2 = [[r[k] for k in keys] for r in r2["per_seed"]]
    assert rows1 == rows2


def test_mismatched_molecules_raise(tmp_path):
    data = _synthetic(shared=True)
    a = _frames(data, "a")
    b = _frames(data, "worse")
    b[2] = b[2].iloc[1:].reset_index(drop=True)
    with pytest.raises(ValueError, match="molecule sets differ"):
        _run_cli(tmp_path, a, b)


def test_mismatched_y_true_raise(tmp_path):
    data = _synthetic(shared=True)
    a = _frames(data, "a")
    b = _frames(data, "worse")
    b[1].loc[3, "y_true"] += 0.5
    with pytest.raises(ValueError, match="y_true differs"):
        _run_cli(tmp_path, a, b)


def _pairs(data, which="worse"):
    pairs = []
    a, b = _frames(data, "a"), _frames(data, which)
    for s in data:
        al = bc.align_pair(a[s], b[s])
        al["seed"] = s
        pairs.append(al)
    return pairs


def test_pooled_replicates_are_mean_of_per_seed_replicates():
    for shared in (True, False):
        res = bc.compare(_pairs(_synthetic(shared)), n_boot=500, rng_seed=1, return_replicates=True)
        reps = res["_replicates"]
        np.testing.assert_allclose(reps["pooled"], reps["per_seed"].mean(axis=0))
        diffs = [r["diff"] for r in res["per_seed"]]
        assert res["pooled"]["diff"] == pytest.approx(np.mean(diffs))


def test_shared_pooling_does_not_treat_seeds_as_independent():
    # Five seeds with the SAME predictions on the same molecules carry no more information
    # about molecule sampling than one seed: the shared-resample pooled CI must equal the
    # single-seed CI, whereas forcing independent resampling would shrink it ~sqrt(5)x.
    data1 = _synthetic(shared=True, n_seeds=1)
    data5 = {s: data1[1] for s in range(1, 6)}
    pairs5 = _pairs(data5)
    shared = bc.compare(pairs5, n_boot=4000, rng_seed=3)
    single = bc.compare(pairs5[:1], n_boot=4000, rng_seed=3)
    indep = bc.compare(pairs5, n_boot=4000, rng_seed=3, pooling="independent")
    assert shared["pooling"] == "shared_test_set"
    assert shared["pooled"]["ci_low"] == pytest.approx(single["pooled"]["ci_low"])
    assert shared["pooled"]["ci_high"] == pytest.approx(single["pooled"]["ci_high"])
    w_shared = shared["pooled"]["ci_high"] - shared["pooled"]["ci_low"]
    w_indep = indep["pooled"]["ci_high"] - indep["pooled"]["ci_low"]
    assert w_indep < 0.7 * w_shared


def test_forced_shared_pooling_requires_identical_sets():
    with pytest.raises(ValueError, match="different test molecule sets"):
        bc.compare(_pairs(_synthetic(shared=False)), n_boot=100, pooling="shared")


def test_bootstrap_matches_direct_resampling():
    # boot_rmse_diff (chunked) must agree with a naive per-replicate computation.
    rng = np.random.default_rng(5)
    y = rng.normal(size=50)
    mu_a, mu_b = y + rng.normal(0, 0.5, 50), y + rng.normal(0, 0.7, 50)
    idx = rng.integers(0, 50, size=(37, 50))
    got = bc.boot_rmse_diff((y - mu_a) ** 2, (y - mu_b) ** 2, idx, chunk=10)
    want = [bc.rmse(y[i], mu_a[i]) - bc.rmse(y[i], mu_b[i]) for i in idx]
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_parse_spec_uses_last_colon(tmp_path):
    root = tmp_path / "with:colon"
    root.mkdir()
    path, mode = bc.parse_spec(f"{root}:beta_nll")
    assert path == root.resolve() and mode == "beta_nll"
    with pytest.raises(ValueError):
        bc.parse_spec(str(root) + ":")
