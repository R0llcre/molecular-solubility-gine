"""Tests for the resume fingerprint utility (pure-Python; no pandas/torch)."""

from molfuse_gine.utils import (
    file_sha256,
    protocol_fingerprint,
    resume_is_valid,
    write_fingerprint,
)


def test_fingerprint_is_order_independent():
    a = protocol_fingerprint({"experts": "x,y", "folds": 5, "seeds": "1,2"})
    b = protocol_fingerprint({"seeds": "1,2", "folds": 5, "experts": "x,y"})
    assert a == b


def test_fingerprint_changes_with_params():
    base = {"experts": "x,y", "folds": 5}
    assert protocol_fingerprint(base) != protocol_fingerprint({**base, "folds": 10})
    assert protocol_fingerprint(base) != protocol_fingerprint({**base, "experts": "x,z"})


def test_resume_is_valid_only_for_matching_params(tmp_path):
    fp = tmp_path / "run_fingerprint.json"
    params = {"stage": "tabular_oof_stack", "experts": "a:ridge", "folds": 5}

    # No fingerprint yet -> not safe to resume.
    assert resume_is_valid(fp, params) is False

    write_fingerprint(fp, params)
    assert resume_is_valid(fp, params) is True

    # Any parameter change invalidates the cached run.
    assert resume_is_valid(fp, {**params, "folds": 3}) is False
    assert resume_is_valid(fp, {**params, "experts": "b:xgb"}) is False


def test_file_sha256_tracks_content(tmp_path):
    assert file_sha256(tmp_path / "nope.yaml") == "missing"

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("train_ratio: 0.7\n", encoding="utf-8")
    first = file_sha256(cfg)
    assert first != "missing" and len(first) == 64

    cfg.write_text("train_ratio: 0.8\n", encoding="utf-8")
    assert file_sha256(cfg) != first


def test_corrupt_fingerprint_file_is_not_resumable(tmp_path):
    fp = tmp_path / "run_fingerprint.json"
    fp.write_text("{ not json", encoding="utf-8")
    assert resume_is_valid(fp, {"any": "params"}) is False
