import numpy as np
import pytest

from head_detective.detective import (
    classify_head,
    detective_report,
    head_features,
    summarize,
)

T = 16


def local_pattern():
    p = np.eye(T)
    return p


def prev_pattern():
    p = np.zeros((T, T))
    for i in range(1, T):
        p[i, i - 1] = 1.0
    p[0, 0] = 1.0
    return p


def diffuse_pattern():
    return np.ones((T, T))


def sink_pattern():
    p = np.zeros((T, T))
    p[:, 3] = 1.0
    return p


def long_pattern():
    p = np.zeros((T, T))
    for i in range(T):
        if i >= 5:
            p[i, i - 5] = 1.0
        else:
            p[i, 0] = 1.0
    return p


def test_local_head():
    r = classify_head(local_pattern())
    assert r["role"] == "local"
    assert r["confidence"] > 0.99


def test_previous_token_head():
    r = classify_head(prev_pattern())
    assert r["role"] == "previous-token"
    assert r["confidence"] > 0.99


def test_diffuse_head():
    r = classify_head(diffuse_pattern())
    assert r["role"] == "diffuse"
    assert r["confidence"] > 0.99


def test_sink_head():
    r = classify_head(sink_pattern())
    assert r["role"] == "sink"
    assert r["features"]["max_column"] == 3


def test_long_range_head():
    r = classify_head(long_pattern())
    assert r["role"] == "long-range"


def test_features_keys():
    f = head_features(prev_pattern())
    for key in ["t", "diag_mass", "prev_token_mass", "band_mass",
                "long_range_mass", "entropy", "max_column", "max_column_mass"]:
        assert key in f
    assert f["t"] == T


def test_report_shapes():
    single = detective_report(local_pattern())
    assert len(single) == 1
    stack = detective_report(np.stack([local_pattern(), prev_pattern()]))
    assert len(stack) == 2
    batched = detective_report(np.stack([np.stack([local_pattern(), prev_pattern()])]))
    assert len(batched) == 2
    counts = summarize(stack)
    assert counts == {"local": 1, "previous-token": 1}


def test_bad_input():
    with pytest.raises(ValueError):
        head_features(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        detective_report(np.zeros((2, 2, 2, 2, 2)))


def test_tiny_sequence_no_crash():
    r = classify_head(np.ones((1, 1)))
    assert r["role"] in ("local", "sink", "mixed")