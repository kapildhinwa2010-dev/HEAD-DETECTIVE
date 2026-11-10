"""Classify attention-head roles from attention-pattern statistics.

The core is pure NumPy and works on any square (T, T) attention
matrix, no matter which framework produced it.

Rows are queries, columns are keys. Row i is the distribution of
attention from position i.
"""

import numpy as np


def head_features(pattern):
    """Return role-relevant statistics for one attention head."""
    p = np.asarray(pattern, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("pattern must be a square (T, T) matrix")
    t = p.shape[0]
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    idx = np.arange(t)
    diag_mass = float(np.mean(p[idx, idx]))
    if t > 1:
        prev_mass = float(np.mean(p[idx[1:], idx[1:] - 1]))
    else:
        prev_mass = 0.0
    band = np.abs(idx[:, None] - idx[None, :]) <= 2
    band_mass = float(np.mean(p[band]))
    far = np.abs(idx[:, None] - idx[None, :]) > max(2, t // 4)
    far_mass = float(np.sum(p[far]) / t) if far.any() else 0.0
    safe = np.clip(p, 1e-12, None)
    row_entropy = -np.sum(safe * np.log(safe), axis=1)
    entropy = float(np.mean(row_entropy) / np.log(max(t, 2)))
    col_sums = p.sum(axis=0)
    max_col = int(np.argmax(col_sums))
    sink = float(np.mean(p[:, max_col]))
    return {
        "t": t,
        "diag_mass": diag_mass,
        "prev_token_mass": prev_mass,
        "band_mass": band_mass,
        "long_range_mass": far_mass,
        "entropy": entropy,
        "max_column": max_col,
        "max_column_mass": sink,
    }