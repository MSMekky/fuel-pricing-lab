import numpy as np
import pandas as pd

from fuellab.fe import feols, demean


def _panel(seed=0, n=40, t=15):
    rng = np.random.default_rng(seed)
    i = np.repeat(np.arange(n), t)
    d = np.tile(np.arange(t), n)
    x = rng.normal(size=n * t) + 0.3 * (i % 3)
    y = 1.5 * x + rng.normal(size=n)[i] + rng.normal(size=t)[d] + rng.normal(size=n * t)
    return pd.DataFrame({"i": i, "d": d, "x": x, "y": y})


def test_matches_dummy_variable_regression():
    df = _panel()
    f = feols(df, "y", ["x"], ["i", "d"], "i")
    D = pd.get_dummies(df[["i", "d"]].astype(str), drop_first=False).to_numpy(float)
    X = np.column_stack([df["x"].to_numpy(), D])
    b, *_ = np.linalg.lstsq(X, df["y"].to_numpy(), rcond=None)
    assert np.isclose(f.coef[0], b[0], atol=1e-8)
    # cluster-robust SE by hand from the dummy regression
    u = df["y"].to_numpy() - X @ b
    rank = np.linalg.matrix_rank(X)
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = sum(np.outer(X[g].T @ u[g], X[g].T @ u[g]) for g in [df["i"].to_numpy() == k for k in range(40)])
    V = XtX_inv @ meat @ XtX_inv
    n, G = len(df), 40
    se_lsdv = np.sqrt(V[0, 0] * G / (G - 1) * (n - 1) / (n - rank))
    # same sandwich for x (Frisch-Waugh-Lovell); only the degrees-of-freedom factor differs:
    # the within estimator counts K = 1, the dummy regression counts every dummy.
    rescale = np.sqrt((n - 1) / (n - rank)) / np.sqrt((n - 1) / (n - 1))
    assert np.isclose(f.se[0] * rescale, se_lsdv, rtol=1e-6)


def test_demean_removes_both_effects():
    df = _panel(1)
    Z = demean(df["y"].to_numpy(), [df["i"].to_numpy(), df["d"].to_numpy()])[:, 0]
    assert np.abs(pd.Series(Z).groupby(df["i"]).mean()).max() < 1e-8
    assert np.abs(pd.Series(Z).groupby(df["d"]).mean()).max() < 1e-8
