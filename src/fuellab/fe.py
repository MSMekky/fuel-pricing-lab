"""Fixed-effects OLS with cluster-robust standard errors, in numpy.

    y_it = X_it b + a_i + d_t (+ further absorbed effects) + e_it

The absorbed effects are removed by alternating projections (demean by each factor in turn until the
change is below `tol`), which is exact at convergence (Frisch-Waugh-Lovell) and scales to millions of rows.
Standard errors are CRV1, clustered on one factor, with the usual small-sample factor
G/(G-1) * (N-1)/(N-K). Tested against a dummy-variable regression in tests/test_fe.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FitResult:
    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    vcov: np.ndarray
    n_obs: int
    n_clusters: int

    def table(self) -> pd.DataFrame:
        t = pd.DataFrame({"coef": self.coef, "se": self.se}, index=self.names)
        t["ci_low"] = t["coef"] - 1.96 * t["se"]
        t["ci_high"] = t["coef"] + 1.96 * t["se"]
        return t


def _codes(s: pd.Series) -> tuple[np.ndarray, int]:
    c, u = pd.factorize(s, sort=False)
    return c, len(u)


def demean(M: np.ndarray, factors: list[np.ndarray], tol: float = 1e-10, max_iter: int = 1000) -> np.ndarray:
    M = M.astype(float).copy()
    if M.ndim == 1:
        M = M[:, None]
    meta = []
    for f in factors:
        c, k = _codes(pd.Series(f))
        meta.append((c, k, np.bincount(c, minlength=k).astype(float)))
    for _ in range(max_iter):
        delta = 0.0
        for c, k, cnt in meta:
            means = np.vstack([np.bincount(c, weights=M[:, j], minlength=k) for j in range(M.shape[1])]).T / cnt[:, None]
            M -= means[c]
            delta = max(delta, float(np.abs(means).max()))
        if delta < tol:
            break
    return M


def feols(df: pd.DataFrame, y: str, x: list[str], absorb: list[str], cluster: str) -> FitResult:
    d = df.dropna(subset=[y, *x])
    factors = [d[a].to_numpy() for a in absorb]
    Z = demean(d[[y, *x]].to_numpy(dtype=float), factors)
    yt, Xt = Z[:, 0], Z[:, 1:]
    XtX = Xt.T @ Xt
    XtX_inv = np.linalg.pinv(XtX)
    b = XtX_inv @ (Xt.T @ yt)
    u = yt - Xt @ b
    g, G = _codes(d[cluster])
    scores = np.vstack([np.bincount(g, weights=Xt[:, j] * u, minlength=G) for j in range(Xt.shape[1])]).T
    meat = scores.T @ scores
    n, k = Xt.shape
    adj = G / (G - 1) * (n - 1) / (n - k)
    V = adj * XtX_inv @ meat @ XtX_inv
    return FitResult(list(x), b, np.sqrt(np.diag(V)), V, n, G)
