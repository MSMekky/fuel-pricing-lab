"""A synthetic fuel market with a known answer, used to test every estimator before it touches real data.

Germany before the rule: several increases a day (06:00, 13:00, 17:00, 22:00) with cuts in between.
Germany after the rule: one increase at 12:00, cuts through the afternoon and evening, flat overnight.
France: one price per day, no intraday cycle.

Each intraday shape has a time-weighted mean of zero, so a station's daily mean price is exactly its level:
    level_it = cost_t + country markup + station effect + delta * 1[Germany, after rule] + noise
and a correct difference-in-differences must recover `delta`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RULE_START, WAR_SHOCK

PRE_SHAPE = np.array([0, 0, 0, 0, 0, 0, 6, 5, 4, 3, 2, 1, 0, 4, 3, 2, 1, 4, 3, 2, 1, 0, 5, 4], float)
POST_SHAPE = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, 9, 8, 6, 5, 4, 3, 2, 2, 1, 1, 1, 1], float)
FR_SHAPE = np.zeros(24)


def _centre(shape: np.ndarray, scale_ct: float) -> np.ndarray:
    s = shape - shape.mean()
    return s / max(np.abs(s).max(), 1e-9) * scale_ct / 100.0


@dataclass
class Market:
    events: pd.DataFrame       # station, ts, price
    stations: pd.DataFrame     # station, country, latitude, longitude, brand
    delta: float               # true level effect of the rule on German prices (euro per litre)


def make_market(n_de: int = 120, n_fr: int = 80, start="2026-01-05", end="2026-06-28", delta: float = 0.02,
                pre_amp_ct: float = 6.0, post_amp_ct: float = 9.0, noise_ct: float = 0.5, seed: int = 0) -> Market:
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    D = len(days)
    shock = (days >= WAR_SHOCK).astype(float) * 0.35                       # common oil shock, both countries
    cost = 1.55 + np.cumsum(rng.normal(0, 0.004, D)) + shock
    post = days >= RULE_START

    ids = [f"DE{i:04d}" for i in range(n_de)] + [f"FR{i:04d}" for i in range(n_fr)]
    country = np.array(["DE"] * n_de + ["FR"] * n_fr)
    st_fe = rng.normal(0, 0.03, len(ids))
    markup = np.where(country == "DE", 0.12, 0.05)
    lat = np.where(country == "DE", rng.uniform(52.3, 52.7, len(ids)), rng.uniform(48.7, 49.0, len(ids)))
    lon = np.where(country == "DE", rng.uniform(13.1, 13.7, len(ids)), rng.uniform(2.1, 2.6, len(ids)))
    brands = rng.choice(["ARAL", "Shell", "JET", "ESSO", "TotalEnergies", "free"], len(ids))
    stations = pd.DataFrame({"station": ids, "country": country, "latitude": lat, "longitude": lon, "brand": brands})

    level = (cost[None, :] + markup[:, None] + st_fe[:, None]
             + delta * ((country == "DE")[:, None] & post[None, :])
             + rng.normal(0, noise_ct / 100.0, (len(ids), D)))
    pre, pst, fr = _centre(PRE_SHAPE, pre_amp_ct), _centre(POST_SHAPE, post_amp_ct), FR_SHAPE
    rows = []
    for s, sid in enumerate(ids):
        for d, day in enumerate(days):
            shape = fr if country[s] == "FR" else (pst if post[d] else pre)
            hourly = np.round(level[s, d] + shape, 3)
            ts = day + pd.to_timedelta(np.arange(24), unit="h") + pd.to_timedelta(rng.integers(0, 120, 24), unit="s")
            rows.append(pd.DataFrame({"station": sid, "ts": ts, "price": hourly}))
    ev = pd.concat(rows, ignore_index=True)
    # keep only reports that change the price (plus each station's first), like the real feeds
    chg = ev.groupby("station")["price"].diff().fillna(1.0).abs() > 1e-9
    ev = ev[chg].sort_values(["station", "ts"]).reset_index(drop=True)
    return Market(ev, stations, delta)
