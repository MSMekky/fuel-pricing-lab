"""The questions, each answered from the station-day panel (see daily.py).

1. mechanics      Did stations comply, and how did the daily price cycle change?
2. did            Did the average price paid change? Germany vs France, station and day fixed effects.
3. event_study    Week-by-week version of (2): shows pre-trends and how the effect evolves.
4. placebo        The same estimate at fake rule dates before April: how big are gaps that are not the rule?
5. competition    Within Germany only: did stations with many rivals nearby react differently?
6. timing         For drivers: what does the hour of filling up cost, before and after?

All prices are reported in euro cents per litre.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RULE_START, WAR_SHOCK, COMPETITION_RADIUS_KM
from .daily import HOUR_COLS
from .fe import feols

PERIODS = ("calm", "shock", "rule")   # before 28 Feb, 28 Feb to 31 Mar, from 1 Apr


def period(day: pd.Series) -> pd.Series:
    return pd.Series(np.select([day < WAR_SHOCK, day < RULE_START], ["calm", "shock"], "rule"), index=day.index)


def build_panel(daily_de: pd.DataFrame, daily_fr: pd.DataFrame, min_cover_h: float = 20.0) -> pd.DataFrame:
    """Stack the two countries. Station-days with less than `min_cover_h` hours of known price are dropped."""
    p = pd.concat([daily_de.assign(country="DE"), daily_fr.assign(country="FR")], ignore_index=True)
    p = p[p["covered_h"] >= min_cover_h].copy()
    p["day"] = pd.to_datetime(p["day"])
    p["period"] = period(p["day"])
    p["post"] = (p["day"] >= RULE_START).astype(int)
    p["de"] = (p["country"] == "DE").astype(int)
    p["treat"] = p["de"] * p["post"]
    p["price_ct"] = p["mean_tw"].astype(float) * 100
    p["spread_ct"] = (p["p_max"] - p["p_min"]).astype(float) * 100
    p["week"] = ((p["day"] - RULE_START).dt.days // 7).astype(int)
    return p


# ------------------------------------------------------------------ 1. mechanics
def mechanics(p: pd.DataFrame) -> pd.DataFrame:
    g = p.groupby(["country", "period"])
    out = pd.DataFrame({
        "station_days": g.size(),
        "increases_per_day": g["n_up"].mean(),
        "cuts_per_day": g["n_down"].mean(),
        "share_of_increases_at_noon": g["n_up_noon"].sum() / g["n_up"].sum().clip(lower=1),
        "noon_jump_ct": g.apply(lambda d: d.loc[d["n_up_noon"] > 0, "up_noon_ct"].mean(), include_groups=False),
        "intraday_spread_ct": g["spread_ct"].mean(),
        "stations_with_off_noon_increase": g.apply(lambda d: (d["n_up"] > d["n_up_noon"]).mean(), include_groups=False),
    })
    return out.reindex([(c, s) for c in ("DE", "FR") for s in PERIODS]).dropna(how="all")


def profile(p: pd.DataFrame, country: str = "DE") -> pd.DataFrame:
    """Average price at each hour minus the station-day mean, by period (cents)."""
    d = p[p["country"] == country]
    rel = d[HOUR_COLS].astype(float).sub(d["mean_tw"].astype(float), axis=0) * 100
    rel["period"] = d["period"].to_numpy()
    out = rel.groupby("period")[HOUR_COLS].mean().T
    out.index = range(24)
    out.index.name = "hour"
    return out.reindex(columns=[c for c in PERIODS if c in out.columns])


# ------------------------------------------------------------------ 2-4. level effect
def did(p: pd.DataFrame, pre_periods=("calm", "shock"), y: str = "price_ct") -> dict:
    d = p[p["period"].isin([*pre_periods, "rule"])]
    f = feols(d, y, ["treat"], ["station", "day"], "station")
    t = f.table().loc["treat"]
    return {"pre": "+".join(pre_periods), "y": y, "coef": float(t["coef"]), "se": float(t["se"]),
            "ci_low": float(t["ci_low"]), "ci_high": float(t["ci_high"]), "n_obs": f.n_obs, "stations": f.n_clusters}


def event_study(p: pd.DataFrame, ref_week: int = -1, y: str = "price_ct", lo: int = -12, hi: int = 25) -> pd.DataFrame:
    d = p[(p["week"] >= lo) & (p["week"] <= hi)].copy()
    weeks = [w for w in range(lo, hi + 1) if w != ref_week and (d["week"] == w).any()]
    cols = []
    for w in weeks:
        c = f"w{w}"
        d[c] = ((d["week"] == w) & (d["de"] == 1)).astype(float)
        cols.append(c)
    f = feols(d, y, cols, ["station", "day"], "station")
    t = f.table()
    t["week"] = weeks
    ref = pd.DataFrame({"coef": [0.0], "se": [0.0], "ci_low": [0.0], "ci_high": [0.0], "week": [ref_week]})
    return pd.concat([t.reset_index(drop=True), ref]).sort_values("week").reset_index(drop=True)


def placebo(p: pd.DataFrame, dates: list[str], window_days: int = 28, y: str = "price_ct") -> pd.DataFrame:
    """For each date, estimate the DE x after-date effect using `window_days` either side, data before the real rule only
    (and the real date, for comparison). Gaps of the placebo size occur without any rule."""
    rows = []
    for ds in [*dates, str(RULE_START.date())]:
        t0 = pd.Timestamp(ds)
        d = p[(p["day"] >= t0 - pd.Timedelta(days=window_days)) & (p["day"] < t0 + pd.Timedelta(days=window_days))].copy()
        if t0 < RULE_START:
            d = d[d["day"] < RULE_START]
        d["fake"] = ((d["day"] >= t0) & (d["de"] == 1)).astype(float)
        f = feols(d, y, ["fake"], ["station", "day"], "station")
        rows.append({"date": ds, "real": t0 == RULE_START, "coef": float(f.coef[0]), "se": float(f.se[0])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 5. competition (Germany only)
def rivals_within(stations: pd.DataFrame, radius_km: float = COMPETITION_RADIUS_KM) -> pd.Series:
    """Number of other stations within radius, by a grid search (exact haversine within neighbouring cells)."""
    lat = np.radians(stations["latitude"].to_numpy(float))
    lon = np.radians(stations["longitude"].to_numpy(float))
    cell = radius_km / 111.0
    gx = np.floor(stations["latitude"].to_numpy(float) / cell).astype(int)
    gy = np.floor(stations["longitude"].to_numpy(float) / (cell / np.cos(np.radians(51)))).astype(int)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, k in enumerate(zip(gx, gy)):
        buckets.setdefault(k, []).append(i)
    count = np.zeros(len(stations), int)
    for i in range(len(stations)):
        cand = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in buckets.get((gx[i] + dx, gy[i] + dy), [])]
        cand = np.array([j for j in cand if j != i], int)
        if len(cand) == 0:
            continue
        a = np.sin((lat[cand] - lat[i]) / 2) ** 2 + np.cos(lat[i]) * np.cos(lat[cand]) * np.sin((lon[cand] - lon[i]) / 2) ** 2
        count[i] = int((2 * 6371.0 * np.arcsin(np.sqrt(a)) <= radius_km).sum())
    return pd.Series(count, index=stations["station"].to_numpy(), name="rivals")


MAJORS = {"ARAL", "SHELL", "ESSO", "TOTALENERGIES", "TOTAL", "JET", "AVIA", "STAR", "AGIP", "ENI", "HEM", "OMV", "BFT"}


def competition(p: pd.DataFrame, stations: pd.DataFrame, y: str = "price_ct") -> pd.DataFrame:
    """Germany only. post x (many rivals) and post x (independent) with station and day effects.
    Country-wide shocks are absorbed by the day effects, so this does not lean on France at all."""
    st = stations.copy()
    st["rivals"] = rivals_within(st).to_numpy()
    st["brand_u"] = st["brand"].fillna("").str.upper().str.strip()
    st["independent"] = (~st["brand_u"].isin(MAJORS)).astype(int)
    d = p[p["country"] == "DE"].merge(st[["station", "rivals", "independent"]], on="station", how="inner")
    med = d.drop_duplicates("station")["rivals"].median()
    d["dense"] = (d["rivals"] > med).astype(int)
    rows = []
    for col in ("dense", "independent"):
        d["x"] = d["post"] * d[col]
        for yy in (y, "spread_ct"):
            f = feols(d, yy, ["x"], ["station", "day"], "station")
            rows.append({"split": col, "y": yy, "coef": float(f.coef[0]), "se": float(f.se[0]),
                         "share_of_stations": float(d.drop_duplicates("station")[col].mean()),
                         "median_rivals": float(med)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 6. timing for drivers
def timing(p: pd.DataFrame, country: str = "DE", hours=range(6, 23), litres_per_fill: float = 50.0,
           fills_per_year: int = 30, B: int = 500, seed: int = 7) -> pd.DataFrame:
    """Per period: cost of filling at the cheapest vs the dearest hour and vs a random hour (6:30 to 22:30).
    Uncertainty: bootstrap over calendar weeks."""
    rng = np.random.default_rng(seed)
    d = p[p["country"] == country]
    cols = [HOUR_COLS[h] for h in hours]
    rel = d[cols].astype(float).sub(d[cols].astype(float).mean(axis=1), axis=0) * 100
    rel["period"], rel["wk"] = d["period"].to_numpy(), d["day"].dt.to_period("W").astype(str).to_numpy()
    out = []
    for per, g in rel.groupby("period"):
        wk_means = g.groupby("wk")[cols].mean()
        prof = wk_means.mean()
        best, worst = prof.idxmin(), prof.idxmax()
        boots = []
        for _ in range(B):
            s = wk_means.iloc[rng.integers(0, len(wk_means), len(wk_means))].mean()
            boots.append((s[worst] - s[best]))
        boots = np.array(boots)
        gap = prof[worst] - prof[best]
        out.append({"period": per, "cheapest_hour": int(best[1:]), "dearest_hour": int(worst[1:]),
                    "dearest_minus_cheapest_ct": float(gap), "ci_low": float(np.quantile(boots, 0.05)),
                    "ci_high": float(np.quantile(boots, 0.95)),
                    "random_minus_cheapest_ct": float(prof.mean() - prof[best]),
                    "eur_per_year_cheapest_vs_dearest": float(gap / 100 * litres_per_fill * fills_per_year)})
    return pd.DataFrame(out).set_index("period").reindex([c for c in PERIODS if c in set(rel["period"])])
