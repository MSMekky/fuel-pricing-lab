"""Part 1: TotalEnergies' voluntary price cap in France, 2026.

From March 2026 TotalEnergies capped pump prices at its ~3,300 stations in mainland France (petrol 1.99 EUR/L;
diesel 1.99, then 2.09 from 13 March, then 2.25 from early April), switched most of it off on 30 June and
back on from 22 July. Competitors had no cap. The questions:

    gap_study        How much cheaper did Total become, week by week, relative to its normal premium?
    stockouts        Did the cap empty Total's pumps, and does that scale with how far the market sits above the cap?
    spillover        Did rivals close to a Total station price differently from rivals far away?
    driver_mc        For a driver: is the cheaper but possibly empty Total pump worth the trip?
    cost_mc          What did the cap cost TotalEnergies, under stated volume assumptions?

Which station is a Total station is taken from OpenStreetMap (brand tags), never from prices, so the
treatment group is defined independently of the outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fe import feols

PRE_END = pd.Timestamp("2026-02-23")           # calm period: weeks starting before this (no shock, no cap)
REF_WEEK = pd.Timestamp("2026-02-16")          # reference week: the last full week before the oil shock and any cap

# Cap level by fuel and date, from TotalEnergies press releases (12 Mar, 31 Mar, 27 May, 30 Jun, 22 Jul 2026).
# The first diesel cap at 1.99 predates the 12 March release; its start is dated from the data (see first_stage).
CAP_CALENDAR = {
    "diesel": [("2026-03-03", "2026-03-12", 1.99), ("2026-03-13", "2026-04-07", 2.09),
               ("2026-04-08", "2026-06-29", 2.25), ("2026-07-22", "2026-12-31", 2.25)],
    "e10": [("2026-03-13", "2026-06-29", 1.99), ("2026-07-22", "2026-12-31", 1.99)],
}


def cap_level(fuel: str, days: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=days.index)
    for a, b, lvl in CAP_CALENDAR[fuel]:
        out[(days >= a) & (days <= b)] = lvl
    return out


# ------------------------------------------------------------------ data
def nearest_total_km(stations: pd.DataFrame) -> pd.Series:
    """Distance from every station to the nearest *other* Total station (haversine, km)."""
    from sklearn.neighbors import BallTree
    tot = stations[stations["brand_group"] == "total"]
    tree = BallTree(np.radians(tot[["latitude", "longitude"]].to_numpy(float)), metric="haversine")
    dist, idx = tree.query(np.radians(stations[["latitude", "longitude"]].to_numpy(float)), k=2)
    self_hit = tot["station"].to_numpy()[idx[:, 0]] == stations["station"].to_numpy()
    d = np.where(self_hit, dist[:, 1], dist[:, 0]) * 6371.0
    return pd.Series(d, index=stations.index)


def build(daily: pd.DataFrame, stations: pd.DataFrame, brands: pd.DataFrame, stockouts: pd.DataFrame,
          fuel: str, min_cover_h: float = 20.0) -> pd.DataFrame:
    """Station-day panel for one fuel, mainland France, stations with a known brand group."""
    st = stations.merge(brands, on="station", how="inner")
    st = st[st["latitude"].between(41.0, 51.5) & st["longitude"].between(-5.5, 10.0)].copy()   # mainland + Corsica
    st = st[st["brand_group"] != "unknown"].reset_index(drop=True)
    st["near_total_km"] = nearest_total_km(st)
    d = daily[daily["covered_h"] >= min_cover_h].merge(
        st[["station", "brand_group", "near_total_km", "road"]], on="station", how="inner")
    d["day"] = pd.to_datetime(d["day"])
    d = d.merge(stockouts, on=["station", "day"], how="left").fillna({"stockout_h": 0.0})
    d["total"] = (d["brand_group"] == "total").astype(int)
    d["price_ct"] = d["mean_tw"].astype(float) * 100
    d["p12_ct"] = d["p12"].astype(float) * 100
    d["cap"] = cap_level(fuel, d["day"])
    d["week"] = d["day"].dt.to_period("W").dt.start_time
    return d


def weekly(d: pd.DataFrame) -> pd.DataFrame:
    """Station-week means: the event studies run on these (same estimand, 7x fewer rows)."""
    g = d.groupby(["station", "week"], as_index=False).agg(
        price_ct=("price_ct", "mean"), stockout_h=("stockout_h", "mean"), total=("total", "first"),
        brand_group=("brand_group", "first"), near_total_km=("near_total_km", "first"), road=("road", "first"),
        days=("price_ct", "size"))
    return g[g["days"] >= 4]


# ------------------------------------------------------------------ first stage
def first_stage(d: pd.DataFrame) -> pd.DataFrame:
    """Per day: market median (non-Total), Total median, cap, and the share of Total stations priced exactly at a cap
    value at 12:30. A share near zero before March and high afterwards shows the brand tags pick out the capped
    network; the share among non-Total stations shows how much contamination there is."""
    levels = sorted({c for v in CAP_CALENDAR.values() for *_, c in v})
    at_cap = d["p12"].astype(float).round(3).isin(levels)
    g = d.assign(at_cap=at_cap).groupby(["day", "total"])
    t = pd.DataFrame({
        "median_ct": g["price_ct"].median(),
        "share_at_a_cap_value": g["at_cap"].mean(),
        "stockout_h": g["stockout_h"].mean(),
        "stations": g.size(),
    }).unstack("total")
    t.columns = [f"{a}_{'total' if b else 'others'}" for a, b in t.columns]
    t["cap_ct"] = d.groupby("day")["cap"].first() * 100
    t["market_minus_cap_ct"] = t["median_ct_others"] - t["cap_ct"]
    return t


# ------------------------------------------------------------------ event studies
def _event_study(w: pd.DataFrame, treat: pd.Series, y: str) -> pd.DataFrame:
    w = w.copy()
    weeks = sorted(wk for wk in w["week"].unique() if wk != REF_WEEK)
    cols = []
    for k, wk in enumerate(weeks):
        c = f"w{k}"
        w[c] = ((w["week"] == wk) & treat.loc[w.index].astype(bool)).astype(float)
        cols.append(c)
    f = feols(w, y, cols, ["station", "week"], "station")
    t = f.table()
    t.index = pd.Index(weeks, name="week")
    ref = pd.DataFrame({"coef": 0.0, "se": 0.0, "ci_low": 0.0, "ci_high": 0.0}, index=pd.Index([REF_WEEK], name="week"))
    t = pd.concat([t, ref]).sort_index()
    # Re-centre on the whole calm period (5 Jan to 22 Feb) rather than one week, so a single noisy
    # reference week does not shift every coefficient. Standard errors are those of the week contrasts.
    shift = t.loc[t.index < PRE_END, "coef"].mean()
    t[["coef", "ci_low", "ci_high"]] -= shift
    return t


def gap_study(w: pd.DataFrame, y: str = "price_ct") -> pd.DataFrame:
    """Total minus other stations, by week, relative to the gap in the reference week (station and week effects).
    Read the coefficient as: how much cheaper (negative) Total was than its usual position in the market."""
    s = w[w["brand_group"] != "elan"]
    return _event_study(s, s["total"], y)


def spillover(w: pd.DataFrame, near_km: float = 1.0, far_km: float = 5.0, y: str = "price_ct") -> pd.DataFrame:
    """Non-Total stations only: within `near_km` of a Total station vs further than `far_km`."""
    s = w[(w["total"] == 0) & (w["brand_group"] != "elan")]
    s = s[(s["near_total_km"] <= near_km) | (s["near_total_km"] > far_km)]
    return _event_study(s, (s["near_total_km"] <= near_km).astype(int), y)


# ------------------------------------------------------------------ dose-response for stockouts
def stockout_dose(fs: pd.DataFrame, bins=(-np.inf, 0, 5, 10, 15, 20, np.inf)) -> tuple[pd.DataFrame, dict]:
    """Excess stockout hours at Total (Total minus others) against how far the market sits above the cap."""
    t = fs.dropna(subset=["cap_ct"]).copy()
    t["excess_h"] = t["stockout_h_total"] - t["stockout_h_others"]
    t["bin"] = pd.cut(t["market_minus_cap_ct"], bins=list(bins))
    tab = t.groupby("bin", observed=True).agg(days=("excess_h", "size"), excess_stockout_h=("excess_h", "mean"),
                                              mean_gap_ct=("market_minus_cap_ct", "mean"))
    x, yv = t["market_minus_cap_ct"].to_numpy(), t["excess_h"].to_numpy()
    X = np.column_stack([np.ones_like(x), np.clip(x, 0, None)])
    b = np.linalg.lstsq(X, yv, rcond=None)[0]
    # weekly block bootstrap for the slope (days within a week are not independent)
    rng = np.random.default_rng(11)
    wk = t.index.to_period("W").to_numpy()
    uw = np.unique(wk)
    slopes = []
    for _ in range(1000):
        pick = np.concatenate([np.flatnonzero(wk == u) for u in rng.choice(uw, len(uw))])
        slopes.append(np.linalg.lstsq(X[pick], yv[pick], rcond=None)[0][1])
    return tab, {"intercept_h": float(b[0]), "slope_h_per_ct": float(b[1]),
                 "slope_ci": [float(np.quantile(slopes, 0.05)), float(np.quantile(slopes, 0.95))], "days": int(len(t))}


# ------------------------------------------------------------------ simulations
def driver_mc(gap: pd.DataFrame, fs: pd.DataFrame, n: int = 20000, litres: float = 50.0, seed: int = 7) -> pd.DataFrame:
    """Per week with the cap on: expected saving of heading to the Total station instead of the nearest rival.

    saving  = litres x (Total discount drawn from the week's estimate and s.e.)
    risk    = P(pump empty on arrival) = Total's stockout hours / 24 that week (drawn from a Beta around it)
    failure = wasted detour: extra km (1 to 8) x 0.07 L/km x pump price + extra minutes (5 to 25) x value of time
              (10 to 20 EUR/h); the driver then buys at the rival anyway.
    """
    rng = np.random.default_rng(seed)
    rows = []
    fs_w = fs.groupby(fs.index.to_period("W").start_time).agg(
        stockout_h_total=("stockout_h_total", "mean"), price_ct=("median_ct_others", "mean"), cap=("cap_ct", "mean"))
    for wk, r in gap.iterrows():
        if wk not in fs_w.index or np.isnan(fs_w.loc[wk, "cap"]) or r["coef"] >= 0:
            continue
        disc = -rng.normal(r["coef"], r["se"], n) / 100.0                  # EUR per litre cheaper at Total
        p = np.clip(fs_w.loc[wk, "stockout_h_total"] / 24.0, 1e-4, 0.99)
        k = 200.0
        p_draw = rng.beta(p * k, (1 - p) * k, n)
        empty = rng.random(n) < p_draw
        km = rng.uniform(1, 8, n)
        minutes = rng.uniform(5, 25, n)
        vot = rng.uniform(10, 20, n)
        fail_cost = km * 0.07 * fs_w.loc[wk, "price_ct"] / 100 + minutes / 60 * vot
        gain = np.where(empty, -fail_cost, litres * disc)
        rows.append({"week": wk, "discount_ct": float(-r["coef"]), "p_empty": float(p),
                     "expected_gain_eur": float(gain.mean()), "p_gain_positive": float((gain > 0).mean()),
                     "gain_p05": float(np.quantile(gain, 0.05)), "gain_p95": float(np.quantile(gain, 0.95))})
    return pd.DataFrame(rows)


def cost_mc(gaps: dict[str, pd.DataFrame], fss: dict[str, pd.DataFrame], n: int = 20000, seed: int = 7,
            stations: int = 3300, litres_per_station_year=(2.0e6, 5.0e6), diesel_share=(0.65, 0.75)) -> dict:
    """Order-of-magnitude cost of the cap to TotalEnergies: weekly discount x litres sold.

    Litres sold per station are not public: drawn uniformly from `litres_per_station_year`, split by `diesel_share`,
    and reduced by the hours the pump was empty. Petrol volume is scaled from E10 (all petrol grades were capped).
    """
    rng = np.random.default_rng(seed)
    L = rng.uniform(*litres_per_station_year, n)
    sh = rng.uniform(*diesel_share, n)
    total = np.zeros(n)
    by_fuel = {}
    for fuel, gap in gaps.items():
        fs = fss[fuel]
        fs_w = fs.groupby(fs.index.to_period("W").start_time).agg(stock=("stockout_h_total", "mean"),
                                                                  cap=("cap_ct", "mean"))
        acc = np.zeros(n)
        for wk, r in gap.iterrows():
            if wk not in fs_w.index or np.isnan(fs_w.loc[wk, "cap"]):
                continue
            disc = np.clip(-rng.normal(r["coef"], r["se"], n), 0, None) / 100.0
            avail = 1 - fs_w.loc[wk, "stock"] / 24.0
            vol = stations * L / 52.0 * (sh if fuel == "diesel" else 1 - sh) * avail
            acc += disc * vol
        by_fuel[fuel] = {"p05": float(np.quantile(acc, 0.05) / 1e6), "p50": float(np.median(acc) / 1e6),
                         "p95": float(np.quantile(acc, 0.95) / 1e6)}
        total += acc
    return {"eur_million": {"p05": float(np.quantile(total, 0.05) / 1e6), "p50": float(np.median(total) / 1e6),
                            "p95": float(np.quantile(total, 0.95) / 1e6)},
            "by_fuel_eur_million": by_fuel,
            "assumptions": {"stations": stations, "litres_per_station_year": list(litres_per_station_year),
                            "diesel_share": list(diesel_share)}}
