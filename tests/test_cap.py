import numpy as np
import pandas as pd

from fuellab import cap, load


def test_cap_calendar_levels():
    days = pd.Series(pd.to_datetime(["2026-02-20", "2026-03-05", "2026-03-13", "2026-04-08", "2026-07-01", "2026-07-22"]))
    d = cap.cap_level("diesel", days).tolist()
    assert np.isnan(d[0]) and d[1:4] == [1.99, 2.09, 2.25] and np.isnan(d[4]) and d[5] == 2.25
    e = cap.cap_level("e10", days).tolist()
    assert np.isnan(e[1]) and e[2] == 1.99 and np.isnan(e[4])


def test_nearest_total_excludes_itself():
    st = pd.DataFrame({"station": ["T1", "T2", "X"], "brand_group": ["total", "total", "supermarket"],
                       "latitude": [48.0, 48.0, 48.0], "longitude": [2.0, 2.0 + 0.0135, 2.0 + 0.0405]})
    d = cap.nearest_total_km(st).to_numpy()
    km = 0.0135 * 111.32 * np.cos(np.radians(48.0))
    assert np.allclose(d, [km, km, 2 * km], rtol=0.01)


def test_stockout_hours_split_across_midnight_and_open_episodes():
    so = pd.DataFrame({"station": ["A", "B"], "fuel": ["Gazole", "Gazole"],
                       "start": ["2026-04-01T20:00:00", "2026-04-02T18:00:00"], "end": ["2026-04-02T02:00:00", None]})
    h = load.stockout_hours(so, "Gazole", "2026-04-01", "2026-04-03").set_index(["station", "day"])["stockout_h"]
    assert h[("A", pd.Timestamp("2026-04-01"))] == 4 and h[("A", pd.Timestamp("2026-04-02"))] == 2
    assert h[("B", pd.Timestamp("2026-04-02"))] == 6 and h[("B", pd.Timestamp("2026-04-03"))] == 24


def test_brand_classification():
    b = pd.Series(["TotalEnergies", "Total Access", "Élan", "Intermarché", "Esso", "", "Gulf"])
    o = pd.Series(["", "", "", "", "", "", ""])
    n = pd.Series(["", "", "", "", "", "Relais Total de la Gare", "Garage Martin"])
    assert load.classify_brand(b, o, n).tolist() == ["total", "total", "elan", "supermarket", "other_brand",
                                                      "total", "independent"]


def _weekly_panel(effect_weeks, effect, seed=0, n_total=150, n_other=300):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2026-01-05", "2026-06-29", freq="7D")
    ids = [f"T{i}" for i in range(n_total)] + [f"O{i}" for i in range(n_other)]
    tot = np.r_[np.ones(n_total), np.zeros(n_other)].astype(int)
    fe = rng.normal(0, 5, len(ids)) + 7 * tot                       # Total normally ~7 ct dearer
    wk_fe = np.cumsum(rng.normal(0, 2, len(weeks))) + 170
    rows = []
    for k, wk in enumerate(weeks):
        bump = effect if wk in effect_weeks else 0.0
        y = fe + wk_fe[k] + bump * tot + rng.normal(0, 1, len(ids))
        rows.append(pd.DataFrame({"station": ids, "week": wk, "price_ct": y, "total": tot,
                                  "brand_group": np.where(tot == 1, "total", "supermarket"), "near_total_km": 3.0}))
    return pd.concat(rows, ignore_index=True)


def test_gap_study_recovers_known_discount_and_flat_pre_period():
    eff = set(pd.date_range("2026-04-06", "2026-05-25", freq="7D"))
    g = cap.gap_study(_weekly_panel(eff, -15.0))
    assert np.allclose(g.loc[list(eff), "coef"], -15.0, atol=0.4)
    assert g.loc[g.index < cap.PRE_END, "coef"].abs().max() < 0.4


def test_stockout_dose_recovers_slope():
    days = pd.date_range("2026-03-13", periods=150)
    rng = np.random.default_rng(1)
    gap = rng.uniform(-15, 25, len(days))
    fs = pd.DataFrame({"cap_ct": 199.0, "market_minus_cap_ct": gap,
                       "stockout_h_others": 0.2, "stockout_h_total": 0.2 + 1.0 + 0.4 * np.clip(gap, 0, None)
                       + rng.normal(0, 0.3, len(days))}, index=days)
    _, fit = cap.stockout_dose(fs)
    assert abs(fit["slope_h_per_ct"] - 0.4) < 0.03 and fit["slope_ci"][0] < 0.4 < fit["slope_ci"][1]
