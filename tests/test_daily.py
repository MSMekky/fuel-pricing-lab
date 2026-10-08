import numpy as np
import pandas as pd

from fuellab.daily import summarise, summarise_day


def ev(rows):
    return pd.DataFrame(rows, columns=["station", "ts", "price"]).assign(ts=lambda d: pd.to_datetime(d["ts"]))


def test_time_weighted_mean_and_counts():
    day = pd.Timestamp("2026-04-02")
    state = pd.Series({"A": 2.00})
    e = ev([("A", "2026-04-02 06:00:00", 1.90),     # cut
            ("A", "2026-04-02 12:00:30", 2.10),     # increase at noon
            ("A", "2026-04-02 18:00:00", 2.00)])    # cut
    rows, new_state = summarise_day(day, e, state)
    r = rows.iloc[0]
    h = 30 / 3600                                    # the noon report arrives at 12:00:30
    expected = (6 * 2.00 + (6 + h) * 1.90 + (6 - h) * 2.10 + 6 * 2.00) / 24
    assert np.isclose(r["mean_tw"], expected)
    assert (r["n_up"], r["n_down"], r["n_up_noon"]) == (1, 2, 1)
    assert np.isclose(r["up_noon_ct"], 20.0)
    assert np.isclose(r["p_min"], 1.90) and np.isclose(r["p_max"], 2.10)
    assert np.isclose(r["p05"], 2.00) and np.isclose(r["p06"], 1.90) and np.isclose(r["p12"], 2.10)
    assert np.isclose(new_state["A"], 2.00)


def test_increase_outside_noon_window_is_not_noon():
    e = ev([("A", "2026-04-02 12:10:00", 2.10)])
    rows, _ = summarise_day(pd.Timestamp("2026-04-02"), e, pd.Series({"A": 2.0}))
    assert rows.iloc[0]["n_up"] == 1 and rows.iloc[0]["n_up_noon"] == 0


def test_price_carries_over_days_without_reports():
    e = ev([("A", "2026-04-01 10:00:00", 2.00), ("B", "2026-04-02 09:00:00", 1.80)])
    out = summarise(e, "2026-04-01", "2026-04-03")
    a = out[out["station"] == "A"].set_index("day")
    assert np.isclose(a.loc["2026-04-03", "mean_tw"], 2.00)            # no report on 3 April, price held
    assert np.isclose(a.loc["2026-04-01", "covered_h"], 14.0)          # unknown before first report
    b = out[out["station"] == "B"].set_index("day")
    assert pd.Timestamp("2026-04-01") not in b.index


def test_repeated_identical_reports_change_nothing():
    base = ev([("A", "2026-04-02 08:00:00", 2.00)])
    dup = ev([("A", "2026-04-02 08:00:00", 2.00), ("A", "2026-04-02 15:00:00", 2.00)])
    s = pd.Series({"A": 2.0})
    r1, _ = summarise_day(pd.Timestamp("2026-04-02"), base, s)
    r2, _ = summarise_day(pd.Timestamp("2026-04-02"), dup, s)
    cols = ["mean_tw", "n_up", "n_down", "p_min", "p_max"]
    assert np.allclose(r1[cols].to_numpy(float), r2[cols].to_numpy(float))
