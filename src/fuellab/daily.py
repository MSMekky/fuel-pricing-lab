"""From an event log to one row per station and day.

A posted price is a step function: it holds until the next report. Each day is summarised from the
price in effect at 00:00 (carried over from the previous day) plus that day's reports:

    mean_tw     time-weighted mean price over the day (what a driver arriving at a random time pays)
    p_min/p_max lowest and highest price in effect during the day
    n_up/n_down number of increases / cuts
    n_up_noon   increases that happen in the noon window (12:00 to 12:09)
    up_noon_ct  size of the noon increase, cents (sum if several)
    p00..p23    price in effect at hh:30, for the intraday profile

The function streams day by day, so a year of German data never has to sit in memory at once.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from .config import HOURS, NOON_WINDOW_MIN, SAMPLE_MINUTE, UP_EPS

HOUR_COLS = [f"p{h:02d}" for h in HOURS]


def summarise_day(day: pd.Timestamp, events: pd.DataFrame, state: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Summarise one calendar day. `state` maps station -> price in effect at 00:00. Returns (rows, new state)."""
    day = pd.Timestamp(day).normalize()
    day_end = day + pd.Timedelta(days=1)
    ev = events[(events["ts"] >= day) & (events["ts"] < day_end)]
    opening = pd.DataFrame({"station": state.index, "ts": day, "price": state.to_numpy(dtype=float), "_open": True})
    ev = ev.assign(_open=False)
    df = pd.concat([opening, ev], ignore_index=True).sort_values(["station", "ts", "_open"],
                                                                  ascending=[True, True, False], kind="mergesort")
    if df.empty:
        return pd.DataFrame(), state
    st = df["station"].to_numpy()
    new_station = np.r_[True, st[1:] != st[:-1]]
    last_of_station = np.r_[st[1:] != st[:-1], True]
    ts = df["ts"].to_numpy()
    nxt = np.where(last_of_station, np.datetime64(day_end), np.r_[ts[1:], np.datetime64(day_end)])
    dur = (nxt - ts).astype("timedelta64[s]").astype(float)
    price = df["price"].to_numpy(dtype=float)
    prev = np.where(new_station, np.nan, np.r_[np.nan, price[:-1]])
    diff = price - prev
    up = np.nan_to_num(diff, nan=0.0) > UP_EPS
    down = np.nan_to_num(diff, nan=0.0) < -UP_EPS
    tsi = pd.DatetimeIndex(ts)
    noon = (tsi.hour == 12) & (tsi.minute < NOON_WINDOW_MIN)

    work = pd.DataFrame({"station": st, "pd": price * dur, "dur": dur, "price": price, "up": up, "down": down,
                         "up_noon": up & noon, "up_noon_ct": np.where(up & noon, diff * 100, 0.0),
                         "max_up_ct": np.where(up, diff * 100, 0.0)})
    g = work.groupby("station", sort=True)
    out = pd.DataFrame({
        "mean_tw": g["pd"].sum() / g["dur"].sum(),
        "covered_h": g["dur"].sum() / 3600.0,
        "p_min": g["price"].min(),
        "p_max": g["price"].max(),
        "n_up": g["up"].sum().astype(int),
        "n_down": g["down"].sum().astype(int),
        "n_up_noon": g["up_noon"].sum().astype(int),
        "up_noon_ct": g["up_noon_ct"].sum(),
        "max_up_ct": g["max_up_ct"].max(),
    })
    # hourly profile: price in effect at hh:30
    # df is sorted by station then time, so (station code, seconds since midnight) is a sorted key
    code = np.cumsum(new_station) - 1                                  # 0..S-1 in the order of out.index
    secs = (ts - np.datetime64(day)).astype("timedelta64[s]").astype(np.int64)
    key = code * 100_000 + secs
    gsec = np.array([h * 3600 + SAMPLE_MINUTE * 60 for h in HOURS], dtype=np.int64)
    gkey = (np.arange(code[-1] + 1)[:, None] * 100_000 + gsec[None, :]).ravel()
    pos = np.searchsorted(key, gkey, side="right") - 1
    ok = (pos >= 0) & (code[np.clip(pos, 0, None)] == gkey // 100_000)
    hourly = np.where(ok, price[np.clip(pos, 0, None)], np.nan).reshape(-1, len(HOURS))
    wide = pd.DataFrame(hourly, index=out.index, columns=HOUR_COLS)
    out = out.join(wide)
    out.insert(0, "day", day)
    out = out.reset_index()
    new_state = df.groupby("station", sort=False)["price"].last()
    new_state = pd.concat([state[~state.index.isin(new_state.index)], new_state])
    return out, new_state


def iter_days(events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
              state: pd.Series | None = None) -> Iterator[pd.DataFrame]:
    state = pd.Series(dtype=float) if state is None else state
    for day in pd.date_range(start, end, freq="D"):
        rows, state = summarise_day(day, events, state)
        if len(rows):
            yield rows


def summarise(events: pd.DataFrame, start=None, end=None, warmup_days: int = 0) -> pd.DataFrame:
    """Summarise every day in [start, end]. The first `warmup_days` only build the 00:00 state and are dropped,
    so the first reported day already starts from a known price for stations that do not report daily."""
    if events.empty:
        return pd.DataFrame()
    start = pd.Timestamp(start or events["ts"].min()).normalize()
    end = pd.Timestamp(end or events["ts"].max()).normalize()
    first_kept = start + pd.Timedelta(days=warmup_days)
    rows = [r for r in iter_days(events, start, end) if r["day"].iloc[0] >= first_kept]
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    float_cols = [c for c in out.columns if out[c].dtype == float]
    out[float_cols] = out[float_cols].astype("float32")
    return out
