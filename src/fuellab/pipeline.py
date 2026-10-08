"""End to end: raw feeds -> station-day panels -> results.

    python -m fuellab.pipeline aggregate-fr --raw data/raw/PrixCarburants_annuel_2026.xml   # France, both parts
    python -m fuellab.pipeline cap                                                          # part 1: Total price cap
    python -m fuellab.pipeline aggregate-de --raw data/raw/tankerkoenig    # part 2: streams one daily file at a time
    python -m fuellab.pipeline noonrule                                    # part 2: Germany's 12:00 rule vs France

The aggregation steps need only numpy and pandas, so they can run next to the raw data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import noonrule as A
from . import load
from .config import FUELS, WINDOWS
from .daily import summarise, summarise_day

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
WARMUP_DAYS = 7


def aggregate_de(raw: Path, start=WINDOWS.start, end=WINDOWS.end) -> None:
    files = load.tk_price_files(raw, start - pd.Timedelta(days=WARMUP_DAYS), end)
    if not files:
        raise SystemExit(f"no Tankerkönig *-prices.csv files under {raw}")
    states = {f: pd.Series(dtype=float) for f in FUELS}
    rows = {f: [] for f in FUELS}
    for i, path in enumerate(files):
        day = pd.Timestamp(path.name[:10])
        for f, fuel in FUELS.items():
            ev = load.read_tk_prices(path, fuel.de)
            r, states[f] = summarise_day(day, ev, states[f])
            if day >= start and len(r):
                rows[f].append(r)
        if i % 10 == 0:
            print(f"{path.name}  ({i + 1}/{len(files)})", flush=True)
    PROC.mkdir(parents=True, exist_ok=True)
    for f in FUELS:
        out = pd.concat(rows[f], ignore_index=True)
        out.to_csv(PROC / f"de_daily_{f}.csv.gz", index=False, float_format="%.4f")
    st = sorted(raw.rglob("*-stations.csv"))
    if st:
        load.read_tk_stations(st[-1]).to_csv(PROC / "de_stations.csv.gz", index=False)


def aggregate_fr(raw: Path, start=WINDOWS.start, end=WINDOWS.end) -> None:
    flat = raw.with_suffix(".csv.gz") if raw.suffix == ".xml" else raw
    if raw.suffix == ".xml" and not flat.exists():
        print("parsing XML ...", load.parse_fr_xml(raw, flat), "rows")
    stock = raw.parent / "fr_stockouts_2026.csv.gz"
    if raw.suffix == ".xml" and not stock.exists():
        print("stockouts ...", load.parse_fr_stockouts(raw, stock), "rows")
    PROC.mkdir(parents=True, exist_ok=True)
    for f, fuel in FUELS.items():
        ev = load.read_fr_prices(flat, fuel.fr)
        d = summarise(ev, start - pd.Timedelta(days=WARMUP_DAYS), end, warmup_days=WARMUP_DAYS)
        d.to_csv(PROC / f"fr_daily_{f}.csv.gz", index=False, float_format="%.4f")
    load.read_fr_stations(flat).to_csv(PROC / "fr_stations.csv.gz", index=False)


def read_panel(fuel: str) -> pd.DataFrame:
    de = pd.read_csv(PROC / f"de_daily_{fuel}.csv.gz", parse_dates=["day"])
    fr = pd.read_csv(PROC / f"fr_daily_{fuel}.csv.gz", parse_dates=["day"])
    return A.build_panel(de, fr)


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.reset_index().to_json(orient="records", date_format="iso"))


def analyse_noonrule() -> dict:
    out = {"fuels": {}}
    st_path = PROC / "de_stations.csv.gz"
    stations = pd.read_csv(st_path) if st_path.exists() else None
    for fuel in FUELS:
        p = read_panel(fuel)
        res = {
            "panel": {"station_days": int(len(p)), "stations_de": int(p.loc[p.de == 1, "station"].nunique()),
                      "stations_fr": int(p.loc[p.de == 0, "station"].nunique()),
                      "first_day": str(p["day"].min().date()), "last_day": str(p["day"].max().date())},
            "mechanics": _records(A.mechanics(p)),
            "profile_de": _records(A.profile(p, "DE")),
            "did": [A.did(p, ("calm", "shock")), A.did(p, ("calm",)), A.did(p, ("shock",)),
                    A.did(p, ("calm", "shock"), y="spread_ct")],
            "event_study": _records(A.event_study(p)),
            "placebo": _records(A.placebo(p, ["2026-01-26", "2026-02-09", "2026-02-23", "2026-03-09"])),
            "timing_de": _records(A.timing(p, "DE")),
            "daily_means": _records(p.groupby(["day", "country"])["price_ct"].mean().unstack()),
        }
        if stations is not None:
            res["competition"] = _records(A.competition(p, stations))
        out["fuels"][fuel] = res
        print(fuel, "done", flush=True)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "noonrule").mkdir(parents=True, exist_ok=True)
    (REPORTS / "noonrule" / "results.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["aggregate-fr", "cap", "aggregate-de", "noonrule"])
    ap.add_argument("--raw", type=Path)
    a = ap.parse_args()
    if a.step == "aggregate-de":
        aggregate_de(a.raw or ROOT / "data" / "raw" / "tankerkoenig")
    elif a.step == "aggregate-fr":
        aggregate_fr(a.raw or ROOT / "data" / "raw" / "PrixCarburants_annuel_2026.xml")
    elif a.step == "cap":
        from . import report_cap
        report_cap.run()
    else:
        analyse_noonrule()


if __name__ == "__main__":
    main()
