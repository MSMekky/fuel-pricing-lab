"""Readers for the two raw sources. Both return the same event log: one row per reported price.

Event log columns: station (str), ts (naive local time), price (float, euro per litre).
Repeated reports of an unchanged price are kept: they do not change any daily statistic.

Germany: Tankerkönig / MTS-K daily CSVs (prices/YYYY/MM/YYYY-MM-DD-prices.csv), CC BY 4.0.
France: prix-carburants annual XML (PrixCarburants_annuel_YYYY.xml), Licence Ouverte 2.0.
"""
from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PRICE_MIN, PRICE_MAX

EVENT_COLS = ["station", "ts", "price"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["price"] >= PRICE_MIN) & (df["price"] <= PRICE_MAX)]
    df = df.dropna(subset=["ts"])
    return df.sort_values(["station", "ts"], kind="mergesort").reset_index(drop=True)[EVENT_COLS]


# ------------------------------------------------------------------ Germany
def read_tk_prices(path: str | Path, fuel_col: str) -> pd.DataFrame:
    """One Tankerkönig daily prices file -> event log for one fuel.

    Timestamps come with a UTC offset ("2026-04-01 12:00:07+02") and are converted to naive Berlin time.
    """
    df = pd.read_csv(path, usecols=["date", "station_uuid", fuel_col], dtype={"station_uuid": str})
    ts = pd.to_datetime(df["date"], utc=True, format="mixed").dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    out = pd.DataFrame({"station": df["station_uuid"], "ts": ts, "price": df[fuel_col].astype(float)})
    return _clean(out)


def read_tk_stations(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"uuid": str, "post_code": str})
    df = df.rename(columns={"uuid": "station"})
    df["brand"] = df["brand"].fillna("").str.strip()
    keep = ["station", "name", "brand", "post_code", "city", "latitude", "longitude"]
    df = df[keep].copy()
    df = df[(df["latitude"].between(47, 55.5)) & (df["longitude"].between(5.5, 15.5))]
    return df.drop_duplicates("station").reset_index(drop=True)


def tk_price_files(root: str | Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    """All daily files under root (any folder layout) whose date lies in [start, end]."""
    out = []
    for p in Path(root).rglob("*-prices.csv"):
        try:
            d = pd.Timestamp(p.name[:10])
        except ValueError:
            continue
        if start <= d <= end:
            out.append(p)
    return sorted(out)


# ------------------------------------------------------------------ France
def parse_fr_xml(src: str | Path, dst: str | Path) -> int:
    """Stream the (large) annual XML into a flat gzipped CSV. Returns the number of price rows."""
    from lxml import etree
    n = 0
    with gzip.open(dst, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["station", "lat", "lon", "cp", "pop", "fuel", "ts", "price"])
        for _, pdv in etree.iterparse(str(src), tag="pdv"):
            meta = [pdv.get(k) for k in ("id", "latitude", "longitude", "cp", "pop")]
            for p in pdv.iterfind("prix"):
                v = p.get("valeur")
                if v is not None:
                    w.writerow(meta + [p.get("nom"), p.get("maj"), v])
                    n += 1
            pdv.clear()
            while pdv.getprevious() is not None:
                del pdv.getparent()[0]
    return n


def read_fr_prices(flat_csv: str | Path, fuel_nom: str) -> pd.DataFrame:
    df = pd.read_csv(flat_csv, dtype={"station": str, "cp": str})
    df = df[df["fuel"] == fuel_nom]
    price = df["price"].astype(float)
    price = np.where(price > 100, price / 1000.0, price)       # older files report millièmes
    out = pd.DataFrame({"station": "FR" + df["station"], "ts": pd.to_datetime(df["ts"], errors="coerce"),
                        "price": price})
    return _clean(out)


def read_fr_stations(flat_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(flat_csv, usecols=["station", "lat", "lon", "cp", "pop"], dtype={"station": str, "cp": str})
    df = df.drop_duplicates("station")
    return pd.DataFrame({"station": "FR" + df["station"], "latitude": df["lat"] / 1e5, "longitude": df["lon"] / 1e5,
                         "post_code": df["cp"], "road": df["pop"]}).reset_index(drop=True)


def concat_events(parts: Iterable[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=EVENT_COLS)
    return pd.concat(parts, ignore_index=True).sort_values(["station", "ts"], kind="mergesort").reset_index(drop=True)


def parse_fr_stockouts(src: str | Path, dst: str | Path) -> int:
    """Temporary stockouts (<rupture type="temporaire">) -> station, fuel, start, end. Returns the row count."""
    from lxml import etree
    n = 0
    with gzip.open(dst, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["station", "fuel", "start", "end"])
        for _, pdv in etree.iterparse(str(src), tag="pdv"):
            sid = pdv.get("id")
            for r in pdv.iterfind("rupture"):
                if r.get("type", "temporaire") == "temporaire":
                    w.writerow(["FR" + sid, r.get("nom"), r.get("debut"), r.get("fin")])
                    n += 1
            pdv.clear()
            while pdv.getprevious() is not None:
                del pdv.getparent()[0]
    return n


def stockout_hours(stockouts: pd.DataFrame, fuel_nom: str, start, end) -> pd.DataFrame:
    """Hours out of stock per station and day, for one fuel. An open episode (no end) is cut at `end`."""
    d = stockouts[stockouts["fuel"] == fuel_nom].copy()
    d["start"] = pd.to_datetime(d["start"], errors="coerce")
    d["end"] = pd.to_datetime(d["end"], errors="coerce").fillna(pd.Timestamp(end) + pd.Timedelta(days=1))
    lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    d = d.dropna(subset=["start"])
    d = d[(d["end"] > lo) & (d["start"] < hi)]
    d["start"], d["end"] = d["start"].clip(lower=lo), d["end"].clip(upper=hi)
    d = d[d["end"] > d["start"]]
    rows = []
    for st, s, e in d[["station", "start", "end"]].itertuples(index=False):
        day = s.normalize()
        while day < e:
            nxt = day + pd.Timedelta(days=1)
            h = (min(e, nxt) - max(s, day)).total_seconds() / 3600
            rows.append((st, day, h))
            day = nxt
    out = pd.DataFrame(rows, columns=["station", "day", "stockout_h"])
    return out.groupby(["station", "day"], as_index=False)["stockout_h"].sum().assign(
        stockout_h=lambda x: x["stockout_h"].clip(upper=24))


BRAND_RULES = [
    ("total", r"\btotal"),            # TotalEnergies, Total, Total Access: the capped network
    ("elan", r"\b[ée]lan\b"),            # rural brand supplied by TotalEnergies; kept separate
    ("supermarket", r"leclerc|intermarch|carrefour|auchan|super u|syst[eè]me u|^u$|station u|hyper u|casino|"
                    r"netto|leader price|cora|g[ée]ant|match|lidl|bi1|colruyt|spar"),
    ("other_brand", r"esso|avia|\bbp\b|shell|\beni\b|agip|dyneff|vito|elf|oil|q8|ad[bp]|bolloré|ecomarch"),
]


def classify_brand(brand: pd.Series, operator: pd.Series, name: pd.Series) -> pd.Series:
    text = (brand.fillna("") + " | " + operator.fillna("") + " | " + name.fillna("")).str.lower()
    b = brand.fillna("").str.lower().str.strip()
    out = pd.Series("independent", index=brand.index)
    out[text.str.strip(" |") == ""] = "unknown"
    for label, pat in reversed(BRAND_RULES):        # first rule wins, so apply in reverse
        hit = b.str.contains(pat, regex=True) | ((b == "") & text.str.contains(pat, regex=True))
        out[hit] = label
    return out


def read_osm_brands(path: str | Path) -> pd.DataFrame:
    """OpenStreetMap fuel stations (Overpass CSV export) -> station id of the French price feed and brand group."""
    o = pd.read_csv(path, dtype=str)
    o.columns = ["osm_type", "osm_id", "lat", "lon", "ref", "brand", "operator", "name"]
    o = o.dropna(subset=["ref"])
    o["ref"] = o["ref"].str.strip()
    o = o[o["ref"].str.fullmatch(r"\d{7,8}")]
    o = o.drop_duplicates("ref", keep=False)        # a price id on two OSM objects is ambiguous: drop both
    o["brand_group"] = classify_brand(o["brand"], o["operator"], o["name"])
    return pd.DataFrame({"station": "FR" + o["ref"], "brand": o["brand"], "brand_group": o["brand_group"]})
