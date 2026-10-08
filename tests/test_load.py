import gzip

import numpy as np
import pandas as pd

from fuellab import load

TK = """date,station_uuid,diesel,e5,e10,dieselchange,e5change,e10change
2026-04-01 11:58:02+02,s1,2.299,2.199,2.139,1,1,1
2026-04-01 12:00:07+02,s1,2.379,2.279,2.219,1,1,1
2026-04-01 12:00:09+02,s2,0.000,2.259,2.199,0,1,1
"""

FR = """<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?><pdv_liste>
<pdv id="75000001" latitude="4885000" longitude="235000" cp="75001" pop="R">
  <adresse>1 RUE</adresse><ville>PARIS</ville>
  <prix nom="Gazole" id="1" maj="2026-04-01T06:01:00" valeur="2.109"/>
  <prix nom="E10" id="5" maj="2026-04-01T06:01:00" valeur="1959"/>
</pdv>
</pdv_liste>
"""


def test_tankerkoenig_reader_converts_time_and_drops_invalid(tmp_path):
    f = tmp_path / "2026-04-01-prices.csv"
    f.write_text(TK)
    d = load.read_tk_prices(f, "diesel")
    assert list(d["station"]) == ["s1", "s1"]                      # 0.000 = not offered, dropped
    assert d["ts"].iloc[1] == pd.Timestamp("2026-04-01 12:00:07")  # local Berlin time, naive
    assert np.isclose(d["price"].iloc[1], 2.379)


def test_price_file_discovery(tmp_path):
    (tmp_path / "prices" / "2026" / "04").mkdir(parents=True)
    for day in ("2026-03-31", "2026-04-01", "2026-04-02"):
        (tmp_path / "prices" / "2026" / "04" / f"{day}-prices.csv").write_text(TK)
    got = load.tk_price_files(tmp_path, pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-30"))
    assert [p.name[:10] for p in got] == ["2026-04-01", "2026-04-02"]


def test_france_xml_roundtrip_and_millimes(tmp_path):
    x = tmp_path / "fr.xml"
    x.write_text(FR, encoding="latin-1")
    flat = tmp_path / "fr.csv.gz"
    assert load.parse_fr_xml(x, flat) == 2
    d = load.read_fr_prices(flat, "E10")
    assert np.isclose(d["price"].iloc[0], 1.959)                   # 1959 millièmes -> 1.959 euro
    st = load.read_fr_stations(flat)
    assert np.isclose(st["latitude"].iloc[0], 48.85)
