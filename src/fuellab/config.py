"""Dates, fuels and thresholds used across the pipeline. Everything that is a judgement call lives here."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

RULE_START = pd.Timestamp("2026-04-01")      # first day prices may rise only at 12:00
WAR_SHOCK = pd.Timestamp("2026-02-28")       # start of the oil price shock (crude spike); marks a second pre-period
NOON_WINDOW_MIN = 10                         # an increase between 12:00:00 and 12:09:59 counts as "at noon"
PRICE_MIN, PRICE_MAX = 0.90, 3.50            # euro per litre; outside this a reported price is treated as invalid
UP_EPS = 0.0005                              # a change smaller than 0.05 ct is not a change (rounding noise)
HOURS = list(range(24))
SAMPLE_MINUTE = 30                           # hourly profile: the price in effect at hh:30


@dataclass(frozen=True)
class Fuel:
    name: str      # our name
    de: str        # Tankerkönig column
    fr: str        # French open data "nom"


FUELS = {
    "diesel": Fuel("diesel", "diesel", "Gazole"),
    "e10": Fuel("e10", "e10", "E10"),
}


@dataclass(frozen=True)
class Windows:
    """Analysis windows. 'calm' is before the oil shock, 'shock' is the month between shock and rule."""
    start: pd.Timestamp = pd.Timestamp("2026-01-05")
    end: pd.Timestamp = pd.Timestamp("2026-09-27")   # last complete week in the 2026 file used here


WINDOWS = Windows()
COMPETITION_RADIUS_KM = 2.0
SEED = 7
