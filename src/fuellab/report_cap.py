"""Run part 1 (TotalEnergies price cap) and write tables, figures and results.md.

    python -m fuellab.report_cap

Inputs (built by `python -m fuellab.pipeline aggregate-fr` and the OSM export):
    data/processed/fr_daily_{diesel,e10}.csv.gz, data/processed/fr_stations.csv.gz,
    data/raw/fr_stockouts_2026.csv.gz, data/raw/osm_fr_fuel_stations.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import cap, load

ROOT = Path(__file__).resolve().parents[2]
PROC, RAW, REP = ROOT / "data" / "processed", ROOT / "data" / "raw", ROOT / "reports" / "cap"
FIG = REP / "figures"
FUELS = {"diesel": "Gazole", "e10": "E10"}
LABEL = {"diesel": "Diesel", "e10": "E10 petrol"}

# reference palette (dataviz skill), light mode: slot 1 blue, slot 2 orange; neutrals for text and grid
TOTAL_C, OTHER_C = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CAP_SHADE = "#2a78d6"


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _dates(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def _cap_spans(ax, fuel):
    _dates(ax)
    for a, b, lvl in cap.CAP_CALENDAR[fuel]:
        ax.axvspan(pd.Timestamp(a), min(pd.Timestamp(b), pd.Timestamp("2026-09-27")), color=CAP_SHADE, alpha=0.06, lw=0)


def _fig(n=2, h=3.6):
    fig, axes = plt.subplots(1, n, figsize=(11, h), sharey=False)
    fig.patch.set_facecolor(SURF)
    return fig, np.atleast_1d(axes)


def fig_prices(fss):
    fig, axes = _fig()
    for ax, (fuel, fs) in zip(axes, fss.items()):
        _style(ax)
        _cap_spans(ax, fuel)
        w = fs.resample("W-MON", label="left", closed="left")[["median_ct_total", "median_ct_others", "cap_ct"]].mean()
        ax.plot(w.index, w["median_ct_others"], color=OTHER_C, lw=2, label="Other stations")
        ax.plot(w.index, w["median_ct_total"], color=TOTAL_C, lw=2, label="TotalEnergies")
        ax.step(fs.index, fs["cap_ct"], where="post", color=INK2, lw=1, ls="--", label="Cap")
        ax.set_title(LABEL[fuel], loc="left", color=INK, fontsize=11)
        ax.set_ylabel("Median pump price, ct/L", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Total went from dearer than the market to cheaper whenever the cap bound", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "prices.png", dpi=150)
    plt.close(fig)


def _coef_plot(ax, t, color, label):
    ax.axhline(0, color=INK2, lw=0.8)
    ax.fill_between(t.index, t["ci_low"], t["ci_high"], color=color, alpha=0.18, lw=0)
    ax.plot(t.index, t["coef"], color=color, lw=2, marker="o", ms=3, label=label)


def fig_gap(gaps, so):
    fig, axes = _fig()
    for ax, fuel in zip(axes, gaps):
        _style(ax)
        _cap_spans(ax, fuel)
        _coef_plot(ax, gaps[fuel], TOTAL_C, "Price gap")
        ax.set_title(f"{LABEL[fuel]}: Total vs other stations", loc="left", color=INK, fontsize=11)
        ax.set_ylabel("ct/L vs usual gap (Jan-Feb = 0)", color=INK2, fontsize=9)
    fig.suptitle("Price: Total's discount relative to its usual position, by week (95% CI)", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "gap.png", dpi=150)
    plt.close(fig)

    fig, axes = _fig()
    for ax, fuel in zip(axes, so):
        _style(ax)
        _cap_spans(ax, fuel)
        _coef_plot(ax, so[fuel], TOTAL_C, "Stockout gap")
        ax.set_title(f"{LABEL[fuel]}: extra hours a day without fuel at Total", loc="left", color=INK, fontsize=11)
        ax.set_ylabel("hours per day vs usual (Jan-Feb = 0)", color=INK2, fontsize=9)
    fig.suptitle("Shortage: the cap emptied Total's pumps when the market rose far above it", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "stockouts.png", dpi=150)
    plt.close(fig)


def fig_dose(fss):
    fig, axes = _fig()
    for ax, (fuel, fs) in zip(axes, fss.items()):
        _style(ax)
        t = fs.dropna(subset=["cap_ct"])
        x = t["market_minus_cap_ct"]
        y = t["stockout_h_total"] - t["stockout_h_others"]
        ax.scatter(x, y, s=14, color=TOTAL_C, alpha=0.55, edgecolor="none", label="One day")
        tab, fit = cap.stockout_dose(fs)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, fit["intercept_h"] + fit["slope_h_per_ct"] * np.clip(xs, 0, None), color=INK, lw=1.5,
                label=f"+{fit['slope_h_per_ct']:.2f} h per ct above the cap")
        ax.axvline(0, color=INK2, lw=0.8, ls="--")
        ax.set_title(LABEL[fuel], loc="left", color=INK, fontsize=11)
        ax.set_xlabel("Market median minus cap, ct/L (right of 0: the cap binds)", color=INK2, fontsize=9)
        ax.set_ylabel("Extra stockout hours at Total", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("The further the market sits above the cap, the longer Total runs dry", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "dose_response.png", dpi=150)
    plt.close(fig)


def fig_spill(spills):
    fig, axes = _fig()
    for ax, fuel in zip(axes, spills):
        _style(ax)
        _cap_spans(ax, fuel)
        _coef_plot(ax, spills[fuel], OTHER_C, "Rivals within 1 km minus rivals > 5 km")
        ax.set_title(f"{LABEL[fuel]}: rivals near a Total station", loc="left", color=INK, fontsize=11)
        ax.set_ylabel("ct/L vs usual gap (Jan-Feb = 0)", color=INK2, fontsize=9)
    fig.suptitle("Spillover: rivals next to Total cut 1-2 ct in April-June, but not when Total ran dry in September",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "spillover.png", dpi=150)
    plt.close(fig)


def _rec(df):
    return json.loads(df.reset_index().to_json(orient="records", date_format="iso"))


def run() -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    brands = load.read_osm_brands(RAW / "osm_fr_fuel_stations.csv")
    stations = pd.read_csv(PROC / "fr_stations.csv.gz", dtype={"post_code": str})
    so = pd.read_csv(RAW / "fr_stockouts_2026.csv.gz")
    out, fss, gaps, sos, spills, gaps_noa = {}, {}, {}, {}, {}, {}
    for fuel, nom in FUELS.items():
        daily = pd.read_csv(PROC / f"fr_daily_{fuel}.csv.gz", parse_dates=["day"])
        last = daily["day"].max()
        h = load.stockout_hours(so, nom, daily["day"].min(), last)
        h["day"] = pd.to_datetime(h["day"])
        d = cap.build(daily, stations, brands, h, fuel)
        w = cap.weekly(d)
        fs = fss[fuel] = cap.first_stage(d)
        gaps[fuel], sos[fuel] = cap.gap_study(w), cap.gap_study(w, y="stockout_h")
        spills[fuel] = cap.spillover(w)
        gaps_noa[fuel] = cap.gap_study(w[w["road"] != "A"])          # robustness: no motorway stations
        dose_tab, dose = cap.stockout_dose(fs)
        drv = cap.driver_mc(gaps[fuel], fs)
        on = fs["cap_ct"].notna() & (fs["market_minus_cap_ct"] > 0)
        pre = (fs.index < cap.PRE_END)
        out[fuel] = {
            "stations": {"total": int(d.loc[d.total == 1, "station"].nunique()),
                         "others": int(d.loc[d.total == 0, "station"].nunique()),
                         "station_days": int(len(d)), "first_day": str(d["day"].min().date()),
                         "last_day": str(d["day"].max().date())},
            "usual_premium_ct": float((fs.loc[pre, "median_ct_total"] - fs.loc[pre, "median_ct_others"]).mean()),
            "share_at_cap_value": {"total_when_binding": float(fs.loc[on, "share_at_a_cap_value_total"].mean()),
                                   "others_when_binding": float(fs.loc[on, "share_at_a_cap_value_others"].mean()),
                                   "total_calm": float(fs.loc[pre, "share_at_a_cap_value_total"].mean())},
            "days_binding": int(on.sum()),
            "gap_peak": {"week": str(gaps[fuel]["coef"].idxmin().date()), "ct": float(gaps[fuel]["coef"].min())},
            "gap_mean_when_binding_ct": float(gaps[fuel].loc[gaps[fuel].index.isin(
                fs[on].index.to_period("W").start_time.unique()), "coef"].mean()),
            "gap_peak_no_motorway_ct": float(gaps_noa[fuel]["coef"].min()),
            "stockout_peak": {"week": str(sos[fuel]["coef"].idxmax().date()), "hours": float(sos[fuel]["coef"].max())},
            "stockout_calm_h": {"total": float(fs.loc[pre, "stockout_h_total"].mean()),
                                "others": float(fs.loc[pre, "stockout_h_others"].mean())},
            "dose": dose, "dose_table": _rec(dose_tab.astype({"days": int}).rename_axis("bin").reset_index()
                                             .assign(bin=lambda x: x["bin"].astype(str)).set_index("bin")),
            "spillover_min_ct": float(spills[fuel]["coef"].min()),
            "spillover_mean_apr_jun_ct": float(spills[fuel].loc["2026-04-13":"2026-06-28", "coef"].mean()),
            "driver": {"weeks": len(drv), "p_gain_positive_min": float(drv["p_gain_positive"].min()),
                       "worst_week": str(drv.loc[drv["p_gain_positive"].idxmin(), "week"].date()),
                       "p_empty_max": float(drv["p_empty"].max()),
                       "expected_gain_eur_mean": float(drv["expected_gain_eur"].mean())},
        }
        for name, t in (("gap", gaps[fuel]), ("stockout_gap", sos[fuel]), ("spillover", spills[fuel]),
                        ("gap_no_motorway", gaps_noa[fuel])):
            t.to_csv(REP / f"{name}_{fuel}.csv", float_format="%.4f")
        drv.to_csv(REP / f"driver_{fuel}.csv", index=False, float_format="%.4f")
        fs.to_csv(REP / f"daily_{fuel}.csv", float_format="%.4f")
        print(fuel, "done", flush=True)
    out["cost"] = cap.cost_mc(gaps, fss)
    fig_prices(fss)
    fig_gap(gaps, sos)
    fig_dose(fss)
    fig_spill(spills)
    (REP / "results.json").write_text(json.dumps(out, indent=1))
    write_md(out)
    return out


def write_md(r: dict) -> None:
    L = []
    L.append("# Results: TotalEnergies price cap, France 2026\n")
    L.append("Generated by `python -m fuellab.report_cap`. Prices in euro cents per litre.\n")
    L.append("| | Diesel | E10 |\n|---|---|---|")
    rows = [
        ("Total stations / other stations", lambda x: f"{x['stations']['total']:,} / {x['stations']['others']:,}"),
        ("Station-days", lambda x: f"{x['stations']['station_days']:,}"),
        ("Usual Total premium (Jan-Feb, median vs median)", lambda x: f"+{x['usual_premium_ct']:.1f} ct"),
        ("Days the cap bound (market median above cap)", lambda x: f"{x['days_binding']}"),
        ("Total stations exactly at a cap value, binding days", lambda x: f"{x['share_at_cap_value']['total_when_binding']:.0%}"),
        ("Other stations exactly at a cap value, binding days", lambda x: f"{x['share_at_cap_value']['others_when_binding']:.0%}"),
        ("Largest weekly price gap vs usual", lambda x: f"{x['gap_peak']['ct']:.1f} ct (week of {x['gap_peak']['week']})"),
        ("Same, motorway stations excluded", lambda x: f"{x['gap_peak_no_motorway_ct']:.1f} ct"),
        ("Mean weekly gap in binding weeks", lambda x: f"{x['gap_mean_when_binding_ct']:.1f} ct"),
        ("Stockout hours per day, calm period: Total / others", lambda x: f"{x['stockout_calm_h']['total']:.2f} / {x['stockout_calm_h']['others']:.2f}"),
        ("Largest weekly extra stockout at Total", lambda x: f"+{x['stockout_peak']['hours']:.1f} h/day (week of {x['stockout_peak']['week']})"),
        ("Extra stockout per ct the market sits above the cap", lambda x: f"+{x['dose']['slope_h_per_ct']:.2f} h (90% CI {x['dose']['slope_ci'][0]:.2f} to {x['dose']['slope_ci'][1]:.2f})"),
        ("Rivals within 1 km vs > 5 km, mean mid-Apr to Jun", lambda x: f"{x['spillover_mean_apr_jun_ct']:.1f} ct"),
        ("Driver: worst week, chance the Total trip pays off", lambda x: f"{x['driver']['p_gain_positive_min']:.0%} (week of {x['driver']['worst_week']}, pump empty {x['driver']['p_empty_max']:.0%} of the time)"),
    ]
    for name, fn in rows:
        L.append(f"| {name} | {fn(r['diesel'])} | {fn(r['e10'])} |")
    c = r["cost"]
    L.append("\n## Cost of the cap to TotalEnergies (order of magnitude)\n")
    L.append(f"Median **EUR {c['eur_million']['p50']:.0f} million**, 90% interval {c['eur_million']['p05']:.0f} to "
             f"{c['eur_million']['p95']:.0f} (diesel {c['by_fuel_eur_million']['diesel']['p50']:.0f}, "
             f"petrol {c['by_fuel_eur_million']['e10']['p50']:.0f}), March to September 2026.\n")
    a = c["assumptions"]
    L.append(f"Assumptions: {a['stations']:,} stations; {a['litres_per_station_year'][0]/1e6:.0f} to "
             f"{a['litres_per_station_year'][1]/1e6:.0f} million litres per station per year; diesel share "
             f"{a['diesel_share'][0]:.0%} to {a['diesel_share'][1]:.0%}; litres scaled down by the hours the pump was empty. "
             "Station volumes are not public, so this is the least certain number in the project.\n")
    for fuel in ("diesel", "e10"):
        L.append(f"\n## Stockout dose-response, {LABEL[fuel]}\n")
        L.append("| market minus cap (ct) | days | extra stockout hours at Total |\n|---|---|---|")
        for row in r[fuel]["dose_table"]:
            L.append(f"| {row['bin']} | {row['days']} | {row['excess_stockout_h']:.1f} |")
    (REP / "results.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    run()
