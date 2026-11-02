
"""
Experiments of "Aggregation as Admission to Reserve Markets".

Uses the formulation of scheduling_milp.py on the GME day-ahead prices (PUN)
and the ex-ante MSD outcomes of the Nord zone, 2023 and 2024, with the MSD
statistics of each year used for that year only.

Modes (they can be combined in one call):

    --scenarios ref fcr25 ...     standalone, aggregated and threshold-free runs
                                  of the portfolio for each scenario and year
    --game                        exact daily cooperative game of the 20 units:
                                  every composition (n_small, n_medium, n_large)
                                  whose threshold-free schedules clear the minimum
                                  bid in every product-hour earns the sum of the
                                  threshold-free values; only the others are solved
    --comps "8,8,3;7,8,4"         annual value of given compositions (e.g. under gran1)
    --durations 1 2 4             value of admission of a 0.5 MW unit by duration
    --bridge                      the formulation changes one at a time, from the
                                  preliminary version to the reference

Example (run from the src folder, data files in ../data/raw):

    python experiments.py --scenarios ref --game --durations 1 2 4 --dump \
        --pun-xlsx ../data/raw/20230101_20231231_PUN.xlsx ../data/raw/20240101_20241231_PUN.xlsx \
        --msd-xlsx ../data/raw/EsitiMSD_PrezziExAnte_*_Nord.xlsx --outdir ../results/new

See the README of the repository for the commands that reproduce every result.
"""
import argparse
import copy
import csv
import json
import math
import os
import time
from dataclasses import replace
from datetime import datetime, timedelta
from itertools import product

import numpy as np

from scheduling_milp import BatteryUnit, ModelOptions, UvamProduct, solve_day

#Lorenzo Giannuzzo: the portfolio, (label, units, power MW, duration h)
CLASSES = (("small", 8, 0.5, 4.0), ("medium", 8, 1.0, 2.0), ("large", 4, 2.5, 1.0))
N_TOT = tuple(c[1] for c in CLASSES)
CYCLE_LIFE = 8000.0
REPLACEMENT_EUR_PER_MWH = 100000.0      # 100 EUR/kWh, reference formulation

#Lorenzo Giannuzzo: scenarios; "opts" overrides ModelOptions, "rules" changes the market data,
# "repl" the replacement cost (EUR/MWh), "forecast" schedules on lagged prices
SCENARIOS = {
    "ref":        dict(),
    "minbid05":   dict(opts=dict(min_bid=0.5)),
    "minbid025":  dict(opts=dict(min_bid=0.25)),
    "gran1":      dict(opts=dict(granularity=1.0)),
    "fcrsym":     dict(opts=dict(fcr_directional=False)),
    "phi047":     dict(opts=dict(fcr_phi=0.047)),   # with the +-10 mHz insensitivity band
    "fcr25":      dict(rules=[("fcr_cap", 25.0)]),
    "fcr10":      dict(rules=[("fcr_cap", 10.0)]),
    "nofcr":      dict(rules=[("no_fcr",)]),
    "afrr05":     dict(rules=[("afrr_cap_scale", 0.5)]),
    "afrr15":     dict(rules=[("afrr_cap_scale", 1.5)]),
    "energyonly": dict(rules=[("all_cap", 0.0)]),
    "alpha05":    dict(rules=[("alpha_scale", 0.5)]),
    "alpha025":   dict(rules=[("alpha_scale", 0.25)]),
    "deg150":     dict(repl=150000.0),
    "deg25":      dict(repl=25000.0),
    "arbonly":    dict(rules=[("arbonly",)]),
    #Lorenzo Giannuzzo: historical regime of 2023-2024: restoration energy only, no FCR for virtual
    # units, and the forward product of the UVAM (upward offers 18-22 on working days, whole MW)
    "hist":       dict(rules=[("all_cap", 0.0), ("no_fcr",)], opts=dict(uvam=UvamProduct(20000.0))),
    "hist30":     dict(rules=[("all_cap", 0.0), ("no_fcr",)], opts=dict(uvam=UvamProduct(30000.0))),
    "histnouvam": dict(rules=[("all_cap", 0.0), ("no_fcr",)]),
    #Lorenzo Giannuzzo: price-responsive FCR, linear residual demand reaching zero at 30 MW per direction
    "fcrdepth":   dict(opts=dict(fcr_depth_mw=30.0)),
    #Lorenzo Giannuzzo: day-ahead schedule on persistence forecasts, settled at the actual prices
    "persist1":   dict(forecast=1),
    "persist7":   dict(forecast=7),
}

#Lorenzo Giannuzzo: from the preliminary formulation to the reference, one change at a time (cumulative)
BRIDGE = [
    ("b0_preliminary",   dict(pooled=True, repl=25000.0, opts=dict(preliminary=True))),
    ("b1_yearly_msd",  dict(repl=25000.0, opts=dict(preliminary=True))),
    ("b2_cyclic",      dict(repl=25000.0, opts=dict(cyclic_soc=True, headroom=False, direction_mutex=True, fcr_directional=False,
                                                     tied_activation=False, sustain_eta=False, sustain_end_of_hour=False,
                                                     throughput_full=True, fcr_soc_neutral=True))),
    ("b3_degradation", dict(opts=dict(cyclic_soc=True, headroom=False, direction_mutex=True, fcr_directional=False,
                                      tied_activation=False, sustain_eta=False, sustain_end_of_hour=False,
                                      throughput_full=True, fcr_soc_neutral=True))),
    ("b4_throughput",  dict(opts=dict(cyclic_soc=True, headroom=False, direction_mutex=True, fcr_directional=False,
                                      tied_activation=False, sustain_eta=False, sustain_end_of_hour=False))),
    ("b5_tied",        dict(opts=dict(cyclic_soc=True, headroom=False, direction_mutex=True, fcr_directional=False,
                                      sustain_eta=False, sustain_end_of_hour=False))),
    ("b6_sustain",     dict(opts=dict(cyclic_soc=True, headroom=False, direction_mutex=True, fcr_directional=False))),
    ("b7_headroom",    dict(opts=dict(direction_mutex=True, fcr_directional=False))),
    ("b8_no_mutex",    dict(opts=dict(fcr_directional=False))),
    ("b9_reference",   dict()),
]


# ============================================================================
# Market data
# ============================================================================

def build_window(year, n_days, pun_paths, msd_paths, pooled=False):
    """Day-ahead prices of the year and directional reserve catalog from the MSD records of
    the same year (pooled=True reproduces the preliminary version, both years pooled)."""
    import contextlib, io
    from market_data import (load_pun_from_gme_xlsx_multi, msd_conditioned_profiles,
                                     build_overrides_from_conditioned_profiles,
                                     build_yearly_service_catalog, MarketWindow)
    start = datetime(year, 1, 1); end = start + timedelta(days=n_days)
    with contextlib.redirect_stdout(io.StringIO()):
        prices, _ = load_pun_from_gme_xlsx_multi(pun_paths, start_date=start, end_date=end, price_column="PUN")
        n_h = n_days * 24
        prices = np.asarray(prices, float)
        if len(prices) < n_h:
            prices = np.concatenate([prices, np.full(n_h - len(prices), prices[-1])])
        prices = prices[:n_h]
        cond = msd_conditioned_profiles(msd_paths, year=None if pooled else year)
        ov = build_overrides_from_conditioned_profiles(cond, start, end, directional=True)
        services = build_yearly_service_catalog(start, end, None, directional=True, **ov)
    return MarketWindow(start_date=start, end_date=end, prices_hourly=prices, services_hourly=services)


class RuledWindow:
    """Applies the market rules of a scenario to the services of each day."""

    def __init__(self, window, rules):
        self.w, self.rules = window, list(rules or [])
        self.start = window.start_date

    def is_working_day(self, day):
        return (self.start + timedelta(days=day)).weekday() < 5

    def prices(self, day):
        return list(self.w.prices_hourly[day * 24:(day + 1) * 24])

    def day(self, day):
        prices, services = self.w.slice_day(day)
        if not self.rules:
            return list(prices), services
        out = []
        for hour in services:
            new = []
            for s in hour:
                s = copy.copy(s)
                name = str(getattr(s.service_type, "value", s.service_type)).lower()
                for rule in self.rules:
                    kind = rule[0]
                    if kind == "arbonly":
                        s.award_probability = 0.0
                    elif kind == "no_fcr" and name == "fcr":
                        s.award_probability = 0.0
                    elif kind == "fcr_cap" and name == "fcr":
                        s.capacity_price = float(rule[1])
                    elif kind == "all_cap":
                        s.capacity_price = float(rule[1])
                    elif kind == "afrr_cap_scale" and name.startswith("afrr"):
                        s.capacity_price = float(s.capacity_price) * float(rule[1])
                    elif kind == "alpha_scale" and (name.startswith("afrr") or name.startswith("mfrr")):
                        s.award_probability = float(s.award_probability) * float(rule[1])
                new.append(s)
            out.append(new)
        return list(prices), out


# ============================================================================
# Runs
# ============================================================================

def make_units(counts, repl, durations=None):
    units = []
    for (label, _, p, d), n in zip(CLASSES, counts):
        dd = d if durations is None else durations[label]
        units += [BatteryUnit(p_max=p, e_cap=p * dd, c_deg=repl / (2.0 * CYCLE_LIFE), label=label) for _ in range(n)]
    return units


def solve_year(units, win, n_days, opts, forecast=0, dump=None, tag=None, mult=None):
    """Daily problems over the year; SOC carried only if the condition is not cyclic. With mult, each unit
    stands for mult[j] identical units (merged units), and the per-unit results are expanded at the end."""
    N = len(units)
    keys = None
    acc = dict(arb=np.zeros(N), deg=np.zeros(N), rev=None, band=None, uvam=np.zeros(N), net=np.zeros(N))
    gaps, abs_gap, n_days_open, runtime = [], 0.0, 0, 0.0
    soc = [opts.soc0] * N
    sched = []
    t0 = time.time()
    for day in range(n_days):
        prices, services = win.day(day)
        plan_prices = prices if not forecast else win.prices(max(day - forecast, 0))
        r = solve_day(units, plan_prices, services, opts, soc_start=soc,
                      working_day=win.is_working_day(day), keep_schedule=bool(dump) or bool(forecast))
        if keys is None:
            keys = r["keys"]
            acc["rev"] = np.zeros((N, len(keys))); acc["band"] = np.zeros((N, len(keys)))
        arb = r["arb"]
        if forecast:
            # the schedule is decided on forecasts and settled at the actual prices
            arb = (r["Pdis"] - r["Pch"]) @ np.asarray(prices, float)
        acc["arb"] += arb; acc["deg"] += r["deg"]; acc["rev"] += r["rev"]; acc["band"] += r["band"]
        acc["uvam"] += r["uvam"]
        acc["net"] += arb - r["deg"] + r["rev"].sum(axis=1) + r["uvam"]
        if r["gap"] is not None:
            gaps.append(r["gap"])
            if r["bound"] is not None:
                abs_gap += abs(r["bound"] - r["objective"])
            if r["gap"] > opts.gap * 1.5:
                n_days_open += 1
        runtime += r["runtime"]
        if not opts.cyclic_soc and not opts.preliminary:
            soc = list(r["soc_end"])
        elif opts.preliminary:
            soc = list(r["soc_end"])
        if dump:
            sched.append(dict(Pch=r["Pch"], Pdis=r["Pdis"], SOC=r["SOC"], R=r["R"], alpha=r["alpha"],
                              beta=r["beta"], ccap=r["ccap"], cen=r["cen"], prices=np.asarray(prices, float)))
        if tag and (day + 1) % 61 == 0:
            print(f"    [{tag}] day {day + 1}/{n_days} | {(time.time() - t0) / (day + 1):.2f} s/day", flush=True)
    if mult is not None:
        m = np.asarray(mult, float)
        for name in ("arb", "deg", "uvam", "net"):
            acc[name] = np.repeat(acc[name] / m, mult)
        for name in ("rev", "band"):
            acc[name] = np.repeat(acc[name] / m[:, None], mult, axis=0)
    out = dict(keys=keys, gaps=gaps, abs_gap=abs_gap, days_open=n_days_open, runtime=runtime, **acc)
    if dump:
        np.savez_compressed(dump, keys=np.array(keys), labels=np.array([u.label for u in units]),
                            p_max=np.array([u.p_max for u in units]), e_cap=np.array([u.e_cap for u in units]),
                            eta=np.array([u.eta for u in units]), c_deg=np.array([u.c_deg for u in units]),
                            mult=np.array(mult if mult is not None else [1] * N),
                            **{f"{name}": np.stack([d[name] for d in sched]) for name in sched[0]})
    return out


def per_class(values, units):
    return {c[0]: float(np.sum([v for v, u in zip(values, units) if u.label == c[0]])) for c in CLASSES}


def allocate(agg, units, solo):
    """Rated power, band supplied, reserve floor and pass-through, per unit."""
    base = agg["arb"] - agg["deg"]
    pool = float(agg["rev"].sum() + agg["uvam"].sum())
    p = np.array([u.p_max for u in units]); w_pow = p / p.sum()
    band = agg["band"].sum(axis=1)
    w_band = band / band.sum() if band.sum() > 0 else w_pow
    joint = float(agg["net"].sum())
    return {
        "power": base + pool * w_pow,
        "band": base + pool * w_band,
        "floor": solo + (joint - solo.sum()) * w_band,
        "passthru": agg["net"].copy(),
    }


def run_scenario(name, cfg, year, a, base_window, pooled_window):
    win = RuledWindow(pooled_window if cfg.get("pooled") else base_window, cfg.get("rules"))
    opts = replace(ModelOptions(gap=a.gap), **cfg.get("opts", {}))
    repl = cfg.get("repl", REPLACEMENT_EUR_PER_MWH)
    fc = cfg.get("forecast", 0)
    tag = f"{name} {year}"
    print(f"\n=== {tag} ===", flush=True)
    t0 = time.time()
    # standalone and threshold-free: one unit per class
    sa, tf = {}, {}
    dumping = a.dump and name in ("ref", "b9_reference")
    for k, (label, n, p, d) in enumerate(CLASSES):
        cnt = [0, 0, 0]; cnt[k] = 1
        u1 = make_units(cnt, repl)
        sa[label] = solve_year(u1, win, a.days, opts, forecast=fc, tag=f"{tag} standalone {label}",
                               dump=os.path.join(a.outdir, f"schedules_{name}_{year}_standalone_{label}.npz") if dumping else None)
        tf[label] = solve_year(u1, win, a.days, replace(opts, min_bid=0.0, granularity=0.0, uvam=None, fcr_depth_mw=0.0),
                               forecast=fc, tag=f"{tag} threshold-free {label}") if not a.no_tf else None
    units = make_units(N_TOT, repl)
    dump = os.path.join(a.outdir, f"schedules_{name}_{year}.npz") if dumping else None
    #Lorenzo Giannuzzo: the identical units of each class are merged (same value, see super_units), so that each
    # unit of a class receives the same schedule, one n-th of the merged one
    agg = solve_year(super_units(N_TOT, repl), win, a.days, opts, forecast=fc, dump=dump, tag=f"{tag} aggregated",
                     mult=list(N_TOT))
    solo = np.array([sa[u.label]["net"][0] for u in units])
    alloc = allocate(agg, units, solo)
    S, A = solo.sum(), agg["net"].sum()
    row = dict(scenario=name, year=year, standalone_keur=S / 1e3, aggregated_keur=A / 1e3,
               delta_keur=(A - S) / 1e3, delta_pct=100 * (A - S) / S if S else float("nan"),
               agg_arb_keur=agg["arb"].sum() / 1e3, agg_reserve_keur=(agg["rev"].sum() + agg["uvam"].sum()) / 1e3,
               agg_deg_keur=agg["deg"].sum() / 1e3,
               agg_gap_max_pct=100 * max(agg["gaps"]) if agg["gaps"] else float("nan"),
               agg_abs_gap_keur=agg["abs_gap"] / 1e3, agg_days_open=agg["days_open"],
               solo_abs_gap_keur=sum(sa[c[0]]["abs_gap"] * c[1] for c in CLASSES) / 1e3,
               minutes=(time.time() - t0) / 60)
    for j, k in enumerate(agg["keys"]):
        row[f"agg_rev_{k}_keur"] = agg["rev"][:, j].sum() / 1e3
        row[f"solo_rev_{k}_keur"] = sum(sa[c[0]]["rev"][0, j] * c[1] for c in CLASSES) / 1e3
    if agg["uvam"].sum() > 0:
        row["agg_uvam_keur"] = agg["uvam"].sum() / 1e3
    for c in CLASSES:
        row[f"sa_{c[0]}_unit_keur"] = sa[c[0]]["net"][0] / 1e3
        if tf[c[0]] is not None:
            row[f"tf_{c[0]}_unit_keur"] = tf[c[0]]["net"][0] / 1e3
    if tf["small"] is not None:
        row["value_of_admission_keur_per_mw_year"] = (tf["small"]["net"][0] - sa["small"]["net"][0]) / 1e3 / CLASSES[0][2]
    for rule, x in alloc.items():
        lose = [i for i in range(len(units)) if x[i] < solo[i] - 1e-6]
        row[f"{rule}_units_below"] = len(lose)
        row[f"{rule}_worst_pct"] = min((100 * (x[i] / solo[i] - 1) for i in lose), default=0.0)
        pc, ps = per_class(x, units), per_class(solo, units)
        for c in CLASSES:
            row[f"{rule}_{c[0]}_gain_pct"] = 100 * (pc[c[0]] / ps[c[0]] - 1) if ps[c[0]] else float("nan")
            row[f"{rule}_{c[0]}_unit_keur"] = pc[c[0]] / c[1] / 1e3
            row[f"{rule}_{c[0]}_share_pct"] = 100 * (pc[c[0]] - ps[c[0]]) / (A - S) if A != S else float("nan")
    row["passthru_small_gain_keur_per_mw_year"] = (per_class(alloc["passthru"], units)["small"] - per_class(solo, units)["small"]) / 1e3 / (CLASSES[0][1] * CLASSES[0][2])
    # per-class detail of the two configurations (Table 4 of the paper)
    for c in CLASSES:
        idx = [i for i, u in enumerate(units) if u.label == c[0]]
        row[f"agg_{c[0]}_arb_keur"] = agg["arb"][idx].sum() / 1e3
        row[f"agg_{c[0]}_deg_keur"] = agg["deg"][idx].sum() / 1e3
        row[f"agg_{c[0]}_band_mwh"] = agg["band"][idx].sum()
        row[f"sa_{c[0]}_arb_keur"] = sa[c[0]]["arb"][0] * c[1] / 1e3
        row[f"sa_{c[0]}_deg_keur"] = sa[c[0]]["deg"][0] * c[1] / 1e3
        row[f"sa_{c[0]}_band_mwh"] = sa[c[0]]["band"][0].sum() * c[1]
    #Lorenzo Giannuzzo: per-unit allocations, standalone and threshold-free values, for the stability tests
    with open(os.path.join(a.outdir, f"units_{name}_{year}.json"), "w") as fh:
        json.dump(dict(labels=[u.label for u in units], standalone=solo.tolist(),
                       threshold_free={c[0]: (tf[c[0]]["net"][0] if tf[c[0]] is not None else None) for c in CLASSES},
                       allocations={r: x.tolist() for r, x in alloc.items()},
                       aggregated=float(A), keys=list(agg["keys"]),
                       agg_rev_by_unit=agg["rev"].tolist(), agg_band_by_unit=agg["band"].tolist(),
                       agg_arb=agg["arb"].tolist(), agg_deg=agg["deg"].tolist(),
                       tolerance=dict(aggregated=agg["abs_gap"], standalone={c[0]: sa[c[0]]["abs_gap"] for c in CLASSES},
                                      threshold_free={c[0]: (tf[c[0]]["abs_gap"] if tf[c[0]] is not None else None) for c in CLASSES})),
                  fh, indent=1)
    print(f"  value of aggregation {row['delta_keur']:.1f} kEUR ({row['delta_pct']:.1f}%) | value of admission "
          f"{row.get('value_of_admission_keur_per_mw_year', float('nan')):.1f} kEUR/(MW year) | units below standalone: "
          + ", ".join(f"{r} {row[f'{r}_units_below']}" for r in alloc) + f" | {row['minutes']:.1f} min", flush=True)
    return row


# ============================================================================
# Exact daily game of the twenty units
# ============================================================================

def compositions():
    return [c for c in product(*(range(n + 1) for n in N_TOT))]


def clears(counts, offers, min_bid, gran=0.0):
    """True if the aggregate of the threshold-free offers of a composition is, in every product
    and hour, either null or not below the minimum bid (and a multiple of the granularity)."""
    agg = sum(n * offers[c[0]] for n, c in zip(counts, CLASSES))
    pos = agg > 1e-6
    if np.any(pos & (agg < min_bid - 1e-6)):
        return False
    if gran > 0:
        r = np.mod(agg[pos] + 1e-7, gran)
        if np.any((r > 2e-6) & (r < gran - 2e-6)):
            return False
    return True


def super_units(counts, repl):
    """The identical units of each class merged into one unit of the summed power and energy: since every
    constraint of a unit is linear and homogeneous in its power and energy, and the minimum bid acts only on the
    sum of the offers, a schedule of the merged unit divided by the number of units is feasible for each of them,
    and the sum of their schedules is feasible for the merged unit, so that the two problems have the same value."""
    return [BatteryUnit(p_max=n * p, e_cap=n * p * d, c_deg=repl / (2.0 * CYCLE_LIFE), label=label)
            for (label, _, p, d), n in zip(CLASSES, counts) if n > 0]


_G = {}


def _game_init(win, opts, repl, threads):
    _G.update(win=win, opts=opts, repl=repl)
    if threads:
        os.environ["MILP_THREADS"] = str(threads)


def game_day(day):
    """Values of all compositions on one day (days are independent under the cyclic SOC condition)."""
    win, opts, repl = _G["win"], _G["opts"], _G["repl"]
    prices, services = win.day(day)
    wd = win.is_working_day(day)
    g, sa, offers = {}, {}, {}
    out = dict(day=day, v={}, solved=[], tol={})
    for k, (label, n, p, d) in enumerate(CLASSES):
        cnt = [0, 0, 0]; cnt[k] = 1
        u1 = make_units(cnt, repl)
        tf_opts = replace(opts, min_bid=0.0, granularity=0.0, uvam=None, fcr_depth_mw=0.0)
        rt = solve_day(u1, prices, services, tf_opts, working_day=wd)
        g[label] = rt["net"][0]
        floor = g[label] - max(1e-7 * abs(g[label]), 1e-6)
        rb = solve_day(u1, prices, services, replace(tf_opts, tiebreak_floor=floor, gap=1e-9), working_day=wd,
                       keep_schedule=True)
        offers[label] = rb["R"][:, 0, :]
        rs = solve_day(u1, prices, services, opts, working_day=wd)
        sa[label] = rs["net"][0]
        if rs["bound"] is not None:
            out["tol"][str(tuple(1 if j == k else 0 for j in range(3)))] = abs(rs["bound"] - rs["objective"])
    out["g"] = g
    for c in compositions():
        if sum(c) == 0:
            continue
        gc = sum(n * g[cl[0]] for n, cl in zip(c, CLASSES))
        if sum(c) == 1:
            v = sa[CLASSES[[i for i, n in enumerate(c) if n][0]][0]]
        elif clears(c, offers, opts.min_bid, opts.granularity) and opts.uvam is None and opts.fcr_depth_mw == 0:
            v = gc
        else:
            r = solve_day(super_units(c, repl), prices, services, opts, working_day=wd)
            v = float(r["net"].sum()); out["solved"].append(str(c))
            if r["bound"] is not None:
                out["tol"][str(c)] = abs(r["bound"] - r["objective"])
        out["v"][str(c)] = [v, gc]
    return out


def run_game(year, a, base_window, scen="ref"):
    """Exact daily game. Every composition is valued on every day, as the sum of the threshold-free values of its
    members when their least-band threshold-free schedules clear the minimum bid, and by solving it (merged units)
    otherwise. Days already in the checkpoint file are skipped, so an interrupted run can be resumed."""
    cfg = SCENARIOS[scen]
    win = RuledWindow(base_window, cfg.get("rules"))
    opts = replace(ModelOptions(gap=a.gap), **cfg.get("opts", {}))
    repl = cfg.get("repl", REPLACEMENT_EUR_PER_MWH)
    ck = os.path.join(a.outdir, f"game_{scen}_{year}_days.jsonl")
    done = {}
    if os.path.exists(ck):
        for line in open(ck):
            try:
                rec = json.loads(line); done[rec["day"]] = rec
            except Exception:
                pass
    todo = [d for d in range(a.days) if d not in done]
    print(f"  [game {scen} {year}] {len(done)} days already done, {len(todo)} to go, {a.workers} worker(s)", flush=True)
    t0 = time.time()
    with open(ck, "a") as fh:
        if a.workers > 1:
            import multiprocessing as mp
            with mp.Pool(a.workers, initializer=_game_init, initargs=(win, opts, repl, 1)) as pool:
                for i, rec in enumerate(pool.imap_unordered(game_day, todo)):
                    fh.write(json.dumps(rec) + "\n"); fh.flush(); done[rec["day"]] = rec
                    print(f"  [game {scen} {year}] day {rec['day'] + 1} | {i + 1}/{len(todo)} | {(time.time() - t0) / 60:.1f} min | "
                          f"solved {len(rec['solved'])}", flush=True)
        else:
            _game_init(win, opts, repl, None)
            for i, d in enumerate(todo):
                rec = game_day(d)
                fh.write(json.dumps(rec) + "\n"); fh.flush(); done[d] = rec
                print(f"  [game {scen} {year}] day {d + 1} | {i + 1}/{len(todo)} | {(time.time() - t0) / 60:.1f} min | "
                      f"solved {len(rec['solved'])}", flush=True)
    comps = [c for c in compositions() if sum(c) > 0]
    vals = {str(c): 0.0 for c in comps}; short = {str(c): 0.0 for c in comps}
    nsol = {str(c): 0 for c in comps}; tol = {str(c): 0.0 for c in comps}
    gsum = {c[0]: 0.0 for c in CLASSES}
    for rec in done.values():
        for c, (v, gc) in rec["v"].items():
            vals[c] += v; short[c] += gc - v
        for c in rec["solved"]:
            nsol[c] += 1
        for c, t in rec["tol"].items():
            tol[c] += t
        for l, x in rec["g"].items():
            gsum[l] += x
    vals[str((0, 0, 0))] = 0.0
    fn = os.path.join(a.outdir, f"game_{scen}_{year}.json")
    with open(fn, "w") as fh2:
        json.dump(dict(values=vals, days_solved=nsol, shortfall=short, tolerance=tol, threshold_free_per_unit=gsum,
                       n_days=len(done)), fh2, indent=1)
    print(f"  game written to {fn}", flush=True)
    return vals


# ============================================================================
# Value of admission by duration
# ============================================================================

def run_durations(year, a, base_window, scen_list):
    rows = []
    for scen in scen_list:
        cfg = SCENARIOS[scen]
        win = RuledWindow(base_window, cfg.get("rules"))
        opts = replace(ModelOptions(gap=a.gap), **cfg.get("opts", {}))
        repl = cfg.get("repl", REPLACEMENT_EUR_PER_MWH)
        for d in a.durations:
            u = [BatteryUnit(p_max=0.5, e_cap=0.5 * d, c_deg=repl / (2 * CYCLE_LIFE), label="small")]
            sa = solve_year(u, win, a.days, opts)
            tf = solve_year(u, win, a.days, replace(opts, min_bid=0.0, granularity=0.0, uvam=None))
            row = dict(scenario=scen, year=year, duration_h=d, standalone_keur_per_mw=sa["net"][0] / 1e3 / 0.5,
                       threshold_free_keur_per_mw=tf["net"][0] / 1e3 / 0.5,
                       value_of_admission_keur_per_mw_year=(tf["net"][0] - sa["net"][0]) / 1e3 / 0.5)
            print(f"  [durations {scen} {year}] {d} h: value of admission {row['value_of_admission_keur_per_mw_year']:.1f} kEUR/(MW year)", flush=True)
            rows.append(row)
    return rows


# ============================================================================

def write_csv(rows, fn):
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("scenario", "year", "duration_h"), k))
    with open(fn, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=[])
    ap.add_argument("--game", nargs="*", default=None, help="exact daily game (default scenario: ref)")
    ap.add_argument("--comps", default=None)
    ap.add_argument("--comps-scenario", default="ref")
    ap.add_argument("--durations", nargs="*", type=float, default=None)
    ap.add_argument("--durations-scenarios", nargs="*", default=["ref", "afrr05", "afrr15", "energyonly", "hist"])
    ap.add_argument("--bridge", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes for the game (one Gurobi thread each)")
    ap.add_argument("--dump", action="store_true", help="save the daily schedules of the reference (for the replay)")
    ap.add_argument("--no-tf", action="store_true", help="skip the threshold-free runs")
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--gap", type=float, default=1e-4)
    ap.add_argument("--pun-xlsx", nargs="+", default=None)
    ap.add_argument("--msd-xlsx", nargs="+", default=None)
    ap.add_argument("--outdir", default="../results/new")
    a = ap.parse_args()
    import glob
    if not a.pun_xlsx:
        a.pun_xlsx = [os.path.join("..", "data", "raw", "20230101_20231231_PUN.xlsx"), os.path.join("..", "data", "raw", "20240101_20241231_PUN.xlsx")]
    if not a.msd_xlsx:
        a.msd_xlsx = sorted(glob.glob(os.path.join("..", "data", "raw", "EsitiMSD*Nord*.xls*")))
    os.makedirs(a.outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows, drows = [], []
    for year in a.years:
        base = build_window(year, a.days, a.pun_xlsx, a.msd_xlsx)
        pooled = build_window(year, a.days, a.pun_xlsx, a.msd_xlsx, pooled=True) if a.bridge else None
        for scen in a.scenarios:
            rows.append(run_scenario(scen, SCENARIOS[scen], year, a, base, pooled))
            write_csv(rows, os.path.join(a.outdir, f"summary_{stamp}.csv"))
        if a.bridge:
            for name, cfg in BRIDGE:
                rows.append(run_scenario(name, cfg, year, a, base, pooled))
                write_csv(rows, os.path.join(a.outdir, f"summary_{stamp}.csv"))
        if a.game is not None:
            for scen in (a.game or ["ref"]):
                run_game(year, a, base, scen)
        if a.comps:
            cfg = SCENARIOS[a.comps_scenario]
            win = RuledWindow(base, cfg.get("rules"))
            opts = replace(ModelOptions(gap=a.gap), **cfg.get("opts", {}))
            out = {}
            for t in a.comps.split(";"):
                c = tuple(int(x) for x in t.split(","))
                r = solve_year(super_units(c, cfg.get("repl", REPLACEMENT_EUR_PER_MWH)), win, a.days, opts,
                               tag=f"{a.comps_scenario} {year} {c}")
                out[str(c)] = dict(value=float(r["net"].sum()), tolerance=r["abs_gap"], days_open=r["days_open"])
                print(f"  {a.comps_scenario} {year} {c}: v = {r['net'].sum() / 1e3:.2f} kEUR", flush=True)
            with open(os.path.join(a.outdir, f"comps_{a.comps_scenario}_{year}_{stamp}.json"), "w") as fh:
                json.dump(out, fh, indent=1)
        if a.durations:
            drows += run_durations(year, a, base, a.durations_scenarios)
            write_csv(drows, os.path.join(a.outdir, f"durations_{stamp}.csv"))
    print(f"\nresults in {a.outdir}")


if __name__ == "__main__":
    main()
