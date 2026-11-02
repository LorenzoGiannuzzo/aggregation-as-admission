"""
Rebuilds the values of Tables 4 to 10 of the paper from the files in ../results and
writes one CSV per table in this folder (money in kEUR, rounded as in the paper).

    python make_tables.py
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
CL = ["small", "medium", "large"]
N = {"small": 8, "medium": 8, "large": 4}
YEARS = (2023, 2024)


def res(name):
    return os.path.join(RES, name)


def save(df, name):
    fn = os.path.join(HERE, name)
    df.to_csv(fn, index=False)
    print(f"\n{name}\n{df.to_string(index=False)}")


SC = pd.read_csv(res("scenarios.csv"))
POST = json.load(open(res("post_ref.json")))


def row(scen, y):
    return SC[(SC.scenario == scen) & (SC.year == y)].iloc[0]


#Lorenzo Giannuzzo: Table 4, annual results of each class, standalone (SA) and aggregated (AG)
rows = []
for y in YEARS:
    r = row("ref", y); u = json.load(open(res(f"units_ref_{y}.json"))); lab = u["labels"]
    tot = dict(sa_arb=0, ag_arb=0, sa_res=0, ag_res=0, sa_deg=0, ag_deg=0, sa_net=0, ag_net=0)
    for c in CL:
        sa_net = r[f"sa_{c}_unit_keur"] * N[c]; sa_arb = r[f"sa_{c}_arb_keur"]; sa_deg = r[f"sa_{c}_deg_keur"]
        ag_arb = r[f"agg_{c}_arb_keur"]; ag_deg = r[f"agg_{c}_deg_keur"]
        ag_res = sum(sum(x) for x, l in zip(u["agg_rev_by_unit"], lab) if l == c) / 1e3
        d = dict(sa_arb=sa_arb, ag_arb=ag_arb, sa_res=sa_net - sa_arb + sa_deg, ag_res=ag_res, sa_deg=-sa_deg, ag_deg=-ag_deg,
                 sa_net=sa_net, ag_net=ag_arb + ag_res - ag_deg)
        rows.append(dict(year=y, cls=c, **{k: round(v, 1) + 0.0 for k, v in d.items()}))
        for k in tot:
            tot[k] += d[k]
    rows.append(dict(year=y, cls="portfolio", **{k: round(v, 1) for k, v in tot.items()}))
save(pd.DataFrame(rows), "table4_class_results.csv")

#Lorenzo Giannuzzo: Table 5, value of admission of a 0.5 MW unit by duration and regime
du = pd.read_csv(res("durations.csv"))
t5 = du.pivot_table(index="duration_h", columns=["scenario", "year"], values="value_of_admission_keur_per_mw_year").round(1)
t5.columns = [f"{s}_{y}" for s, y in t5.columns]
save(t5.reset_index(), "table5_value_of_admission.csv")

#Lorenzo Giannuzzo: Table 6, profit per unit under the four rules and the Shapley value, with the gain over standalone
rows = []
for y in YEARS:
    r = row("ref", y); sh = POST[str(y)]["shapley_per_unit_keur"]
    for c in CL:
        sa = r[f"sa_{c}_unit_keur"]; d = dict(year=y, cls=c, standalone=round(sa, 2))
        for rule in ("power", "band", "floor", "passthru"):
            v = r[f"{rule}_{c}_unit_keur"]; d[rule] = round(v, 2); d[f"{rule}_gain_pct"] = round(100 * (v / sa - 1), 1)
        d["shapley"] = round(sh[c], 2); d["shapley_gain_pct"] = round(100 * (sh[c] / sa - 1), 1)
        rows.append(d)
save(pd.DataFrame(rows), "table6_allocations.csv")

#Lorenzo Giannuzzo: Table 7, game of the twenty units, profit per unit
rows = []
for y in YEARS:
    g = json.load(open(res(f"game_ref_{y}.json"))); v = {k: x / 1e3 for k, x in g["values"].items()}
    p = POST[str(y)]; core = p["core"]
    for c, pair, pool, n in (("small", "(2, 0, 0)", "(8, 0, 0)", 8), ("medium", "(0, 2, 0)", "(0, 8, 0)", 8), ("large", "(0, 0, 2)", "(0, 0, 4)", 4)):
        lo, hi = core[f"{c}_interval"]; plo, phi_ = p["unit_core_projection_keur"][c]
        rows.append(dict(year=y, cls=c, pair=round(v[pair] / 2, 2), pool=round(v[pool] / n, 2),
                         threshold_free=round(g["threshold_free_per_unit"][c] / 1e3, 2),
                         core_equal_treatment=f"{lo:.2f}-{hi:.2f}", core_single_unit=f"{plo:.2f}-{phi_:.2f}",
                         nucleolus=round(core["nucleolus_keur"][c], 2), shapley=round(p["shapley_per_unit_keur"][c], 2)))
save(pd.DataFrame(rows), "table7_game.csv")

#Lorenzo Giannuzzo: Table 8, largest excess of each allocation over all proper coalitions
rows = []
for name, key in (("By rated power", "power"), ("By band supplied", "band"), ("Reserve floor", "floor"), ("Pass-through", "passthru")):
    d = dict(allocation=name)
    for y in YEARS:
        rr = POST[str(y)]["rules"][key]; d[f"excess_{y}"] = round(rr["max_excess_keur"], 3); d[f"coalition_{y}"] = rr["blocked_by"]
    rows.append(d)
d = dict(allocation="Shapley value")
for y in YEARS:
    d[f"excess_{y}"] = round(POST[str(y)]["shapley_max_excess_keur"], 1); d[f"coalition_{y}"] = POST[str(y)]["shapley_blocked_by"]
rows.append(d)
d = dict(allocation="Nucleolus (least-core value)")
for y in YEARS:
    d[f"excess_{y}"] = round(-POST[str(y)]["core"]["least_core_margin_keur"], 3); d[f"coalition_{y}"] = "several"
rows.append(d)
save(pd.DataFrame(rows), "table8_excess.csv")

#Lorenzo Giannuzzo: Table 9, sensitivity of the value of aggregation and of the value of admission
VARIANTS = [("ref", "Reference"), ("minbid05", "Minimum bid of 0.5 MW"), ("minbid025", "Minimum bid of 0.25 MW"),
            ("gran1", "Aggregate offer in whole MW"), ("fcrsym", "Symmetric FCR"),
            ("fcr25", "Symmetric FCR price of 25 EUR/(MW h), 12.5 per direction"),
            ("fcr10", "Symmetric FCR price of 10 EUR/(MW h), 5 per direction"), ("nofcr", "No FCR"),
            ("fcrdepth", "FCR price falling to zero at 30 MW"), ("afrr05", "aFRR capacity price x 0.5"),
            ("afrr15", "aFRR capacity price x 1.5"), ("energyonly", "No capacity price (energy only)"),
            ("hist", "Historical regime, UVAM at 20 kEUR/(MW year)"), ("hist30", "Historical regime, UVAM at 30 kEUR/(MW year)"),
            ("alpha05", "aFRR and mFRR award probability x 0.5"), ("alpha025", "aFRR and mFRR award probability x 0.25"),
            ("deg150", "Replacement cost of 150 EUR/kWh"), ("deg25", "Replacement cost of 25 EUR/kWh"),
            ("phi047", "FCR activation of 0.024 per direction (+-10 mHz deadband)"),
            ("persist1", "Persistence forecast, previous day"), ("persist7", "Persistence forecast, previous week")]
rows = []
for key, name in VARIANTS:
    d = dict(variant=name)
    for y in YEARS:
        r = row(key, y)
        d[f"value_of_aggregation_{y}"] = f"{r.delta_keur:.1f} ({r.delta_pct:+.1f}%)"
        d[f"value_of_admission_{y}"] = round(r.value_of_admission_keur_per_mw_year, 1)
        d[f"units_below_{y}"] = f"{int(r.power_units_below)}/{int(r.band_units_below)}/{int(r.passthru_units_below)}"
    rows.append(d)
save(pd.DataFrame(rows), "table9_sensitivity.csv")

#Lorenzo Giannuzzo: Table 10, bridge from the preliminary to the reference formulation
STEPS = [("b0_preliminary", "Preliminary formulation"), ("b1_yearly_msd", "Year-specific MSD statistics"), ("b2_cyclic", "Cyclic SOC"),
         ("b3_degradation", "Degradation at 100 EUR/kWh"), ("b4_throughput", "Degradation on expected energy"),
         ("b5_tied", "Activation tied to the band"), ("b6_sustain", "Sustain with losses"), ("b7_headroom", "Headroom per direction"),
         ("b8_no_mutex", "No up/down exclusion"), ("b9_reference", "Directional FCR (reference)")]
rows = []; prev = {}
for key, name in STEPS:
    d = dict(step=name)
    for y in YEARS:
        r = row(key, y); v = float(r.delta_keur)
        d[f"value_of_aggregation_{y}"] = round(v, 1); d[f"change_{y}"] = round(v - prev[y], 1) if y in prev else ""
        d[f"value_of_admission_{y}"] = round(r.value_of_admission_keur_per_mw_year, 1); prev[y] = v
    rows.append(d)
save(pd.DataFrame(rows), "table10_bridge.csv")
