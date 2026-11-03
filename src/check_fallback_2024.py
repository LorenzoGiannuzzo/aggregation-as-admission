"""
Effect of the 52 hours of 2024 in which the MSD records of the Nord zone show no upward
acceptance and market_data.py assigns to the upward products its default profile
(Section 2.1 and Section S1 of the Supplementary Material of the paper).

Solves the reference configurations of 2024 from the processed inputs twice, as built
("base") and with a null upward award probability in those hours ("fix"), and writes the
value of aggregation and the value of admission of both runs to results/fallback_2024.json.

    python check_fallback_2024.py            # a few minutes per run with Gurobi
"""
import json
import os
import sys
import types
from datetime import datetime

import numpy as np
import pandas as pd

import market_data as md
import experiments as ex

HERE = os.path.dirname(os.path.abspath(__file__))
Y = 2024


def run(mode, d, outdir):
    #Lorenzo Giannuzzo: the fallback hours carry the default upward profile of aFRR (0.65 at 250 EUR/MWh)
    fb = ((d.afrr_up_energy_price_eur_mwh == 250.0) & (d.afrr_up_award_probability == 0.65)).to_numpy()
    ov = {}
    for k in ("afrr_up", "afrr_dn", "mfrr_up", "mfrr_dn"):
        ov[f"{k}_award_overrides"] = d[f"{k}_award_probability"].to_numpy().copy()
        ov[f"{k}_energy_overrides"] = d[f"{k}_energy_price_eur_mwh"].to_numpy().copy()
    start = datetime(Y, 1, 1)
    services = md.build_yearly_service_catalog(start, datetime(Y, 12, 31), None, directional=True, **ov)
    if mode == "fix":
        for h in np.where(fb)[0]:
            for s in services[h]:
                if str(getattr(s.service_type, "value", s.service_type)).lower() in ("afrr_up", "mfrr_up"):
                    s.award_probability = 0.0
    win = md.MarketWindow(start_date=start, end_date=datetime(Y, 12, 31),
                          prices_hourly=d.day_ahead_price_eur_mwh.to_numpy(), services_hourly=services)
    a = types.SimpleNamespace(gap=1e-4, dump=False, no_tf=False, days=365, outdir=outdir)
    row = ex.run_scenario("ref", ex.SCENARIOS["ref"], Y, a, win, None)
    return int(fb.sum()), {k: float(row[k]) for k in ("standalone_keur", "aggregated_keur", "delta_keur", "delta_pct",
                                                      "value_of_admission_keur_per_mw_year")}


def main():
    d = pd.read_csv(os.path.join(HERE, "..", "data", "processed", f"market_inputs_{Y}.csv"), parse_dates=["date"])
    outdir = os.path.join(HERE, "..", "results", "fallback_runs")
    os.makedirs(outdir, exist_ok=True)
    res = {}
    for mode in ("base", "fix"):
        n, res[mode] = run(mode, d, outdir)
        print(mode, res[mode])
    res["fallback_hours"] = n
    with open(os.path.join(HERE, "..", "results", "fallback_2024.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
