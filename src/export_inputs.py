"""
Exports the hourly market inputs of the reference runs (schedules_ref_<year>.npz)
to data/processed/market_inputs_<year>.csv: day-ahead price and, for each of the
six reserve products, award probability, expected activation, capacity price and
energy price, as entered in the scheduling model.

    python export_inputs.py --outdir ../results --dest ../data/processed
"""
import argparse
import csv
import os
from datetime import date, timedelta

import numpy as np

FCR_ACTIVATION = 0.088 / 2  # expected FCR activation per direction entering the model (fcr_phi of scheduling_milp.py, halved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="../results")
    ap.add_argument("--dest", default="../data/processed")
    ap.add_argument("--years", nargs="*", type=int, default=[2023, 2024])
    a = ap.parse_args()
    os.makedirs(a.dest, exist_ok=True)
    for y in a.years:
        d = np.load(os.path.join(a.outdir, f"schedules_ref_{y}.npz"))
        keys = [str(k) for k in d["keys"]]
        cols = ["date", "hour", "day_ahead_price_eur_mwh"]
        for k in keys:
            cols += [f"{k}_award_probability", f"{k}_expected_activation", f"{k}_capacity_price_eur_mw_h",
                     f"{k}_energy_price_eur_mwh"]
        fn = os.path.join(a.dest, f"market_inputs_{y}.csv")
        with open(fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for day in range(d["prices"].shape[0]):
                for t in range(24):
                    row = [(date(y, 1, 1) + timedelta(days=day)).isoformat(), t, round(float(d["prices"][day, t]), 4)]
                    for j, k in enumerate(keys):
                        vals = [float(d[x][day, j, t]) for x in ("alpha", "beta", "ccap", "cen")]
                        #Lorenzo Giannuzzo: FCR enters the model with the activation fcr_phi/2 per direction, its expected
                        # energy being settled at the day-ahead price of the hour (fcr_e of scheduling_milp.py)
                        if k.startswith("fcr"):
                            vals[1], vals[3] = FCR_ACTIVATION, float(d["prices"][day, t])
                        row += [round(v, 6) for v in vals]
                    w.writerow(row)
        print(f"{y} -> {fn}")


if __name__ == "__main__":
    main()
