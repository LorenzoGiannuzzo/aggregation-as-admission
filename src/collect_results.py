"""
Merges the summary_<timestamp>.csv and durations_<timestamp>.csv files written by
experiments.py into scenarios.csv and durations.csv, keeping for every
(scenario, year) the most recent row. The figure and table scripts read the merged files.

    python collect_results.py --outdir ../results
"""
import argparse
import csv
import glob
import os


def merge(pattern, keys, out_fn):
    rows, cols = {}, []
    for fn in sorted(glob.glob(pattern)):          # timestamps sort chronologically
        for r in csv.DictReader(open(fn, newline="")):
            rows[tuple(r[k] for k in keys)] = r
            cols += [c for c in r if c not in cols]
    lead = [c for c in keys if c in cols]
    cols = lead + sorted(c for c in cols if c not in lead)
    with open(out_fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    print(f"{len(rows)} rows -> {out_fn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="../results")
    a = ap.parse_args()
    merge(os.path.join(a.outdir, "summary_*.csv"), ("scenario", "year"), os.path.join(a.outdir, "scenarios.csv"))
    merge(os.path.join(a.outdir, "durations_*.csv"), ("scenario", "year", "duration_h"), os.path.join(a.outdir, "durations.csv"))


if __name__ == "__main__":
    main()
