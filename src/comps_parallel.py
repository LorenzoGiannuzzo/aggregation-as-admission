"""
Annual value of given compositions (as --comps of experiments.py), with the days solved in parallel.

Under the reference formulation the state of charge is cyclic (same value at the start and at the end of every
day), so the days are independent and can be solved by separate processes. Each process runs one Gurobi model
at a time with --threads threads, and every solved day is appended to a checkpoint file, so an interrupted run
resumes from where it stopped. The output JSON has the same fields as the comps file of experiments.py.

Run from the src folder, for example:

    python comps_parallel.py --comps "1,0,2;0,8,4" --comps-scenario gran1 --year 2023 \
        --workers 8 --threads 2 --pun-xlsx $PUN --msd-xlsx $MSD --outdir $OUT
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import replace
from datetime import datetime

import experiments as R

_W = {}


#Lorenzo Giannuzzo: each worker keeps the market window, the options and the merged units of every composition
def _init(win, opts, units, threads):
    _W.update(win=win, opts=opts, units=units)
    os.environ["MILP_THREADS"] = str(threads)


def _solve(task):
    comp, day = task
    win, opts = _W["win"], _W["opts"]
    prices, services = win.day(day)
    t0 = time.time()
    r = R.solve_day(_W["units"][comp], prices, services, opts, working_day=win.is_working_day(day))
    tol = abs(r["bound"] - r["objective"]) if r["bound"] is not None else 0.0
    is_open = r["gap"] is not None and r["gap"] > opts.gap * 1.5
    return dict(comp=comp, day=day, net=float(r["net"].sum()), tol=float(tol), open=bool(is_open),
                gap=r["gap"], sec=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comps", required=True, help='e.g. "1,0,2;0,8,4"')
    ap.add_argument("--comps-scenario", default="ref")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--gap", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--threads", type=int, default=2, help="Gurobi threads per process")
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

    cfg = R.SCENARIOS[a.comps_scenario]
    base = R.build_window(a.year, a.days, a.pun_xlsx, a.msd_xlsx)
    win = R.RuledWindow(base, cfg.get("rules"))
    opts = replace(R.ModelOptions(gap=a.gap), **cfg.get("opts", {}))
    if not opts.cyclic_soc or opts.preliminary:
        raise SystemExit("the days are not independent in this scenario (SOC carried over): use experiments.py")
    comps = [tuple(int(x) for x in t.split(",")) for t in a.comps.split(";")]
    units = {str(c): R.super_units(c, cfg.get("repl", R.REPLACEMENT_EUR_PER_MWH)) for c in comps}

    #Lorenzo Giannuzzo: checkpoint of the solved days, one line per (composition, day)
    ck = os.path.join(a.outdir, f"comps_{a.comps_scenario}_{a.year}_days.jsonl")
    done = {}
    if os.path.exists(ck):
        for line in open(ck):
            try:
                rec = json.loads(line); done[(rec["comp"], rec["day"])] = rec
            except Exception:
                pass
    todo = [(str(c), d) for c in comps for d in range(a.days) if (str(c), d) not in done]
    print(f"  [{a.comps_scenario} {a.year}] {len(done)} composition-days already done, {len(todo)} to go, "
          f"{a.workers} process(es) x {a.threads} thread(s)", flush=True)
    t0 = time.time()
    with open(ck, "a") as fh, mp.Pool(a.workers, initializer=_init, initargs=(win, opts, units, a.threads)) as pool:
        for i, rec in enumerate(pool.imap_unordered(_solve, todo)):
            fh.write(json.dumps(rec) + "\n"); fh.flush(); done[(rec["comp"], rec["day"])] = rec
            el = time.time() - t0
            eta = el / (i + 1) * (len(todo) - i - 1) / 3600
            print(f"  {rec['comp']} day {rec['day'] + 1} | {i + 1}/{len(todo)} | {rec['sec']:.0f} s | "
                  f"elapsed {el / 3600:.2f} h | about {eta:.1f} h left", flush=True)

    out = {}
    for c in comps:
        recs = [done[(str(c), d)] for d in range(a.days) if (str(c), d) in done]
        out[str(c)] = dict(value=sum(r["net"] for r in recs), tolerance=sum(r["tol"] for r in recs),
                           days_open=sum(r["open"] for r in recs), days=len(recs))
        print(f"  {a.comps_scenario} {a.year} {c}: v = {out[str(c)]['value'] / 1e3:.2f} kEUR, "
              f"tolerance {out[str(c)]['tolerance'] / 1e3:.2f} kEUR, {len(recs)} days", flush=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = os.path.join(a.outdir, f"comps_{a.comps_scenario}_{a.year}_par_{stamp}.json")
    with open(fn, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nresults in {fn}")


if __name__ == "__main__":
    main()
