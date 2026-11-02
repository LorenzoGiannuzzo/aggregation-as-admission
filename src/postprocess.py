"""
Post-processing of the outputs of experiments.py.

    python postprocess.py --outdir ../results --years 2023 2024

1. Exact cooperative game of the twenty units (game_ref_<year>.json): Shapley value,
   nucleolus and core of the class-symmetric allocations, core projection of each
   single unit, least core, and the largest excess of every allocation rule
   (units_ref_<year>.json) over all 2^20 coalitions, with the blocking composition.
2. Condition (iii): days on which the composition had to be solved (its
   threshold-free schedules did not clear the minimum bid).
3. Fixed cost of the virtual unit: largest cost each unit can bear.
4. Monte Carlo replay of the schedules (schedules_ref_<year>*.npz) with sampled
   awards and activations, unit by unit and with reallocation within the portfolio.
"""
import argparse
import json
import math
import os
from itertools import product

import numpy as np
from scipy.optimize import linprog

CLASSES = (("small", 8, 0.5, 4.0), ("medium", 8, 1.0, 2.0), ("large", 4, 2.5, 1.0))
N_TOT = tuple(c[1] for c in CLASSES)
LAB = [c[0] for c in CLASSES]


def load_game(fn):
    d = json.load(open(fn))
    v = {tuple(int(x) for x in k.strip("()").split(",")): float(val) for k, val in d["values"].items()}
    solved = {tuple(int(x) for x in k.strip("()").split(",")): int(n) for k, n in d["days_solved"].items()}
    tol = {tuple(int(x) for x in k.strip("()").split(",")): float(n) for k, n in d.get("tolerance", {}).items()}
    load_game.shortfall = {tuple(int(x) for x in k.strip("()").split(",")): float(n) for k, n in d.get("shortfall", {}).items()}
    return v, solved, tol


def shapley(v):
    n = sum(N_TOT); phi = {}
    for k in range(3):
        others = list(N_TOT); others[k] -= 1
        tot = 0.0
        for comp in product(*(range(m + 1) for m in others)):
            s = sum(comp)
            mult = math.comb(others[0], comp[0]) * math.comb(others[1], comp[1]) * math.comb(others[2], comp[2])
            w = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
            wk = list(comp); wk[k] += 1
            tot += mult * w * (v[tuple(wk)] - v[tuple(comp)])
        phi[LAB[k]] = tot
    return phi


def sym_core(v):
    """Least core, and interval of each class in the core (class-symmetric allocations)."""
    vN = v[N_TOT]
    rows = [(c, val) for c, val in v.items() if 0 < sum(c) < sum(N_TOT)]
    A = [[-c[0], -c[1], -c[2], 1.0] for c, _ in rows]; b = [-val for _, val in rows]
    res = linprog([0, 0, 0, -1], A_ub=A, b_ub=b, A_eq=[[*N_TOT, 0]], b_eq=[vN], bounds=[(None, None)] * 4, method="highs")
    eps = res.x[3]                      # >= 0: core non-empty with margin eps
    out = dict(least_core_margin=float(eps), core_nonempty=bool(eps >= -1e-6))
    A2 = [[-c[0], -c[1], -c[2]] for c, _ in rows]
    shift = max(0.0, -eps)              # if empty, intervals of the least core
    b2 = [-val + shift for _, val in rows]
    for k in range(3):
        cvec = [0, 0, 0]; cvec[k] = 1
        lo = linprog(cvec, A_ub=A2, b_ub=b2, A_eq=[list(N_TOT)], b_eq=[vN], bounds=[(None, None)] * 3, method="highs")
        hi = linprog([-x for x in cvec], A_ub=A2, b_ub=b2, A_eq=[list(N_TOT)], b_eq=[vN], bounds=[(None, None)] * 3, method="highs")
        out[f"{LAB[k]}_interval"] = (float(lo.x[k]), float(hi.x[k]))
    # nucleolus among class-symmetric allocations (equal to the nucleolus by symmetry):
    # successive maximization of the smallest margin, fixing the binding coalitions
    fixed = []
    x = None
    remaining = list(rows)
    for _ in range(6):
        A3 = [[-c[0], -c[1], -c[2], 1.0] for c, _ in remaining]; b3 = [-val for _, val in remaining]
        Aeq = [[*N_TOT, 0]] + [[c[0], c[1], c[2], 0] for c, val, _ in fixed]
        beq = [vN] + [val + m for c, val, m in fixed]
        r = linprog([0, 0, 0, -1], A_ub=A3, b_ub=b3, A_eq=Aeq, b_eq=beq, bounds=[(None, None)] * 4, method="highs")
        if r.status != 0:
            break
        x, m = r.x[:3], r.x[3]
        binding = [(c, val) for c, val in remaining if abs(np.dot(c, x) - val - m) < 1e-6]
        fixed += [(c, val, m) for c, val in binding]
        remaining = [(c, val) for c, val in remaining if (c, val) not in binding]
        if not remaining:
            break
    out["nucleolus"] = {LAB[k]: float(x[k]) for k in range(3)} if x is not None else None
    return out


def max_excess(x, labels, v):
    """Largest excess v(C) - x(C) over all coalitions: for each composition, the members with
    the smallest allocations in each class form the most dissatisfied coalition."""
    xs = {c: np.sort([xi for xi, l in zip(x, labels) if l == c]) for c in LAB}
    best, arg = -np.inf, None
    for comp, val in v.items():
        if sum(comp) in (0, sum(N_TOT)):
            continue
        pay = sum(xs[LAB[k]][:comp[k]].sum() for k in range(3))
        if val - pay > best:
            best, arg = val - pay, comp
    return best, arg


def unit_core_projection(v, labels, shift=0.0):
    """Range of each single unit's allocation over the core (over the least core, relaxing every coalition by
    shift, when the core is empty), by constraint generation."""
    n = len(labels)
    vN = v[N_TOT]
    idx = {c: [i for i, l in enumerate(labels) if l == c] for c in LAB}
    v = {c: val - (shift if 0 < sum(c) < sum(N_TOT) else 0.0) for c, val in v.items()}
    cuts = []
    # individual rationality and the coalitions of all members but one bound the polytope from the start
    for i, l in enumerate(labels):
        k = LAB.index(l)
        row = np.zeros(n); row[i] = -1.0
        cuts.append((row, -v[tuple(1 if j == k else 0 for j in range(3))]))
        row = -np.ones(n); row[i] = 0.0
        cuts.append((row, -v[tuple(m - (1 if j == k else 0) for j, m in enumerate(N_TOT))]))

    def solve(obj):
        nonlocal cuts
        for _ in range(200):
            A = [r for r, _ in cuts] or None; b = [bb for _, bb in cuts] or None
            r = linprog(obj, A_ub=A, b_ub=b, A_eq=[[1.0] * n], b_eq=[vN], bounds=[(None, None)] * n, method="highs")
            x = r.x
            added = False
            for comp, val in v.items():
                if sum(comp) in (0, sum(N_TOT)):
                    continue
                members = []
                for k in range(3):
                    order = sorted(idx[LAB[k]], key=lambda i: x[i])
                    members += order[:comp[k]]
                if val - x[members].sum() > 1e-7:
                    row = np.zeros(n); row[members] = -1.0
                    cuts.append((row, -val)); added = True
            if not added:
                return x
        return x

    out = {}
    for c in LAB:
        i = idx[c][0]
        e = np.zeros(n); e[i] = 1.0
        lo = solve(e)[i]; hi = solve(-e)[i]
        out[c] = (float(lo), float(hi))
    return out


# ----------------------------------------------------------------------------
# Monte Carlo replay
# ----------------------------------------------------------------------------

def replay(fn, n_samples=200, k_pen=1.5, fcr_phi=0.088, pool=False, seed=0, expected=False, act_mode="bernoulli"):
    """Samples awards (one draw per product-hour for the whole offer, at most one direction of each
    restoration product) and activations (full band for the hour, with the activation probability),
    and simulates the SOC of each unit. Energy that a unit cannot deliver (or absorb) is not paid and
    is charged at k_pen times the higher of the activation and day-ahead prices; with pool=True a unit's
    shortfall is first covered by the other units of the portfolio. Each day starts from the planned
    SOC, and the deviation left at the end of the day is settled at the mean day-ahead price."""
    d = np.load(fn, allow_pickle=False)
    keys = [str(k) for k in d["keys"]]
    Pch, Pdis, R, alpha, beta, ccap, cen, lam = (d[k] for k in ("Pch", "Pdis", "R", "alpha", "beta", "ccap", "cen", "prices"))
    SOC = d["SOC"]; e_cap = d["e_cap"]; eta = d["eta"]; c_deg = d["c_deg"]
    if "mult" in d.files and np.any(d["mult"] > 1):
        # merged units: each of the m identical units holds one m-th of the merged schedule
        m = d["mult"].astype(int)
        Pch = np.repeat(Pch / m[None, :, None], m, axis=1); Pdis = np.repeat(Pdis / m[None, :, None], m, axis=1)
        R = np.repeat(R / m[None, None, :, None], m, axis=2); SOC = np.repeat(SOC, m, axis=1)
        e_cap = np.repeat(e_cap / m, m); eta = np.repeat(eta, m); c_deg = np.repeat(c_deg, m)
    smin, smax = 0.10, 0.90
    D, K, N, T = R.shape[0], R.shape[1], R.shape[2], R.shape[3]
    rng = np.random.default_rng(seed)
    kidx = {k: j for j, k in enumerate(keys)}
    up = [k for k in keys if k.endswith("_up")]; dn = [k for k in keys if k.endswith("_dn")]
    plan = np.zeros(N); real = np.zeros((n_samples, N)); pen_tot = np.zeros(n_samples); short_mwh = np.zeros(n_samples); short_hours = np.zeros(n_samples)
    for day in range(D):
        # planned expected value of the day
        for j, k in enumerate(keys):
            z = 1.0 if k.endswith("_up") else (-1.0 if k.endswith("_dn") else 0.0)
            if k.startswith("fcr"):
                e_coef = alpha[day, j] * fcr_phi / 2 * lam[day]
            else:
                e_coef = alpha[day, j] * beta[day, j] * cen[day, j]
            plan += (alpha[day, j] * ccap[day, j] * R[day, j] + z * e_coef * R[day, j]).sum(axis=1)
        plan += ((Pdis[day] - Pch[day]) * lam[day]).sum(axis=1)
        for s in range(n_samples):
            soc = SOC[day, :, 0].copy() * e_cap         # MWh stored
            rev = np.zeros(N); pen = np.zeros(N); thr = np.zeros(N)
            for t in range(T):
                # awards and activations
                aw = {}
                uf = rng.random()
                for root in ("afrr", "mfrr"):
                    u = rng.random(); a_up = alpha[day, kidx[f"{root}_up"], t]; a_dn = alpha[day, kidx[f"{root}_dn"], t]
                    aw[f"{root}_up"] = u < a_up; aw[f"{root}_dn"] = a_up <= u < a_up + a_dn
                for k in keys:
                    if k.startswith("fcr"):
                        aw[k] = uf < alpha[day, kidx[k], t]
                if act_mode == "bernoulli":
                    # an awarded band is activated in full for the hour with the activation probability
                    act = {k: aw[k] and (k.startswith("fcr") or rng.random() < beta[day, kidx[k], t]) for k in keys}
                else:
                    # an awarded band delivers its expected activated energy (only the award is random)
                    act = {k: (float(beta[day, kidx[k], t]) if (aw[k] and not k.startswith("fcr")) else float(bool(aw[k])))
                           for k in keys}
                e_up = np.zeros(N); e_dn = np.zeros(N); v_up = np.zeros(N); v_dn = np.zeros(N)
                if expected:
                    # check of the replay itself: awards and activations at their expected values reproduce the plan
                    aw = {k: alpha[day, kidx[k], t] for k in keys}
                    act = {k: alpha[day, kidx[k], t] * (1.0 if k.startswith("fcr") else beta[day, kidx[k], t]) for k in keys}
                for k in keys:
                    j = kidx[k]; r = R[day, j, :, t]
                    if aw[k]:
                        rev += ccap[day, j, t] * r * float(aw[k])
                    if act[k]:
                        e = r * (fcr_phi / 2 if k.startswith("fcr") else 1.0) * float(act[k])
                        # restoration energy at its price, FCR energy at the day-ahead price
                        p_e = lam[day, t] if k.startswith("fcr") else cen[day, j, t]
                        if k in up:
                            e_up += e; v_up += p_e * e
                            rev += p_e * e
                        else:
                            e_dn += e; v_dn += p_e * e
                            rev -= p_e * e
                # energy flows of the hour, day-ahead schedule included
                out_e = Pdis[day, :, t] + e_up       # delivered
                in_e = Pch[day, :, t] + e_dn         # absorbed
                new = soc + eta * in_e - out_e / eta
                lo_v = np.maximum(smin * e_cap - new, 0.0)     # energy missing (stored side)
                hi_v = np.maximum(new - smax * e_cap, 0.0)
                if pool:
                    spare_lo = np.maximum(new - smin * e_cap, 0.0); spare_hi = np.maximum(smax * e_cap - new, 0.0)
                    cover_lo = min(lo_v.sum(), spare_lo.sum()); cover_hi = min(hi_v.sum(), spare_hi.sum())
                    if lo_v.sum() > 0 and cover_lo > 0:
                        new -= spare_lo / spare_lo.sum() * cover_lo
                        new += lo_v / lo_v.sum() * cover_lo
                        lo_v = lo_v * (1 - cover_lo / lo_v.sum())
                    if hi_v.sum() > 0 and cover_hi > 0:
                        new += spare_hi / spare_hi.sum() * cover_hi
                        new -= hi_v / hi_v.sum() * cover_hi
                        hi_v = hi_v * (1 - cover_hi / hi_v.sum())
                    lo_v = np.maximum(smin * e_cap - new, 0.0); hi_v = np.maximum(new - smax * e_cap, 0.0)
                miss_up = lo_v * eta                     # energy not delivered to the grid
                miss_dn = hi_v / eta                     # energy not absorbed from the grid
                p_pen = k_pen * max([lam[day, t]] + [cen[day, kidx[k], t] for k in keys if not k.startswith("fcr")])
                pen += (miss_up + miss_dn) * p_pen
                #Lorenzo Giannuzzo: activated energy that is not delivered (absorbed) is not paid (not charged), at the
                # mean energy price of the activations of the hour, the shortfall being attributed first to the activations
                p_up = np.divide(v_up, e_up, out=np.zeros(N), where=e_up > 0); p_dn = np.divide(v_dn, e_dn, out=np.zeros(N), where=e_dn > 0)
                rev -= np.minimum(miss_up, e_up) * p_up
                rev += np.minimum(miss_dn, e_dn) * p_dn
                if np.any(miss_up + miss_dn > 1e-9):
                    short_hours[s] += 1
                short_mwh[s] += (miss_up + miss_dn).sum()
                soc = np.clip(new, smin * e_cap, smax * e_cap)
                thr += in_e + out_e
                rev += (Pdis[day, :, t] - Pch[day, :, t]) * lam[day, t]
            # end-of-day deviation from the planned SOC, settled at the mean day-ahead price
            dev = soc - SOC[day, :, -1] * e_cap
            rev += dev * lam[day].mean()
            real[s] += rev - pen
            pen_tot[s] += pen.sum()
    return dict(plan_eur=plan, real_mean_eur=real.mean(axis=0), real_p05_total=float(np.percentile(real.sum(axis=1), 5)),
                real_total_mean=float(real.sum(axis=1).mean()), plan_total=float(plan.sum()),
                penalty_mean=float(pen_tot.mean()),
                shortfall_mwh_per_year=float(short_mwh.mean()), shortfall_hours_per_year=float(short_hours.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="../results")
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    ap.add_argument("--scenario", default="ref")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--no-replay", action="store_true")
    a = ap.parse_args()
    report = {}
    for year in a.years:
        rep = {}
        gfn = os.path.join(a.outdir, f"game_{a.scenario}_{year}.json")
        ufn = os.path.join(a.outdir, f"units_{a.scenario}_{year}.json")
        if os.path.exists(gfn):
            v, solved, tol = load_game(gfn)
            labels = sum(([c[0]] * c[1] for c in CLASSES), [])
            phi = shapley(v)
            core = sym_core(v)
            g = {LAB[k]: v[tuple(1 if j == k else 0 for j in range(3))] for k in range(3)}
            rep["shapley_per_unit_keur"] = {k: x / 1e3 for k, x in phi.items()}
            #Lorenzo Giannuzzo: every money figure of the report in kEUR (margin and nucleolus included)
            rep["core"] = {}
            for k, val in core.items():
                if isinstance(val, tuple):
                    rep["core"][k] = tuple(z / 1e3 for z in val)
                elif k == "least_core_margin":
                    rep["core"]["least_core_margin_keur"] = val / 1e3
                elif k == "nucleolus" and val is not None:
                    rep["core"]["nucleolus_keur"] = {c: z / 1e3 for c, z in val.items()}
                else:
                    rep["core"][k] = val
            rep["unit_core_projection_keur"] = {k: tuple(z / 1e3 for z in val) for k, val in
                                                unit_core_projection(v, labels, shift=max(0.0, -core["least_core_margin"])).items()}
            rep["v_N_keur"] = v[N_TOT] / 1e3
            rep["v_N_minus_i_keur"] = {LAB[k]: v[tuple(n - (1 if j == k else 0) for j, n in enumerate(N_TOT))] / 1e3 for k in range(3)}
            rep["days_solved"] = {str(c): n for c, n in solved.items() if n > 0}
            rep["days_solved_N_and_N_minus_i"] = {str(c): solved[c] for c in
                                                  [N_TOT] + [tuple(n - (1 if j == k else 0) for j, n in enumerate(N_TOT)) for k in range(3)]}
            rep["shortfall_from_threshold_free_keur"] = {str(c): load_game.shortfall.get(c, 0.0) / 1e3 for c in
                                                         [N_TOT] + [tuple(n - (1 if j == k else 0) for j, n in enumerate(N_TOT)) for k in range(3)]}
            rep["compositions_solved_on_some_day"] = sum(1 for n in solved.values() if n > 0)
            rep["tolerance_keur"] = {str(c): t / 1e3 for c, t in tol.items() if t > 0}
            shap_x = [phi[l] for l in labels]
            rep["shapley_max_excess_keur"] = max_excess(shap_x, labels, v)[0] / 1e3
            rep["shapley_blocked_by"] = str(max_excess(shap_x, labels, v)[1])
            if os.path.exists(ufn):
                u = json.load(open(ufn))
                rep["rules"] = {}
                for rule, x in u["allocations"].items():
                    e, c = max_excess(np.array(x), u["labels"], v)
                    rep["rules"][rule] = dict(max_excess_keur=e / 1e3, blocked_by=str(c),
                                              per_unit_range_keur={l: (min(xx for xx, ll in zip(x, u["labels"]) if ll == l) / 1e3,
                                                                       max(xx for xx, ll in zip(x, u["labels"]) if ll == l) / 1e3) for l in LAB})
                sa = {l: next(s for s, ll in zip(u["standalone"], u["labels"]) if ll == l) for l in LAB}
                rep["fixed_cost_cap_per_unit_keur"] = {l: (u["threshold_free"][l] - sa[l]) / 1e3 for l in LAB if u["threshold_free"][l] is not None}
                caps = rep["fixed_cost_cap_per_unit_keur"]
                rep["fixed_cost_equal_split_max_keur"] = min(caps.values()) * sum(N_TOT)
        if not a.no_replay:
            for name in (f"schedules_{a.scenario}_{year}.npz",) + tuple(f"schedules_{a.scenario}_{year}_standalone_{l}.npz" for l in LAB):
                fn = os.path.join(a.outdir, name)
                if os.path.exists(fn):
                    clean = lambda r: {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
                    entry = dict(check_expected=clean(replay(fn, n_samples=1, expected=True)),
                                 unit_level=clean(replay(fn, n_samples=a.samples)),
                                 unit_level_mean_activation=clean(replay(fn, n_samples=a.samples, act_mode="mean")))
                    if "standalone" not in name:
                        entry["pool_level"] = clean(replay(fn, n_samples=a.samples, pool=True))
                        entry["pool_level_mean_activation"] = clean(replay(fn, n_samples=a.samples, pool=True, act_mode="mean"))
                    rep[f"replay_{name}"] = entry
        report[year] = rep
    fn = os.path.join(a.outdir, f"post_{a.scenario}.json")
    with open(fn, "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    print(json.dumps(report, indent=1, default=float)[:6000])
    print(f"\nwritten {fn}")


if __name__ == "__main__":
    main()
