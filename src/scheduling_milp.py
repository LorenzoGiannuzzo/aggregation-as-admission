"""
Daily scheduling MILP of a coalition of battery units under a minimum bid.

Each unit trades on the day-ahead market and reserves capacity on six
directional reserve products (FCR, aFRR and mFRR, upward and downward). The
minimum bid applies to the aggregate offer of the coalition in every product
and hour, so the same function solves a single unit (standalone), the whole
portfolio (aggregated) and every coalition of the cooperative game.

Reference formulation (ModelOptions() defaults):
  - each day starts and ends at the same SOC (cyclic condition), so that the
    daily problems are independent and the annual game is the sum of daily games;
  - power capability written with separate upward and downward headroom,
        P_dis - P_ch + sum_up R <= P_max,   P_ch - P_dis + sum_dn R <= P_max,
    each band being also bounded by the rated power, and no direction binaries
    (a unit can hold both half-bands in one hour);
  - FCR procured as two directional products (upward and downward), each paid
    half of the symmetric capacity price, as in the Italian procurement of 2025;
  - activated energy of aFRR and mFRR tied to the band (the TSO decides it);
  - expected FCR activation energy equal to phi/2 of the band per direction,
    phi being the mean absolute normalized FCR activation;
  - sustain constraints at the start and at the end of each hour, with the
    conversion efficiency;
  - degradation cost on the expected energy throughput, at a replacement cost
    set by the caller (BatteryUnit.c_deg).

ModelOptions(preliminary=True) rebuilds the formulation of the preliminary
(conference) version of the work, used as the starting point of the bridge.

Optional couplings and market variants: integer bid granularity, a price-
responsive FCR capacity price (residual demand), the forward capacity
product of the UVAM (upward offers in an evening window on working days), and
a margin on the SOC (soc_margin) added to the bounds of the sustain constraints
in every hour and direction in which a unit holds reserve.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pulp

from market_products import ServiceType


# ============================================================================
# Inputs
# ============================================================================

@dataclass
class BatteryUnit:
    p_max: float                 # rated power [MW]
    e_cap: float                 # energy capacity [MWh]
    eta: float = 0.95            # efficiency in each direction [-]
    soc_min: float = 0.10
    soc_max: float = 0.90
    c_deg: float = 6.25          # degradation cost per MWh of throughput [EUR/MWh]
    label: str = ""


@dataclass
class UvamProduct:
    """Forward capacity product of the UVAM (Terna, 2022-2023 'serale' product):
    the contracted quantity, in whole MW, must be offered upward on the
    restoration products in every hour of the window of each working day."""
    premium_eur_per_mw_year: float = 20000.0
    hours: Sequence[int] = (18, 19, 20, 21)   # 18:00-22:00
    working_days_per_year: float = 250.0
    integer_mw: bool = True

    @property
    def premium_per_day(self) -> float:
        return self.premium_eur_per_mw_year / self.working_days_per_year


@dataclass
class ModelOptions:
    preliminary: bool = False             # formulation of the preliminary version (overrides the flags below)
    cyclic_soc: bool = True
    soc0: float = 0.50
    headroom: bool = True            # separate upward / downward headroom (else one budget for all)
    direction_mutex: bool = False    # one direction per restoration product and hour, per unit
    fcr_directional: bool = True
    fcr_phi: float = 0.088           # mean absolute normalized FCR activation: E|df|/200 mHz for a normal df with an assumed sd of 22 mHz, no deadband
    tied_activation: bool = True     # aFRR / mFRR activated energy = band
    sustain_eta: bool = True         # sustain energy divided (up) or multiplied (down) by eta
    sustain_end_of_hour: bool = True
    throughput_full: bool = False    # activated energy counted in full in the throughput (FCR band included)
    fcr_soc_neutral: bool = False    # FCR activations leave the SOC unchanged (no losses, no asymmetry)
    min_bid: float = 1.0
    granularity: float = 0.0         # MW; 0 = continuous offers above the minimum bid
    fcr_depth_mw: float = 0.0        # >0: linear residual demand for FCR, price -> 0 at this offer
    fcr_depth_steps: int = 10
    uvam: Optional[UvamProduct] = None
    tau: Dict[str, float] = field(default_factory=lambda: {"fcr": 0.25, "afrr": 1.0, "mfrr": 2.0})
    gap: float = 1e-4
    time_limit: float = 300.0
    soc_margin: float = 0.0          # SOC margin for estimation error, added to the bounds of the sustain constraints in the hours with reserve
    tiebreak_floor: Optional[float] = None   # if set: among schedules earning at least this, the one with the least band


# product key -> (service type, direction, root)
_PRODUCTS_REF = {
    "fcr_up": (ServiceType.FCR, "up", "fcr"), "fcr_dn": (ServiceType.FCR, "dn", "fcr"),
    "afrr_up": (ServiceType.AFRR_UP, "up", "afrr"), "afrr_dn": (ServiceType.AFRR_DN, "dn", "afrr"),
    "mfrr_up": (ServiceType.MFRR_UP, "up", "mfrr"), "mfrr_dn": (ServiceType.MFRR_DN, "dn", "mfrr"),
}
_PRODUCTS_SYM = {
    "fcr": (ServiceType.FCR, "sym", "fcr"),
    "afrr_up": (ServiceType.AFRR_UP, "up", "afrr"), "afrr_dn": (ServiceType.AFRR_DN, "dn", "afrr"),
    "mfrr_up": (ServiceType.MFRR_UP, "up", "mfrr"), "mfrr_dn": (ServiceType.MFRR_DN, "dn", "mfrr"),
}


def _preliminary_options(o: ModelOptions) -> ModelOptions:
    """The formulation of the preliminary (conference) version of the work."""
    return ModelOptions(preliminary=True, cyclic_soc=False, soc0=o.soc0, headroom=False, direction_mutex=True,
                      fcr_directional=False, fcr_phi=1.0, tied_activation=False, sustain_eta=False,
                      sustain_end_of_hour=False, throughput_full=True, fcr_soc_neutral=True, min_bid=o.min_bid, granularity=o.granularity,
                      fcr_depth_mw=0.0, uvam=None, tau=dict(o.tau), gap=o.gap, time_limit=o.time_limit)


# ============================================================================
# Solver
# ============================================================================

_GUROBI_OK = None


def _grb_extra_params() -> dict:
    """Extra Gurobi parameters from MILP_GRB_PARAMS, e.g. "MIPFocus=1,Cuts=2"."""
    out = {}
    for item in os.environ.get("MILP_GRB_PARAMS", "").split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            v = v.strip()
            out[k.strip()] = float(v) if "." in v or "e" in v.lower() else int(v)
    return out


def build_solver(gap: float, time_limit: float):
    """Gurobi when available (academic licence), else HiGHS, else CBC."""
    global _GUROBI_OK
    threads = os.environ.get("MILP_THREADS")
    try:
        import gurobipy  # noqa: F401
        #Lorenzo Giannuzzo: OutputFlag goes first, so that setting the other parameters prints nothing
        kw = {"OutputFlag": 0}
        if threads not in (None, ""):
            kw["Threads"] = int(threads)
        kw.update(_grb_extra_params())
        s = pulp.GUROBI(msg=0, timeLimit=time_limit, gapRel=gap, **kw)
        #Lorenzo Giannuzzo: the licence is checked once per process (each check opens a Gurobi environment)
        if _GUROBI_OK is None:
            _GUROBI_OK = bool(s.available())
        if _GUROBI_OK:
            return s, "gurobi"
    except Exception:
        pass
    if "HiGHS" in pulp.listSolvers(onlyAvailable=True):
        return pulp.HiGHS(msg=False, timeLimit=time_limit, gapRel=gap, output_flag=False), "highs"
    return pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=gap), "cbc"


# ============================================================================
# Model
# ============================================================================

class DayResult(dict):
    """Per-day result. Arrays have one row per unit."""


def solve_day(units: List[BatteryUnit], prices: Sequence[float], services: Sequence[Sequence],
              opts: ModelOptions, soc_start: Optional[Sequence[float]] = None,
              working_day: bool = True, keep_schedule: bool = False) -> DayResult:
    o = _preliminary_options(opts) if opts.preliminary else opts
    N, T = len(units), len(prices)
    prods = _PRODUCTS_REF if o.fcr_directional else _PRODUCTS_SYM
    keys = list(prods)
    up_keys = [k for k in keys if prods[k][1] in ("up", "sym")]
    dn_keys = [k for k in keys if prods[k][1] in ("dn", "sym")]
    restor = [k for k in keys if prods[k][2] != "fcr"]
    fcr_keys = [k for k in keys if prods[k][2] == "fcr"]
    tau = {k: o.tau[prods[k][2]] for k in keys}
    soc0 = list(soc_start) if soc_start is not None else [o.soc0] * N

    # ---- market data per product and hour ----
    def svc(t, st):
        for s in services[t]:
            if s.service_type == st:
                return s
        return None

    alpha = np.zeros((len(keys), T)); beta = np.zeros((len(keys), T))
    ccap = np.zeros((len(keys), T)); cen = np.zeros((len(keys), T)); avail = np.zeros((len(keys), T), bool)
    for j, k in enumerate(keys):
        for t in range(T):
            s = svc(t, prods[k][0])
            if s is None or float(s.award_probability) <= 0.0:
                continue
            avail[j, t] = True
            alpha[j, t] = float(s.award_probability)
            beta[j, t] = float(s.activation_probability)
            share = 0.5 if (prods[k][2] == "fcr" and prods[k][1] != "sym") else 1.0
            ccap[j, t] = float(s.capacity_price) * share
            cen[j, t] = float(s.energy_price)
    zeta = {k: (1.0 if prods[k][1] == "up" else -1.0) if prods[k][2] != "fcr" else 0.0 for k in keys}

    # expected activated energy per MW of band and hour, entering the SOC (by direction) and the throughput
    def thr_coef(j, k, t):
        """Expected energy moved per unit of the activation variable (the band if tied)."""
        if o.throughput_full:
            return 1.0
        if prods[k][2] == "fcr":
            return alpha[j, t] * o.fcr_phi * (0.5 if prods[k][1] != "sym" else 1.0)
        return alpha[j, t] * beta[j, t]

    def act_coef(j, k, t):
        """Expected energy entering the SOC per unit of the activation variable."""
        if prods[k][2] == "fcr":
            if o.fcr_soc_neutral:
                return 0.0
            return alpha[j, t] * o.fcr_phi * (0.5 if prods[k][1] != "sym" else 1.0)
        return alpha[j, t] * beta[j, t]

    prob = pulp.LpProblem("daily_schedule", pulp.LpMaximize)
    I, H = range(N), range(T)
    Pch = {(i, t): pulp.LpVariable(f"Pch_{i}_{t}", 0, units[i].p_max) for i in I for t in H}
    Pdis = {(i, t): pulp.LpVariable(f"Pdis_{i}_{t}", 0, units[i].p_max) for i in I for t in H}
    SOC = {(i, t): pulp.LpVariable(f"SOC_{i}_{t}", units[i].soc_min, units[i].soc_max) for i in I for t in range(T + 1)}
    R = {(k, i, t): pulp.LpVariable(f"R_{k}_{i}_{t}", 0, units[i].p_max) for k in keys for i in I for t in H}
    A = {}
    for k in restor:
        if not o.tied_activation:
            A.update({(k, i, t): pulp.LpVariable(f"A_{k}_{i}_{t}", 0) for i in I for t in H})

    def act(k, i, t):
        """Activated energy variable (MWh) of product k: the band if tied."""
        if prods[k][2] == "fcr" or o.tied_activation:
            return R[(k, i, t)]
        return A[(k, i, t)]

    def fcr_e(j, k, t):
        """Settlement of the expected FCR activation energy at the day-ahead price, a proxy of the
        imbalance settlement: delivered upward energy is sold, absorbed downward energy is bought
        (a symmetric band nets out)."""
        if prods[k][2] != "fcr" or o.fcr_soc_neutral or prods[k][1] == "sym":
            return 0.0
        return (1.0 if prods[k][1] == "up" else -1.0) * float(prices[t]) * act_coef(j, k, t)

    share = {k: {"up": (1.0, 0.0), "dn": (0.0, 1.0), "sym": (0.5, 0.5)}[prods[k][1]] for k in keys}

    obj = []
    for i, u in enumerate(units):
        prob += SOC[(i, 0)] == soc0[i], f"Init_{i}"
        if o.cyclic_soc:
            prob += SOC[(i, T)] == soc0[i], f"Cyclic_{i}"
        for t in H:
            # expected energy delivered (up) and absorbed (dn) through the bands; a symmetric FCR band
            # moves half of its expected throughput in each direction (none in the preliminary model)
            e_up = pulp.lpSum(share[k][0] * act_coef(keys.index(k), k, t) * act(k, i, t) for k in keys if share[k][0])
            e_dn = pulp.lpSum(share[k][1] * act_coef(keys.index(k), k, t) * act(k, i, t) for k in keys if share[k][1])
            prob += (SOC[(i, t + 1)] == SOC[(i, t)]
                     + (u.eta * (Pch[(i, t)] + e_dn) - (Pdis[(i, t)] + e_up) / u.eta) / u.e_cap), f"Dyn_{i}_{t}"

            # power capability
            if o.headroom:
                prob += Pdis[(i, t)] - Pch[(i, t)] + pulp.lpSum(R[(k, i, t)] for k in up_keys) <= u.p_max, f"Hup_{i}_{t}"
                prob += Pch[(i, t)] - Pdis[(i, t)] + pulp.lpSum(R[(k, i, t)] for k in dn_keys) <= u.p_max, f"Hdn_{i}_{t}"
            else:
                prob += Pch[(i, t)] + Pdis[(i, t)] + pulp.lpSum(R[(k, i, t)] for k in keys) <= u.p_max, f"Bud_{i}_{t}"

            # charge / discharge exclusion only where a non-positive price could make it pay
            if float(prices[t]) <= 0.0:
                bc = pulp.LpVariable(f"bc_{i}_{t}", cat="Binary"); bd = pulp.LpVariable(f"bd_{i}_{t}", cat="Binary")
                prob += Pch[(i, t)] <= u.p_max * bc; prob += Pdis[(i, t)] <= u.p_max * bd; prob += bc + bd <= 1

            # sustain: energy that the upward bands are liable to deliver, and the downward ones to absorb
            up_need = pulp.lpSum(tau[k] * R[(k, i, t)] for k in up_keys)
            dn_need = pulp.lpSum(tau[k] * R[(k, i, t)] for k in dn_keys)
            f_up = (1.0 / u.eta) if o.sustain_eta else 1.0
            f_dn = u.eta if o.sustain_eta else 1.0
            moments = [t, t + 1] if o.sustain_end_of_hour else [t]
            #Lorenzo Giannuzzo: with a SOC margin, the bounds of the sustain constraints are tightened by the margin in
            # every hour and direction in which the unit holds reserve, through an indicator of that reserve
            if o.soc_margin > 0:
                zu = pulp.LpVariable(f"zu_{i}_{t}", cat="Binary"); zd = pulp.LpVariable(f"zd_{i}_{t}", cat="Binary")
                prob += up_need <= sum(tau[k] for k in up_keys) * u.p_max * zu
                prob += dn_need <= sum(tau[k] for k in dn_keys) * u.p_max * zd
                lo = u.soc_min * u.e_cap + o.soc_margin * u.e_cap * zu
                hi = u.soc_max * u.e_cap - o.soc_margin * u.e_cap * zd
            else:
                lo, hi = u.soc_min * u.e_cap, u.soc_max * u.e_cap
            for m in moments:
                prob += SOC[(i, m)] * u.e_cap - f_up * up_need >= lo, f"SusUp_{i}_{t}_{m}"
                prob += SOC[(i, m)] * u.e_cap + f_dn * dn_need <= hi, f"SusDn_{i}_{t}_{m}"

            if o.direction_mutex:
                for root in ("afrr", "mfrr"):
                    y = pulp.LpVariable(f"y_{root}_{i}_{t}", cat="Binary")
                    prob += R[(f"{root}_up", i, t)] <= u.p_max * y
                    prob += R[(f"{root}_dn", i, t)] <= u.p_max * (1 - y)
            if not o.tied_activation:
                for k in restor:
                    prob += A[(k, i, t)] <= R[(k, i, t)]

            # arbitrage and degradation on the expected throughput
            thr = [Pch[(i, t)], Pdis[(i, t)]]
            for j, k in enumerate(keys):
                thr.append(thr_coef(j, k, t) * act(k, i, t))
            obj.append(float(prices[t]) * (Pdis[(i, t)] - Pch[(i, t)]) - u.c_deg * pulp.lpSum(thr))

    # ---- coalition coupling: minimum bid, granularity, revenue ----
    Ptot = sum(u.p_max for u in units)
    #Lorenzo Giannuzzo: tightest big-M of each product implied by the bounds of the bands and by the sustain
    # constraints, tau R / eta <= (soc_max - soc_min) E upward and eta tau R <= (soc_max - soc_min) E downward;
    # a valid inequality, so the optimum is unchanged and only the relaxation is tighter
    Mk = {}
    for k in keys:
        m = 0.0
        for u in units:
            usable = (u.soc_max - u.soc_min) * u.e_cap
            f_up = (1.0 / u.eta) if o.sustain_eta else 1.0
            f_dn = u.eta if o.sustain_eta else 1.0
            caps = [u.p_max]
            if prods[k][1] in ("up", "sym"):
                caps.append(usable / (f_up * tau[k]))
            if prods[k][1] in ("dn", "sym"):
                caps.append(usable / (f_dn * tau[k]))
            m += min(caps)
        Mk[k] = max(min(m, Ptot), 1e-6)
    depth_q = {}
    for j, k in enumerate(keys):
        for t in H:
            agg = pulp.lpSum(R[(k, i, t)] for i in I)
            if not avail[j, t]:
                prob += agg == 0, f"NA_{k}_{t}"
                continue
            if o.granularity > 0:
                K = pulp.LpVariable(f"K_{k}_{t}", 0, int(np.floor(Ptot / o.granularity + 1e-9)), cat="Integer")
                prob += agg == o.granularity * K, f"Gran_{k}_{t}"
                if o.min_bid > o.granularity + 1e-9:
                    ub = pulp.LpVariable(f"u_{k}_{t}", cat="Binary")
                    prob += agg >= o.min_bid * ub; prob += agg <= Mk[k] * ub
            elif o.min_bid > 1e-9:
                ub = pulp.LpVariable(f"u_{k}_{t}", cat="Binary")
                prob += agg >= o.min_bid * ub, f"Min_{k}_{t}"
                prob += agg <= Mk[k] * ub, f"Max_{k}_{t}"
            # capacity revenue
            if prods[k][2] == "fcr" and o.fcr_depth_mw > 0:
                # linear residual demand p(Q) = p0 (1 - Q / D): revenue p(Q) Q, concave, in steps
                D, n = o.fcr_depth_mw, o.fcr_depth_steps
                w = D / n
                qs = [pulp.LpVariable(f"q_{k}_{t}_{s}", 0, w) for s in range(n)]
                prob += agg == pulp.lpSum(qs), f"Depth_{k}_{t}"
                # marginal revenue of step s of p0 (1 - Q/D) Q is p0 (1 - (2s+1)/n)
                for s, q in enumerate(qs):
                    obj.append(alpha[j, t] * ccap[j, t] * (1.0 - (2 * s + 1) / n) * q)
                depth_q[(k, t)] = qs
            else:
                obj.append(alpha[j, t] * ccap[j, t] * agg)
            if prods[k][2] != "fcr":
                obj.append(zeta[k] * alpha[j, t] * beta[j, t] * cen[j, t] * pulp.lpSum(act(k, i, t) for i in I))
            elif fcr_e(j, k, t) != 0.0:
                obj.append(fcr_e(j, k, t) * agg)

    C_uvam = None
    if o.uvam is not None and working_day:
        uv = o.uvam
        cat = "Integer" if uv.integer_mw else "Continuous"
        C_uvam = pulp.LpVariable("C_uvam", 0, int(np.floor(Ptot)), cat=cat)
        bu = pulp.LpVariable("b_uvam", cat="Binary")
        prob += C_uvam >= o.min_bid * bu; prob += C_uvam <= Ptot * bu
        for t in uv.hours:
            prob += pulp.lpSum(R[(k, i, t)] for i in I for k in ("afrr_up", "mfrr_up")) >= C_uvam, f"Uvam_{t}"
        obj.append(uv.premium_per_day * C_uvam)

    if o.tiebreak_floor is None:
        prob += pulp.lpSum(obj)
    else:
        #Lorenzo Giannuzzo: second stage of the threshold-free problem: among the optimal schedules, the one that
        # offers the least band, so that offers of no value (left by the degeneracy of the LP) are null
        prob += pulp.lpSum(obj) >= o.tiebreak_floor, "Floor"
        prob += -pulp.lpSum(R.values())
    solver, sname = build_solver(o.gap, o.time_limit)
    t0 = time.time()
    prob.solve(solver)
    runtime = time.time() - t0
    status = pulp.LpStatus[prob.status]
    gap_rel, bound = None, None
    sm = getattr(prob, "solverModel", None)
    if sm is not None:
        try:
            gap_rel = float(sm.MIPGap); bound = float(sm.ObjBound)
        except Exception:
            pass
    val = lambda x: float(x.varValue) if x.varValue is not None else 0.0
    if status not in ("Optimal",) and Pch[(0, 0)].varValue is None:
        raise RuntimeError(f"no solution ({status})")

    # ---- extraction, per unit ----
    out = DayResult(status=status, solver=sname, runtime=runtime, gap=gap_rel, bound=bound,
                    objective=float(pulp.value(prob.objective)))
    arb = np.zeros(N); deg = np.zeros(N); rev = np.zeros((N, len(keys))); band = np.zeros((N, len(keys)))
    for i, u in enumerate(units):
        for t in H:
            arb[i] += float(prices[t]) * (val(Pdis[(i, t)]) - val(Pch[(i, t)]))
            thr = val(Pch[(i, t)]) + val(Pdis[(i, t)])
            for j, k in enumerate(keys):
                r = val(R[(k, i, t)]); a = val(act(k, i, t))
                band[i, j] += r
                thr += thr_coef(j, k, t) * a
                if prods[k][2] == "fcr":
                    rev[i, j] += fcr_e(j, k, t) * r
                if prods[k][2] == "fcr" and o.fcr_depth_mw > 0:
                    continue
                rev[i, j] += alpha[j, t] * ccap[j, t] * r
                if prods[k][2] != "fcr":
                    rev[i, j] += zeta[k] * alpha[j, t] * beta[j, t] * cen[j, t] * a
            deg[i] += u.c_deg * thr
    # price-responsive FCR and UVAM revenue: shared pro rata to each unit's band
    if depth_q:
        for (k, t), qs in depth_q.items():
            j = keys.index(k)
            n = len(qs)
            tot_rev = sum(alpha[j, t] * ccap[j, t] * (1.0 - (2 * s + 1) / n) * val(q) for s, q in enumerate(qs))
            rs = np.array([val(R[(k, i, t)]) for i in I]); Q = rs.sum()
            if Q > 1e-9:
                rev[:, j] += tot_rev * rs / Q
    uvam_rev = np.zeros(N)
    if C_uvam is not None and val(C_uvam) > 1e-9:
        w = np.array([sum(val(R[(k, i, t)]) for t in o.uvam.hours for k in ("afrr_up", "mfrr_up")) for i in I])
        uvam_rev = o.uvam.premium_per_day * val(C_uvam) * (w / w.sum() if w.sum() > 0 else np.ones(N) / N)
    out.update(keys=keys, arb=arb, deg=deg, rev=rev, band=band, uvam=uvam_rev,
               net=arb - deg + rev.sum(axis=1) + uvam_rev,
               soc_end=np.array([val(SOC[(i, T)]) for i in I]),
               uvam_mw=(val(C_uvam) if C_uvam is not None else 0.0))
    if keep_schedule:
        out.update(Pch=np.array([[val(Pch[(i, t)]) for t in H] for i in I]),
                   Pdis=np.array([[val(Pdis[(i, t)]) for t in H] for i in I]),
                   SOC=np.array([[val(SOC[(i, t)]) for t in range(T + 1)] for i in I]),
                   R=np.array([[[val(R[(k, i, t)]) for t in H] for i in I] for k in keys]),
                   alpha=alpha, beta=beta, ccap=ccap, cen=cen, prices=np.asarray(prices, float))
    return out
