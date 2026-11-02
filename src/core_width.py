"""
Width of the core of every composition of the game, projected on the payment of
one small unit (Fig. 5 of the paper).

Each composition (n_small, n_medium, n_large) is treated as a game on its own,
with the values of its sub-compositions taken from game_ref_<year>.json. The
other members of each class are treated alike, which entails no loss of
generality since the game is symmetric in identical units and its core convex.
When the core of a composition is empty, the least core is used instead.

    python core_width.py --outdir ../results --years 2023 2024

writes core_width_small_<year>.json: {"(s, m, l)": [lower, upper, max_eps]} in kEUR.
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import linprog


def load(fn):
    g = json.load(open(fn))
    v = {tuple(int(x) for x in k.strip("()").split(",")): val / 1e3 for k, val in g["values"].items()}
    v[(0, 0, 0)] = 0.0
    return v


def proj_small(v, S):
    """Lower and upper payment of one small unit in the (least) core of composition S, and the largest eps."""
    s,m,l=S
    if s==0: return None
    #Lorenzo Giannuzzo: variables x1 (the chosen small unit), xs (each other small unit), xm, xl, eps
    A=[]; b=[]
    for a in range(s+1):
        for bb in range(m+1):
            for c in range(l+1):
                if (a,bb,c) in ((0,0,0),S): continue
                if a>=1:  # coalition containing unit 1
                    A.append([-1,-(a-1),-bb,-c,1]); b.append(-v[(a,bb,c)])
                if a<=s-1:  # coalition without unit 1
                    A.append([0,-a,-bb,-c,1]); b.append(-v[(a,bb,c)])
    Aeq=[[1,s-1,m,l,0]]; beq=[v[S]]
    bnd=[(None,None),(None,None) if s>1 else (0,0),(None,None) if m else (0,0),(None,None) if l else (0,0)]
    r=linprog([0,0,0,0,-1],A_ub=A,b_ub=b,A_eq=Aeq,b_eq=beq,bounds=bnd+[(None,None)])
    e=min(r.x[4],0.0)
    lo=linprog([1,0,0,0,0],A_ub=A,b_ub=b,A_eq=Aeq,b_eq=beq,bounds=bnd+[(e,e)]).x[0]
    hi=linprog([-1,0,0,0,0],A_ub=A,b_ub=b,A_eq=Aeq,b_eq=beq,bounds=bnd+[(e,e)]).x[0]
    return lo,hi,r.x[4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="../results")
    ap.add_argument("--years", nargs="*", type=int, default=[2023, 2024])
    a = ap.parse_args()
    for y in a.years:
        v = load(os.path.join(a.outdir, f"game_ref_{y}.json"))
        out = {}
        for s in range(1, 9):
            for m in range(0, 9):
                for l in range(0, 5):
                    S = (s, m, l)
                    if sum(S) < 2:
                        continue
                    out[str(S)] = list(proj_small(v, S))
        fn = os.path.join(a.outdir, f"core_width_small_{y}.json")
        json.dump(out, open(fn, "w"), indent=0)
        w = {k: hi - lo for k, (lo, hi, _) in out.items()}
        print(f"{y}: {len(w)} compositions, width > 10 kEUR in {sum(x > 10 for x in w.values())}, "
              f"below 5 kEUR in {sum(x < 5 for x in w.values())}, full portfolio {w['(8, 8, 4)']:.2f} kEUR -> {fn}")


if __name__ == "__main__":
    main()
