"""Fig. 4: game of 2023, profit per unit of each class against its threshold-free value, and the
core, Shapley value and nucleolus of one small unit joined to one or two large units."""
import itertools
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import linprog

from common import W, out, res, style

style()


#Lorenzo Giannuzzo: game values in kEUR and the small games of scarce admission
def G(y):
    g=json.load(open(res(f'game_ref_{y}.json'))); v={tuple(int(x) for x in k.strip('()').split(',')):val/1e3 for k,val in g['values'].items()}
    v[(0,0,0)]=0.0
    return v,{k:x/1e3 for k,x in g['threshold_free_per_unit'].items()}
def shapley(players, val):
    n=len(players); phi=np.zeros(n)
    from math import factorial
    for i in range(n):
        others=[j for j in range(n) if j!=i]
        for r in range(n):
            for S in itertools.combinations(others,r):
                w=factorial(len(S))*factorial(n-len(S)-1)/factorial(n)
                phi[i]+=w*(val(set(S)|{i})-val(set(S)))
    return phi
def comp(S,types): return tuple(sum(1 for j in S if types[j]==t) for t in range(3))
def nucleolus3(types,val):
    # small games: lexicographic via LP sequence (max min excess)
    n=len(types); coal=[set(c) for r in range(1,n) for c in itertools.combinations(range(n),r)]
    vN=val(set(range(n)))
    # maximize eps: x(S)-v(S) >= eps
    c=np.zeros(n+1); c[-1]=-1
    A=[]; b=[]
    for S in coal:
        row=np.zeros(n+1); row[list(S)]=-1; row[-1]=1; A.append(row); b.append(-val(S))
    Aeq=[np.r_[np.ones(n),0]]; beq=[vN]
    res=linprog(c,A_ub=A,b_ub=b,A_eq=Aeq,b_eq=beq,bounds=[(None,None)]*(n+1))
    eps=res.x[-1]
    # second stage: fix tight coalitions approx; for symmetric games enough to symmetrize
    return res.x[:n], eps
out_={}
for y in (2023,2024):
    v,g=G(y); sa={'small':v[(1,0,0)],'medium':v[(0,1,0)],'large':v[(0,0,1)]}
    res_={}
    for name,types in (('1s1l',[0,2]),('1s2l',[0,2,2])):
        val=lambda S: v[comp(S,types)]
        phi=shapley(list(range(len(types))),val)
        vN=val(set(range(len(types))))
        if name=='1s1l':
            lo, hi = sa['small'], vN - sa['large']
        else:
            lo=max(sa['small'], 2*v[(1,0,1)]-vN); hi=min(vN - v[(0,0,2)], vN-2*sa['large'])
        nuc,eps=nucleolus3(types,val)
        res_[name]=dict(core=(lo,hi),shapley_small=phi[0],nucleolus_small=nuc[0],vN=vN)
    out_[y]=dict(g=g,sa=sa,pair={'small':v[(2,0,0)]/2,'medium':v[(0,2,0)]/2,'large':v[(0,0,2)]/2},
                pool={'small':v[(8,0,0)]/8,'medium':v[(0,8,0)]/8,'large':v[(0,0,4)]/4},scarce=res_)
CL = ['small', 'medium', 'large']
D = {str(k): v for k, v in out_.items()}
P = json.load(open(res('post_ref.json')))
#Lorenzo Giannuzzo: game of 2023, profit per unit minus threshold-free value; Shapley value now exact
y = '2023'; d = D[y]; u = json.load(open(res(f'units_ref_{y}.json')))
pt = {c: np.mean([a for a, l in zip(u['allocations']['passthru'], u['labels']) if l == c]) / 1e3 for c in CL}
sh = P[y]['shapley_per_unit_keur']
SER = [("Standalone", d['sa'], "#FF8A1F", None), ("Same-Class Pair", d['pair'], "#2A9DF4", None), ("Class Pool", d['pool'], matplotlib.colormaps["Greens"](0.85), None),
       ("Pass-Through", pt, "#084A91", None), ("Shapley Value", sh, "#9B4DDB", "\\\\")]
fig = plt.figure(figsize=(W, 5.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[1.2, 1], hspace=0.75, wspace=0.42)
a1 = fig.add_subplot(gs[0, :]); a2 = fig.add_subplot(gs[1, 0]); a3 = fig.add_subplot(gs[1, 1])
x = np.arange(3); w = 0.15
for k, (name, val, col, hat) in enumerate(SER):
    a1.bar(x + (k - 2) * w, [val[c] - d['g'][c] for c in CL], w, color=col, edgecolor="black", lw=0.5, hatch=hat, label=name, zorder=2)
for j, c in enumerate(CL):
    vpt = pt[c] - d['g'][c]
    a1.text(x[j] + (3 - 2) * w, 0.35, f"{vpt:.2f}".replace("-", "−"), ha="center", va="bottom", fontsize=6.3, rotation=90, color="#084A91")
a1.set_yscale("symlog", linthresh=1.0); a1.axhline(0, color="black", lw=0.7)
a1.set_yticks([-100, -10, -1, 0, 1, 10]); a1.set_yticklabels(["−100", "−10", "−1", "0", "1", "10"]); a1.set_ylim(-250, 12)
a1.set_xticks(x, ["Small (0.5 MW, 2 MWh)", "Medium (1.0 MW, 2 MWh)", "Large (2.5 MW, 2.5 MWh)"])
a1.set_ylabel("Difference from\nThreshold-Free Value [k€]"); a1.grid(axis="y", alpha=0.3, zorder=0); a1.set_title("(a)", pad=4)
lg = a1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=5, frameon=True, fancybox=False, edgecolor="black", framealpha=1,
               handlelength=1.6, columnspacing=1.0); lg.get_frame().set_linewidth(0.7)
sc = d['scarce']; gS = d['g']['small']; saS = d['sa']['small']
rows = [("One Small and\nOne Large Unit", sc['1s1l']), ("One Small and\nTwo Large Units", sc['1s2l'])]
CORE = "#FABE78"; SHC = "#9B4DDB"
xl2 = (0, 200); xl3 = (140, 185)
for ax_, xl in ((a2, xl2), (a3, xl3)):
    for k, (lab, r) in enumerate(rows):
        yk = 1 - k; lo, hi = r['core']
        ax_.plot([max(lo, xl[0]), min(hi, xl[1])], [yk, yk], color=CORE, lw=7, solid_capstyle="butt", zorder=2)
        if xl[0] <= r['shapley_small'] <= xl[1]:
            ax_.plot([r['shapley_small']], [yk], marker="o", ms=6.5, color=SHC, mec="black", mew=0.6, zorder=3)
        if xl[0] <= r['nucleolus_small'] <= xl[1]:
            ax_.plot([r['nucleolus_small']], [yk], marker="D", ms=4.2, color="white", mec="black", mew=0.8, zorder=4)
    ax_.axvline(gS, color="black", ls="--", lw=0.8)
    if xl[0] <= saS <= xl[1]: ax_.axvline(saS, color="black", ls=":", lw=0.8)
    ax_.set_yticks([1, 0], [r[0] for r in rows] if ax_ is a2 else ["", ""]); ax_.set_ylim(-0.6, 1.6); ax_.set_xlim(*xl)
    ax_.set_xlabel("Profit of the Small Unit [k€]"); ax_.grid(axis="x", alpha=0.3, zorder=0)
for k, (lab, r) in enumerate(rows):
    yk = 1 - k; lo, hi = r['core']
    a2.text(r['shapley_small'], yk + 0.22, f"{r['shapley_small']:.1f}", ha="center", va="bottom", fontsize=7.5)
    if lo >= xl3[0]: a3.text(lo + 0.4, yk - 0.22, f"{lo:.1f}", ha="left", va="top", fontsize=7.5)
    else: a3.text(xl3[0] + 0.4, yk, "//", ha="left", va="center", fontsize=8, fontweight="bold", color="white")
    a3.text(hi + (0.6 if k == 0 else -0.4), yk + (-0.22 if k == 0 else 0.2), f"{hi:.1f}", ha="left" if k == 0 else "right", va="top" if k == 0 else "bottom", fontsize=7.5)
a2.text(saS + 2, -0.5, "Standalone", fontsize=7.5, va="bottom"); a2.text(saS + 2, 1.25, f"{saS:.1f}", fontsize=7.5)
a3.text(gS - 0.4, 1.42, f"Threshold-Free Value {gS:.1f}", fontsize=7, va="center", ha="right")
a2.set_title("(b)", pad=4); a3.set_title("(c)", pad=4)
hb = [Patch(facecolor=CORE, edgecolor="none", label="Core"), Line2D([], [], marker="o", ls="", ms=6.5, color=SHC, mec="black", label="Shapley Value"), Line2D([], [], marker="D", ls="", ms=4.2, color="white", mec="black", label="Nucleolus"),
      Line2D([], [], color="black", ls=":", lw=0.8, label="Standalone"), Line2D([], [], color="black", ls="--", lw=0.8, label="Threshold-Free Value")]
lg2 = fig.legend(handles=hb, loc="lower center", bbox_to_anchor=(0.5, a2.get_position().y1 + 0.045), ncol=5, frameon=True, fancybox=False, edgecolor="black",
                 framealpha=1, handlelength=1.8, columnspacing=1.0); lg2.get_frame().set_linewidth(0.7)
fig.savefig(out('Fig4_Game.png'), dpi=500, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("Fig. 4 written; pass-through minus threshold-free [kEUR]:", {c: round(float(pt[c] - d["g"][c]), 2) for c in CL})
