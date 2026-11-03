"""Fig. 2 (revenue and cost components by class) and Fig. 3 (shares of the value of aggregation under the four rules)."""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import CL, N, W, out, res, style

PHI_HALF = 0.44   # day-ahead settlement of the expected FCR activation, per direction, in the stored coefficients
style()


#Lorenzo Giannuzzo: reserve revenue of each class by product, from the stored schedules
def class_rev(npz, mult=None):
    d=np.load(npz); lab=list(d['labels']); m=d['mult']
    R=d['R']; a=d['alpha']; b=d['beta']; cc=d['ccap']; ce=d['cen']; pr=d['prices']
    out={}
    for ci,c in enumerate(lab):
        Rc=R[:,:,ci,:]   # total of the class (merged unit)
        rev=[]
        for k in range(6):
            cap=(cc[:,k]*a[:,k]*Rc[:,k]).sum()
            if k<2: e=PHI_HALF*(pr*a[:,k]*b[:,k]*Rc[:,k]).sum()
            else:   e=(ce[:,k]*a[:,k]*b[:,k]*Rc[:,k]).sum()
            sgn=1 if k%2==0 else -1
            rev.append(cap+sgn*e)
        out[str(c)]=np.array(rev)*(1 if mult is None else mult[str(c)])/1e3
    return out
def load(y):
    s=pd.read_csv(res('scenarios.csv')); r=s[(s.scenario=='ref')&(s.year==y)].iloc[0]
    u=json.load(open(res(f'units_ref_{y}.json')))
    agg=class_rev(res(f'schedules_ref_{y}.npz'))
    sa={c:class_rev(res(f'schedules_ref_{y}_standalone_{c}.npz'),{c:N[c]})[c] for c in CL}
    lab=u['labels']
    agg_arb={c:sum(x for x,l in zip(u['agg_arb'],lab) if l==c)/1e3 for c in CL}
    agg_deg={c:sum(x for x,l in zip(u['agg_deg'],lab) if l==c)/1e3 for c in CL}
    sa_arb={c:r[f'sa_{c}_arb_keur'] for c in CL}; sa_deg={c:r[f'sa_{c}_deg_keur'] for c in CL}
    # checks
    aggu={c:np.array([x for x,l in zip(u['agg_rev_by_unit'],lab) if l==c]).sum(axis=0)/1e3 for c in CL}
    for c in CL: assert np.allclose(aggu[c],agg[c],atol=0.05),(c,aggu[c],agg[c])
    sa_tot=sum(sa[c].sum()+sa_arb[c]-sa_deg[c] for c in CL); ag_tot=sum(agg[c].sum()+agg_arb[c]-agg_deg[c] for c in CL)
    return dict(r=r,u=u,agg=agg,sa=sa,agg_arb=agg_arb,agg_deg=agg_deg,sa_arb=sa_arb,sa_deg=sa_deg,sa_tot=sa_tot,ag_tot=ag_tot)
CC = {"small": "#F4A340", "medium": "#5B9BD5", "large": "#084A91"}
NAME = {"small": "Small (0.5 MW, 2 MWh)", "medium": "Medium (1.0 MW, 2 MWh)", "large": "Large (2.5 MW, 2.5 MWh)"}
D = {y: load(y) for y in (2023, 2024)}

#Lorenzo Giannuzzo: Fig. 2, revenue and cost components of the portfolio, standalone (hatched) against aggregated (solid), stacked by class
COMP = [("FCR", lambda r, a, d: r[0] + r[1]), ("aFRR\nUpward", lambda r, a, d: r[2]), ("aFRR\nDownward", lambda r, a, d: r[3]),
        ("Arbitrage", lambda r, a, d: a), ("Degradation", lambda r, a, d: -d)]
fig, axs = plt.subplots(1, 2, figsize=(W, 3.5), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, y in zip(axs, (2023, 2024)):
    d = D[y]; x = np.array([0.0, 1.0, 2.0, 3.0, 4.15]); w = 0.36
    for side, off in (("sa", -w / 2), ("ag", w / 2)):
        pos = np.zeros(len(COMP)); neg = np.zeros(len(COMP))
        for c in CL:
            rev = d[side][c] if side == "sa" else d["agg"][c]
            arb = d["sa_arb" if side == "sa" else "agg_arb"][c]; deg = d["sa_deg" if side == "sa" else "agg_deg"][c]
            v = np.array([f(rev, arb, deg) for _, f in COMP])
            bot = np.where(v >= 0, pos, neg)
            ax.bar(x + off, v, w, bottom=bot, color=CC[c], edgecolor="black", lw=0.5, hatch="////" if side == "sa" else None,
                   alpha=0.55 if side == "sa" else 1.0, zorder=2)
            pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
        if side == "sa": tot_sa = pos + neg
        else: tot_ag = pos + neg
    for k in range(len(COMP)):
        dv = tot_ag[k] - tot_sa[k]; top = max(tot_sa[k], tot_ag[k], 0)
        ax.text(x[k], top + 70, f"{dv:+.1f}".replace("-", "−"), ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8); ax.set_xticks(x, [n for n, _ in COMP], fontsize=6); ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_title(f"({'a' if y == 2023 else 'b'}) {y}")
    ax.text(0.98, 0.97, f"Value of Aggregation\n{d['r'].delta_keur:.0f} k€ ({d['r'].delta_pct:.1f}%)", transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="black", lw=0.6))
axs[0].set_ylabel("Annual Revenue or Cost [k€]"); axs[0].set_ylim(-600, 3800)
h = [Patch(facecolor=CC[c], edgecolor="black", lw=0.5, label=NAME[c]) for c in CL] + \
    [Patch(facecolor="white", edgecolor="black", hatch="////", lw=0.5, label="Standalone"), Patch(facecolor="white", edgecolor="black", lw=0.5, label="Aggregated")]
lg = fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=5, frameon=True, fancybox=False, edgecolor="black",
                framealpha=1, handlelength=1.6, columnspacing=1.0, fontsize=7.5); lg.get_frame().set_linewidth(0.7)
fig.tight_layout(); fig.savefig(out("Fig2_Components.png"), dpi=500, bbox_inches="tight", facecolor="white"); plt.close(fig)

#Lorenzo Giannuzzo: Fig. 3, share of the value of aggregation per class under the four rules, with rating and energy of each class
RULES = [("power", "By Rated Power", plt.get_cmap("Blues")(0.3), "//"), ("band", "By Band Supplied", plt.get_cmap("Blues")(0.5), ".."),
         ("floor", "Reserve Floor", plt.get_cmap("Blues")(0.7), None), ("passthru", "Pass-Through", plt.get_cmap("Blues")(0.92), None)]
fig, axs = plt.subplots(1, 2, figsize=(W, 3.3), sharey=True)
for ax, y in zip(axs, (2023, 2024)):
    u = D[y]["u"]; lab = u["labels"]; sa = u["standalone"]; dl = D[y]["r"].delta_keur
    x = np.arange(3); w = 0.2
    for k, (key, name, col, hat) in enumerate(RULES):
        al = u["allocations"][key]
        sh = [sum(a - s for a, s, l in zip(al, sa, lab) if l == c) / 1e3 / dl * 100 for c in CL]
        ax.bar(x + (k - 1.5) * w, sh, w, color=col, edgecolor="black", lw=0.5, hatch=hat, label=name, zorder=2)
    ax.axhline(0, color="black", lw=0.8); ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_xticks(x, ["Small\n(0.5 MW,\n2 MWh)", "Medium\n(1.0 MW,\n2 MWh)", "Large\n(2.5 MW,\n2.5 MWh)"], fontsize=7.5)
    ax.set_title(f"({'a' if y == 2023 else 'b'}) {y}"); ax.set_ylim(0, 100)
axs[0].set_ylabel("Share of the Value of Aggregation [%]")
h, l = axs[0].get_legend_handles_labels()
lg = fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=True, fancybox=False, edgecolor="black", framealpha=1)
lg.get_frame().set_linewidth(0.7)
fig.tight_layout(); fig.savefig(out("Fig3_Shares.png"), dpi=500, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("Fig. 2 and Fig. 3 written")
