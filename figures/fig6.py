"""Fig. 6: value of aggregation from the preliminary to the reference formulation, one change at a time."""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import W, out, res

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})
#Lorenzo Giannuzzo: value of aggregation from the preliminary formulation (b0) to the reference (b9), one change at a time
STEPS = ["Preliminary Formulation", "Year-Specific MSD Statistics", "Cyclic SOC", "Degradation at 100 €/kWh", "Degradation on Expected Energy",
         "Activation Tied to the Band", "Sustain with Losses", "Headroom per Direction", "No Up/Down Exclusion", "Directional FCR", "Reference Formulation"]
SC = pd.read_csv(res("scenarios.csv"))
BRIDGE = ["b0_preliminary", "b1_yearly_msd", "b2_cyclic", "b3_degradation", "b4_throughput", "b5_tied", "b6_sustain",
          "b7_headroom", "b8_no_mutex", "b9_reference"]
#Lorenzo Giannuzzo: value of aggregation after each cumulative change, rounded to 0.1 kEUR as in the paper
V = {yr: [round(float(SC[(SC.scenario == b) & (SC.year == yr)].delta_keur.iloc[0]), 1) for b in BRIDGE] for yr in (2023, 2024)}
for yr in V:
    V[yr].append(V[yr][-1])
fig, axs = plt.subplots(1, 2, figsize=(W, 3.9), sharey=True, gridspec_kw=dict(wspace=0.08))
y = np.arange(len(STEPS))[::-1]
for ax, yr in zip(axs, (2023, 2024)):
    v = V[yr]
    for k in range(len(v)):
        if k in (0, len(v) - 1):
            ax.barh(y[k], v[k], 0.62, color="#084A91", edgecolor="black", lw=0.5, zorder=2)
            ax.text(v[k] + 25, y[k], f"{v[k]:.0f}", va="center", fontsize=7.5, fontweight="bold")
        else:
            d = v[k] - v[k - 1]; left = min(v[k], v[k - 1])
            ax.barh(y[k], abs(d), 0.62, left=left, color="#F4A340" if d > 0 else "#5B9BD5", edgecolor="black", lw=0.5, zorder=2)
            ax.text(max(v[k], v[k - 1]) + 25, y[k], f"{d:+.0f}".replace("-", "−"), va="center", fontsize=7.5)
        if k < len(v) - 1: ax.plot([v[k], v[k]], [y[k] - 0.31, y[k + 1] + 0.31], color="black", lw=0.6, ls=":", zorder=1)
    ax.set_xlim(0, 2300); ax.grid(axis="x", alpha=0.3, zorder=0); ax.set_title(f"({'a' if yr == 2023 else 'b'}) {yr}")
    ax.set_xlabel("Value of Aggregation [k€]")
axs[0].set_yticks(y, STEPS)
h = [Patch(facecolor="#084A91", edgecolor="black", lw=0.5, label="Preliminary & Reference"), Patch(facecolor="#F4A340", edgecolor="black", lw=0.5, label="Increase"),
     Patch(facecolor="#5B9BD5", edgecolor="black", lw=0.5, label="Decrease")]
lg = fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.55, 0.98), ncol=3, frameon=True, fancybox=False, edgecolor="black", framealpha=1)
lg.get_frame().set_linewidth(0.7)
#Lorenzo Giannuzzo: legend centered on the whole saved figure, tick labels included, not on the plotting area
fig.canvas.draw()
for _ in range(3):
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    xc = (bb.x0 + bb.x1) / 2 / fig.get_size_inches()[0]
    lg.set_bbox_to_anchor((xc, 0.98), transform=fig.transFigure); fig.canvas.draw()
fig.savefig(out("Fig6_Bridge.png"), dpi=500, bbox_inches="tight", facecolor="white"); plt.close(fig)

print("Fig. 6 written:", V)
