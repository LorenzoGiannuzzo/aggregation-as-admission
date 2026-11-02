"""Fig. S1 of the Supplementary Material: hourly award probabilities and energy prices of the
reserve products as they enter the scheduling model (annual means of each hour)."""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import W, out, style

style()
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
PROD = [("fcr_up", "FCR (Each Direction)", "#084A91", "-"), ("afrr_up", "aFRR Upward", "#FF8A1F", "-"),
        ("afrr_dn", "aFRR Downward", "#FF8A1F", "--"), ("mfrr_up", "mFRR Upward", "#9B4DDB", "-"),
        ("mfrr_dn", "mFRR Downward", "#9B4DDB", "--")]
#Lorenzo Giannuzzo: two rows (award probability, energy price) by two columns (2023, 2024)
fig, axs = plt.subplots(2, 2, figsize=(W, 4.6), sharex=True, gridspec_kw=dict(hspace=0.28, wspace=0.22))
for c, y in enumerate((2023, 2024)):
    d = pd.read_csv(os.path.join(DATA, f"market_inputs_{y}.csv"))
    g = d.groupby("hour").mean(numeric_only=True)
    h = g.index.to_numpy()
    for key, name, col, ls in PROD:
        axs[0, c].plot(h, g[f"{key}_award_probability"], color=col, ls=ls, lw=1.3, label=name)
        if key.startswith("afrr"):
            #Lorenzo Giannuzzo: mean energy price over the hours with a positive award probability (mFRR takes the same prices)
            e = d[d[f"{key}_award_probability"] > 0].groupby("hour")[f"{key}_energy_price_eur_mwh"].mean()
            axs[1, c].plot(e.index, e.values, color=col, ls=ls, lw=1.3)
    axs[1, c].plot(h, g["day_ahead_price_eur_mwh"], color="black", lw=1.0, ls=":", label="Day-Ahead Price")
    axs[0, c].set_title(f"({'a' if y == 2023 else 'b'}) {y}"); axs[1, c].set_title(f"({'c' if y == 2023 else 'd'}) {y}")
    axs[0, c].set_ylim(0, 1); axs[1, c].set_ylim(0, 300)
    axs[1, c].set_xlabel("Hour of the Day [h]"); axs[1, c].set_xticks(range(0, 24, 3))
    for r in (0, 1):
        axs[r, c].grid(alpha=0.3)
axs[0, 0].set_ylabel("Award Probability [-]"); axs[1, 0].set_ylabel("Energy Price [€/MWh]")
h1, l1 = axs[0, 0].get_legend_handles_labels(); h2, l2 = axs[1, 0].get_legend_handles_labels()
lg = fig.legend(h1 + h2[-1:], l1 + l2[-1:], loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=3, frameon=True, fancybox=False,
                edgecolor="black", framealpha=1, fontsize=7.5)
lg.get_frame().set_linewidth(0.7)
fig.savefig(out("FigS1_Inputs.png"), dpi=500, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("Fig. S1 written")
