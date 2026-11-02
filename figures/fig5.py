"""Fig. 5: width of the core of every composition, projected on the payment of one small unit
(python fig5.py 2023 for the paper, python fig5.py 2024 for the Supplementary Material)."""
import json
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from common import W, out, res

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.5, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
Y = sys.argv[1] if len(sys.argv) > 1 else '2023'
P = json.load(open(res(f'core_width_small_{Y}.json')))
def width(s, m, l):
    k = str((s, m, l))
    if k not in P: return np.nan
    lo, hi, e = P[k]; return hi - lo
#Lorenzo Giannuzzo: width of the core projected on one small unit, as the partners of other classes and the other small units grow
fig, axs = plt.subplots(1, 2, figsize=(W, 3.9), gridspec_kw=dict(width_ratios=[1.2, 1.3], wspace=0.25))
norm = LogNorm(vmin=0.8, vmax=300); from matplotlib.colors import ListedColormap
cmap = ListedColormap(matplotlib.colormaps["magma_r"](np.linspace(0.0, 0.82, 256)))
grids = [("(a) Small and Large Units", [(s, 0, l) for l in range(5) for s in range(1, 9)], 5, "Large Units [-]"),
         ("(b) Small and Medium Units", [(s, m, 0) for m in range(9) for s in range(1, 9)], 9, "Medium Units [-]")]
for ax, (title, cells, ny, ylab) in zip(axs, grids):
    Z = np.full((ny, 8), np.nan)
    for (s, m, l) in cells:
        j = l if ny == 5 else m
        Z[j, s - 1] = width(s, m, l)
    im = ax.imshow(Z, origin="lower", cmap=cmap, norm=norm, aspect="auto")
    for j in range(ny):
        for i in range(8):
            if np.isnan(Z[j, i]):
                ax.text(i, j, "–", ha="center", va="center", fontsize=7); continue
            z = Z[j, i]; txt = f"{z:.1f}"
            ax.text(i, j, txt, ha="center", va="center", fontsize=6.0, color="white" if z >= 10.5 else "black")
    ax.set_xticks(range(8), [str(s) for s in range(1, 9)]); ax.set_yticks(range(ny), [str(k) for k in range(ny)])
    ax.set_xlabel("Small Units [-]"); ax.set_ylabel(ylab); ax.set_title(title)
    for sp in ax.spines.values(): sp.set_edgecolor("black"); sp.set_linewidth(0.6)
fig.subplots_adjust(top=0.78)
p0, p1 = axs[0].get_position(), axs[1].get_position()
xc = (p0.x0 + p1.x1) / 2; wcb = 0.42
cax = fig.add_axes([xc - wcb / 2, 0.905, wcb, 0.035])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_ticks([1, 10, 100]); cb.set_ticklabels(["1", "10", "100"]); cb.minorticks_off()
cax.xaxis.set_ticks_position("bottom"); cax.xaxis.set_label_position("top")
cb.set_label("Core Width for One Small Unit [k€]"); cb.outline.set_edgecolor("black"); cb.outline.set_linewidth(0.8)
fig.savefig(out("Fig5_CoreWidth.png" if Y == "2023" else f"FigS_CoreWidth_{Y}.png"), dpi=500, bbox_inches="tight", facecolor="white"); plt.close(fig)
print(f"Fig. 5 ({Y}) written")
