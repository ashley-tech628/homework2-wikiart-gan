"""Draw comparable-layout (but distinct-objective) GAN loss panels."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "full_datahub_losses.csv"
TRANSFER = ROOT / "results" / "stylegan2_final" / "losses.csv"
OUT = ROOT / "latex" / "loss_curves.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def smoothed(rows, key, window):
    val = np.array([float(r[key]) for r in rows])
    kernel = np.ones(window) / window
    smooth = np.convolve(np.pad(val, (window - 1, 0), mode="edge"),
                         kernel, mode="valid")
    return val, smooth


baseline, transfer = read(BASE), read(TRANSFER)
bx = np.array([int(r["step"]) for r in baseline]) / 1000
tx = np.array([int(r["nimg"]) for r in transfer]) / 1000
fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.83), dpi=260)
series = [
    (axes[0], bx, baseline, "loss_d", "loss_g", 250,
     "(a) 128px conditional GAN", "Steps (thousands)", "Hinge loss"),
    (axes[1], tx, transfer, "D", "G", 50,
     "(b) 1024px StyleGAN2 transfer", "Fine-tuning (kimg)", "Logistic loss"),
]
for ax, x, rows, dk, gk, window, title, xlabel, ylabel in series:
    for key, color, name in [(dk, "#246889", "D"), (gk, "#BA5D3B", "G")]:
        raw, smooth = smoothed(rows, key, window)
        ax.plot(x, raw, color=color, lw=.25, alpha=.15, rasterized=True)
        ax.plot(x, smooth, color=color, lw=1.15, label=name)
    ax.set_title(title, fontsize=8.3, fontweight="bold", pad=5)
    ax.set_xlabel(xlabel, fontsize=7.3, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=7.3, labelpad=1)
    ax.tick_params(axis="both", labelsize=6.7, width=.5, length=2)
    ax.grid(color="#d9dfe3", linewidth=.35)
    ax.legend(loc="upper right", ncol=2, frameon=False,
              fontsize=6.7, columnspacing=.5, handlelength=1.4)
    ax.set_xlim(x.min(), x.max())
    for spine in ax.spines.values():
        spine.set_linewidth(.55)
fig.subplots_adjust(left=.093, right=.985, top=.83, bottom=.23, wspace=.38)
fig.savefig(OUT, dpi=260, facecolor="white")
plt.close(fig)
print(OUT)
