"""Plot raw and smoothed losses saved by train.py."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smooth(values, window=25):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window)
    return np.convolve(values, kernel, mode="same") / np.convolve(np.ones_like(values), kernel, mode="same")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--losses", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with args.losses.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    steps = [int(row["step"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=180)
    for key, color, label in (("loss_d", "#1e6091", "Discriminator"),
                              ("loss_g", "#c45d39", "Generator")):
        values = np.array([float(row[key]) for row in rows])
        ax.plot(steps, values, color=color, alpha=0.18, linewidth=0.7)
        ax.plot(steps, smooth(values), color=color, linewidth=1.7, label=label)
    ax.set(xlabel="Training step", ylabel="Hinge objective")
    ax.grid(alpha=0.16)
    ax.legend(frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
