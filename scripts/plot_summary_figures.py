#!/usr/bin/env python3
"""Draw descriptive figures from the frozen paper summary table."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
with (ROOT / "results/final_master_summary.csv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

layers = [int(r["layers"]) for r in rows]
pca = [int(r["pca_dim_99"]) for r in rows]
ae = [int(r["ae_dim_99"]) for r in rows]
samples = [float(r["minimum_training_samples"]) for r in rows]
capacity = [float(r["min_parameters_r2_99"]) for r in rows[:-1]]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300,
})


def finish(fig, filename):
    fig.savefig(OUT / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(6.2, 3.75), constrained_layout=True)
ax.plot(layers, pca, "o-", label="PCA, 99% variance", color="#176995")
ax.plot(layers, ae, "s-", label="Autoencoder, R² ≥ 0.99", color="#CA5B19")
ax.set(xlabel="Nominal layer count", ylabel="Components or latent coordinates",
       xticks=layers, ylim=(0, 58))
ax.grid(alpha=0.22)
ax.legend(frameon=False, fontsize=8, loc="upper left")
finish(fig, "Figure1_intrinsic_dimension_vs_layers.png")


def resource_plot(x, y, x_label, y_label, xlim, filename, censored_samples=False):
    fig, ax = plt.subplots(figsize=(6.2, 3.75), constrained_layout=True)
    ax.scatter(x, y, s=42, color="#176995")
    for index, (xi, yi, layer) in enumerate(zip(x, y, layers)):
        if censored_samples and index < 2:
            ax.annotate("", xy=(xi, 700), xytext=(xi, 950),
                        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#176995"})
        else:
            ax.annotate(f"{layer}L", (xi, yi), xytext=(5, 3),
                        textcoords="offset points", fontsize=8)
    if censored_samples:
        ax.annotate("8L and 10L: ≤1,000", (x[1] + 1, 1450), fontsize=7.5)
        ax.set_ylim(550, 51000)
    ax.set(xlabel=x_label, ylabel=y_label, yscale="log", xlim=xlim)
    ax.grid(axis="y", alpha=0.22)
    finish(fig, filename)


resource_plot(ae, samples, "Autoencoder bottleneck threshold",
              "Minimum tested training samples", (3.5, 32.5),
              "Figure3_ae_dimension_vs_sample_requirement.png", True)
resource_plot(pca, samples, "PCA components for 99% variance",
              "Minimum tested training samples", (6, 56),
              "Figure4_pca_dimension_vs_sample_requirement.png", True)
resource_plot(ae[:-1], capacity, "Autoencoder bottleneck threshold",
              "Minimum tested trainable parameters", (4, 22),
              "Figure5_ae_dimension_vs_model_capacity.png")
