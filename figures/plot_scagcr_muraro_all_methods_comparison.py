from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated_figures"
OUT.mkdir(parents=True, exist_ok=True)

FROZEN_ROWS = [
    {"method":"iter2_repro","ACC_mean":0.9124,"NMI_mean":0.8894,"ARI_mean":0.9316},
    {"method":"leiden","ACC_mean":0.9205,"NMI_mean":0.8772,"ARI_mean":0.8722},
    {"method":"scmae","ACC_mean":0.8612,"NMI_mean":0.8071,"ARI_mean":0.8513},
    {"method":"dec","ACC_mean":0.6618,"NMI_mean":0.6323,"ARI_mean":0.5089},
    {"method":"scdcc","ACC_mean":0.6220,"NMI_mean":0.5595,"ARI_mean":0.4235},
    {"method":"scdsc","ACC_mean":0.5895,"NMI_mean":0.4786,"ARI_mean":0.3241},
    {"method":"scgpt_kmeans","ACC_mean":0.4282,"NMI_mean":0.3873,"ARI_mean":0.2436},
    {"method":"scgpt_leiden","ACC_mean":0.4164,"NMI_mean":0.4086,"ARI_mean":0.2436},
]

plt.style.use(["science", "nature", "no-latex"])
plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

HIGHLIGHT = "#2A9D8F"
BASE = "#8D99AE"
ACC_C = "#4C78A8"
NMI_C = "#F2A541"
ARI_C = "#2A9D8F"

DISPLAY = {
    "iter2_repro": "scAGCR",
    "leiden": "Leiden",
    "scmae": "scMAE",
    "dec": "DEC",
    "scdcc": "scDCC",
    "scdsc": "scDSC",
    "scgpt_kmeans": "scGPT-KMeans",
    "scgpt_leiden": "scGPT-Leiden",
}
ORDER = ["iter2_repro", "leiden", "scmae", "dec", "scdcc", "scdsc", "scgpt_kmeans", "scgpt_leiden"]


def load_data() -> pd.DataFrame:
    df = pd.DataFrame(FROZEN_ROWS)
    df["display"] = df["method"].map(DISPLAY)
    df["order"] = df["method"].map({m: i for i, m in enumerate(ORDER)})
    return df.sort_values("order").reset_index(drop=True)


def style_bars(ax, bars, labels):
    for bar, lab in zip(bars, labels):
        bar.set_color(HIGHLIGHT if lab == "scAGCR" else BASE)
        if lab == "scAGCR":
            bar.set_edgecolor("#1f6f66")
            bar.set_linewidth(1.2)


def add_values(ax, vals):
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7)


def plot_two_col(df: pd.DataFrame):
    labels = df["display"].tolist()
    y = np.arange(len(df))
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), sharey=True)
    axes = axes.ravel()

    metrics = [("ACC_mean", "ACC", ACC_C), ("NMI_mean", "NMI", NMI_C), ("ARI_mean", "ARI", ARI_C)]
    for ax, (col, title, color) in zip(axes[:3], metrics):
        bars = ax.barh(y, df[col].to_numpy(), color=color)
        style_bars(ax, bars, labels)
        add_values(ax, df[col].to_numpy())
        ax.set_title(title)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, min(1.0, df[col].max() + 0.12))

    ax = axes[3]
    best_ari = df["ARI_mean"].max()
    delta = best_ari - df["ARI_mean"].to_numpy()
    bars = ax.barh(y, delta, color="#C75C5C")
    for i, bar in enumerate(bars):
        if labels[i] == "scAGCR":
            bar.set_color(HIGHLIGHT)
    for i, v in enumerate(delta):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=7)
    ax.set_title("Gap to best ARI")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(delta) + 0.08)

    fig.suptitle("Muraro pancreas: all-method comparison (two-column layout)", y=0.995, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_case_muraro_all_methods_2col.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_three_col(df: pd.DataFrame):
    labels = df["display"].tolist()
    x = np.arange(len(df))
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.0), sharex=True)

    metrics = [("ACC_mean", "ACC", ACC_C), ("NMI_mean", "NMI", NMI_C), ("ARI_mean", "ARI", ARI_C)]
    for ax, (col, title, color) in zip(axes, metrics):
        bars = ax.bar(x, df[col].to_numpy(), color=[HIGHLIGHT if m == "scAGCR" else color for m in labels])
        for i, v in enumerate(df[col].to_numpy()):
            ax.text(i, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_ylabel(title)
        ax.set_ylim(0, min(1.0, df[col].max() + 0.12))
        ax.set_title(f"Muraro pancreas: {title}")

    axes[-1].set_xticks(x, labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_case_muraro_all_methods_3col.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_data()
    plot_two_col(df)
    plot_three_col(df)
    print(df[["display", "ACC_mean", "NMI_mean", "ARI_mean"]].to_string(index=False))
    print(f"Saved {(OUT / 'fig_case_muraro_all_methods_2col.pdf')}")
    print(f"Saved {(OUT / 'fig_case_muraro_all_methods_3col.pdf')}")


if __name__ == "__main__":
    main()
