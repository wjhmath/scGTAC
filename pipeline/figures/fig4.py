from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/liyang/BioJiaheWang/scAGCR")
INPUT = ROOT / "generated_tables" / "iter2_ablation_summary.csv"
OUT = ROOT / "generated_figures"
ORDER = ["baseline", "no_cl", "no_cluster", "no_edge_aug", "no_feature_aug", "no_zinb"]
LABELS = {
    "baseline": "Baseline",
    "no_cl": "No CL",
    "no_cluster": "No cluster",
    "no_edge_aug": "No edge aug",
    "no_feature_aug": "No feat aug",
    "no_zinb": "No ZINB",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_rows() -> list[dict[str, str]]:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row["dataset"] == "baron"]


def main() -> None:
    rows = {row["variant_name"]: row for row in load_rows()}
    metrics = ["ACC", "NMI", "ARI"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    x = np.arange(len(ORDER))

    for ax, metric in zip(axes, metrics):
        means = [float(rows[name][f"{metric}_mean"]) for name in ORDER]
        stds = [float(rows[name][f"{metric}_std"]) for name in ORDER]
        colors = ["#c44e52" if name == "no_cl" else ("#55a868" if name == "baseline" else "#4c72b0") for name in ORDER]
        ax.bar(x, means, yerr=stds, color=colors, capsize=3, alpha=0.9)
        ax.set_title(metric)
        ax.set_xticks(x, [LABELS[name] for name in ORDER], rotation=25, ha="right")
        ax.set_ylim(0.6 if metric == "ARI" else 0.65, max(means) + 0.08)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Baron ablation study", fontsize=12, fontweight="bold")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig4_baron_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
