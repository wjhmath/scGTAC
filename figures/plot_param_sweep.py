from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "generated_tables" / "iter2_param_sweep_summary.csv"
OUT = ROOT / "manuscript" / "figures" / "supplementary"
ORDER = {"tau": [0.5, 0.8, 1.0], "dropout": [0.0, 0.3, 0.5], "lambda_cl": [0.3, 0.7, 1.0]}
TITLES = {"tau": r"Temperature $\tau$", "dropout": "Dropout", "lambda_cl": r"Contrastive weight $\lambda_{cl}$"}

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
    rows = load_rows()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)

    for ax, param in zip(axes, ["tau", "dropout", "lambda_cl"]):
        subset = [row for row in rows if row["param_name"] == param]
        subset.sort(key=lambda row: ORDER[param].index(float(row["param_value"])))
        x = [float(row["param_value"]) for row in subset]
        y = [float(row["ARI_mean"]) for row in subset]
        yerr = [float(row["ARI_std"]) for row in subset]
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, color="#4c72b0", capsize=3)
        ax.set_title(TITLES[param])
        ax.set_xlabel("Value")
        ax.set_ylabel("ARI")
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Baron parameter sensitivity", fontsize=12, fontweight="bold")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figS2_baron_param_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
