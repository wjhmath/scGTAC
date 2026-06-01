from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_MAIN = ROOT / "generated_figures"

FROZEN_ROWS = [
    {"dataset":"baron","method":"agrv4","ACC_mean":0.6618,"ACC_std":0.0504,"NMI_mean":0.8030,"NMI_std":0.0162,"ARI_mean":0.6692,"ARI_std":0.0813},
    {"dataset":"baron","method":"dec","ACC_mean":0.7545,"ACC_std":0.0373,"NMI_mean":0.7495,"NMI_std":0.0158,"ARI_mean":0.6729,"ARI_std":0.0492},
    {"dataset":"baron","method":"iter2_repro","ACC_mean":0.7361,"ACC_std":0.0235,"NMI_mean":0.8473,"NMI_std":0.0152,"ARI_mean":0.7792,"ARI_std":0.0124},
    {"dataset":"baron","method":"leiden","ACC_mean":0.7249,"ACC_std":0.0085,"NMI_mean":0.8431,"NMI_std":0.0051,"ARI_mean":0.6737,"ARI_std":0.0043},
    {"dataset":"baron","method":"scsimgcl","ACC_mean":0.7221,"ACC_std":0.0050,"NMI_mean":0.8222,"NMI_std":0.0050,"ARI_mean":0.7643,"ARI_std":0.0043},
    {"dataset":"baron","method":"scagcl","ACC_mean":0.5814,"ACC_std":0.0091,"NMI_mean":0.7808,"NMI_std":0.0032,"ARI_mean":0.5513,"ARI_std":0.0039},
    {"dataset":"baron","method":"scdcc","ACC_mean":0.6155,"ACC_std":0.0240,"NMI_mean":0.6180,"NMI_std":0.0227,"ARI_mean":0.4008,"ARI_std":0.0158},
    {"dataset":"baron","method":"scdsc","ACC_mean":0.4889,"ACC_std":0.0233,"NMI_mean":0.4435,"NMI_std":0.0721,"ARI_mean":0.2897,"ARI_std":0.0321},
    {"dataset":"baron","method":"scgnn","ACC_mean":0.5841,"ACC_std":0.0,"NMI_mean":0.5525,"NMI_std":0.0,"ARI_mean":0.3443,"ARI_std":0.0},
    {"dataset":"baron","method":"scgpt_kmeans","ACC_mean":0.3689,"ACC_std":0.0224,"NMI_mean":0.4504,"NMI_std":0.0037,"ARI_mean":0.2076,"ARI_std":0.0099},
    {"dataset":"baron","method":"scgpt_leiden","ACC_mean":0.4112,"ACC_std":0.0096,"NMI_mean":0.4522,"NMI_std":0.0045,"ARI_mean":0.2439,"ARI_std":0.0035},
    {"dataset":"baron","method":"scmae","ACC_mean":0.6875,"ACC_std":0.0326,"NMI_mean":0.7414,"NMI_std":0.0104,"ARI_mean":0.6283,"ARI_std":0.0411},

    {"dataset":"multiome","method":"agrv4","ACC_mean":0.6352,"ACC_std":0.0,"NMI_mean":0.7740,"NMI_std":0.0,"ARI_mean":0.5546,"ARI_std":0.0},
    {"dataset":"multiome","method":"dec","ACC_mean":0.5851,"ACC_std":0.0303,"NMI_mean":0.7118,"NMI_std":0.0029,"ARI_mean":0.4591,"ARI_std":0.0130},
    {"dataset":"multiome","method":"iter2_repro","ACC_mean":0.6836,"ACC_std":0.0260,"NMI_mean":0.7967,"NMI_std":0.0101,"ARI_mean":0.5733,"ARI_std":0.0355},
    {"dataset":"multiome","method":"leiden","ACC_mean":0.7503,"ACC_std":0.0403,"NMI_mean":0.8213,"NMI_std":0.0154,"ARI_mean":0.6643,"ARI_std":0.0471},
    {"dataset":"multiome","method":"scagcl","ACC_mean":0.6648,"ACC_std":0.0246,"NMI_mean":0.7402,"NMI_std":0.0066,"ARI_mean":0.6031,"ARI_std":0.0489},
    {"dataset":"multiome","method":"scdcc","ACC_mean":0.5228,"ACC_std":0.0502,"NMI_mean":0.6273,"NMI_std":0.0240,"ARI_mean":0.4624,"ARI_std":0.1384},
    {"dataset":"multiome","method":"scdsc","ACC_mean":0.4398,"ACC_std":0.0354,"NMI_mean":0.4567,"NMI_std":0.0427,"ARI_mean":0.3754,"ARI_std":0.0303},
    {"dataset":"multiome","method":"scgpt_kmeans","ACC_mean":0.3685,"ACC_std":0.0297,"NMI_mean":0.4931,"NMI_std":0.0096,"ARI_mean":0.2297,"ARI_std":0.0206},
    {"dataset":"multiome","method":"scgpt_leiden","ACC_mean":0.3616,"ACC_std":0.0133,"NMI_mean":0.5212,"NMI_std":0.0032,"ARI_mean":0.2683,"ARI_std":0.0272},
    {"dataset":"multiome","method":"scmae","ACC_mean":0.5722,"ACC_std":0.0254,"NMI_mean":0.6849,"NMI_std":0.0153,"ARI_mean":0.4508,"ARI_std":0.0284},
    {"dataset":"multiome","method":"scsimgcl","ACC_mean":0.6032,"ACC_std":0.0395,"NMI_mean":0.7601,"NMI_std":0.0125,"ARI_mean":0.4903,"ARI_std":0.0393},

    {"dataset":"muraro_pancreas","method":"dec","ACC_mean":0.6618,"ACC_std":0.1040,"NMI_mean":0.6323,"NMI_std":0.0612,"ARI_mean":0.5089,"ARI_std":0.1619},
    {"dataset":"muraro_pancreas","method":"iter2_repro","ACC_mean":0.9124,"ACC_std":0.0111,"NMI_mean":0.8894,"NMI_std":0.0045,"ARI_mean":0.9316,"ARI_std":0.0043},
    {"dataset":"muraro_pancreas","method":"leiden","ACC_mean":0.9205,"ACC_std":0.0660,"NMI_mean":0.8772,"NMI_std":0.0297,"ARI_mean":0.8722,"ARI_std":0.1114},
    {"dataset":"muraro_pancreas","method":"scdcc","ACC_mean":0.6220,"ACC_std":0.1289,"NMI_mean":0.5595,"NMI_std":0.1007,"ARI_mean":0.4235,"ARI_std":0.1606},
    {"dataset":"muraro_pancreas","method":"scdsc","ACC_mean":0.5895,"ACC_std":0.0424,"NMI_mean":0.4786,"NMI_std":0.0482,"ARI_mean":0.3241,"ARI_std":0.0914},
    {"dataset":"muraro_pancreas","method":"scgpt_kmeans","ACC_mean":0.4282,"ACC_std":0.0012,"NMI_mean":0.3873,"NMI_std":0.0016,"ARI_mean":0.2436,"ARI_std":0.0006},
    {"dataset":"muraro_pancreas","method":"scgpt_leiden","ACC_mean":0.4164,"ACC_std":0.0249,"NMI_mean":0.4086,"NMI_std":0.0051,"ARI_mean":0.2436,"ARI_std":0.0114},
    {"dataset":"muraro_pancreas","method":"scmae","ACC_mean":0.8612,"ACC_std":0.0,"NMI_mean":0.8071,"NMI_std":0.0,"ARI_mean":0.8513,"ARI_std":0.0},

    {"dataset":"zheng68k","method":"agrv4","ACC_mean":0.4799,"ACC_std":0.0057,"NMI_mean":0.4754,"NMI_std":0.0058,"ARI_mean":0.2493,"ARI_std":0.0099},
    {"dataset":"zheng68k","method":"dec","ACC_mean":0.4190,"ACC_std":0.0075,"NMI_mean":0.4473,"NMI_std":0.0025,"ARI_mean":0.2210,"ARI_std":0.0050},
    {"dataset":"zheng68k","method":"iter2_repro","ACC_mean":0.5513,"ACC_std":0.0005,"NMI_mean":0.5130,"NMI_std":0.0005,"ARI_mean":0.3057,"ARI_std":0.0011},
    {"dataset":"zheng68k","method":"leiden","ACC_mean":0.5518,"ACC_std":0.0311,"NMI_mean":0.5083,"NMI_std":0.0096,"ARI_mean":0.3046,"ARI_std":0.0258},
    {"dataset":"zheng68k","method":"scdcc","ACC_mean":0.4937,"ACC_std":0.0129,"NMI_mean":0.4578,"NMI_std":0.0054,"ARI_mean":0.2509,"ARI_std":0.0046},
    {"dataset":"zheng68k","method":"scdsc","ACC_mean":0.4655,"ACC_std":0.0425,"NMI_mean":0.3704,"NMI_std":0.0593,"ARI_mean":0.1753,"ARI_std":0.0104},
    {"dataset":"zheng68k","method":"scgpt_kmeans","ACC_mean":0.1768,"ACC_std":0.0609,"NMI_mean":0.0705,"NMI_std":0.1199,"ARI_mean":0.0309,"ARI_std":0.0538},
    {"dataset":"zheng68k","method":"scgpt_leiden","ACC_mean":0.1607,"ACC_std":0.0603,"NMI_mean":0.0675,"NMI_std":0.1149,"ARI_mean":0.0280,"ARI_std":0.0483},
    {"dataset":"zheng68k","method":"scsimgcl","ACC_mean":0.4715,"ACC_std":0.0073,"NMI_mean":0.4727,"NMI_std":0.0074,"ARI_mean":0.2454,"ARI_std":0.0100},
]

DATASET_ORDER = ["baron", "zheng68k", "multiome", "muraro_pancreas"]
DATASET_LABELS = {
    "baron": "Baron",
    "zheng68k": "Zheng68K",
    "multiome": "Multiome",
    "muraro_pancreas": "Muraro Pancreas",
}
DISPLAY = {
    "iter2_repro": r"scAGCR",
    "leiden": "Leiden",
    "scsimgcl": "scSimGCL",
    "agrv4": "AGR-v4",
    "dec": "DEC",
    "scmae": "scMAE",
    "scagcl": "scAGCL",
    "scdcc": "scDCC",
    "scdsc": "scDSC",
    "scgnn": "scGNN",
    "scgpt_kmeans": "scGPT-KMeans",
    "scgpt_leiden": "scGPT-Leiden",
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


def load_rows() -> list[dict[str, float | str]]:
    return list(FROZEN_ROWS)


def method_order(rows: list[dict[str, float | str]]) -> list[str]:
    methods = sorted({str(row["method"]) for row in rows}, key=lambda m: DISPLAY.get(m, m))
    scores: dict[str, list[float]] = {method: [] for method in methods}
    for row in rows:
        scores[str(row["method"])].append(float(row["ARI_mean"]))
    methods.sort(key=lambda m: (-np.mean(scores[m]), DISPLAY.get(m, m)))
    if "iter2_repro" in methods:
        methods.remove("iter2_repro")
        methods.insert(0, "iter2_repro")
    return methods


def make_heatmap(rows: list[dict[str, float | str]]) -> None:
    methods = method_order(rows)
    lookup = {(str(row["method"]), str(row["dataset"])): row for row in rows}
    fig, axes = plt.subplots(1, 3, figsize=(12, 7.5), constrained_layout=True)
    metrics = ["ACC_mean", "NMI_mean", "ARI_mean"]
    titles = ["ACC", "NMI", "ARI"]
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#f1f1f1")

    for ax, metric, title in zip(axes, metrics, titles):
        matrix = np.full((len(methods), len(DATASET_ORDER)), np.nan)
        for i, method in enumerate(methods):
            for j, dataset in enumerate(DATASET_ORDER):
                row = lookup.get((method, dataset))
                if row is not None:
                    matrix[i, j] = float(row[metric])
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
        ax.set_title(title)
        ax.set_xticks(range(len(DATASET_ORDER)), [DATASET_LABELS[d] for d in DATASET_ORDER], rotation=25, ha="right")
        ax.set_yticks(range(len(methods)), [DISPLAY.get(m, m) for m in methods])
        if title != "ACC":
            ax.set_yticklabels([])
        for i in range(len(methods)):
            for j in range(len(DATASET_ORDER)):
                if not np.isnan(matrix[i, j]):
                    color = "white" if matrix[i, j] < (np.nanmin(matrix) + np.nanmax(matrix)) / 2 else "black"
                    ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7, color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle("Benchmark comparison across representative datasets", fontsize=12, fontweight="bold")
    OUT_MAIN.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_MAIN / "fig2_benchmark_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def make_ari_bars(rows: list[dict[str, float | str]]) -> None:
    methods = method_order(rows)
    lookup = {(str(row["method"]), str(row["dataset"])): row for row in rows}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.ravel()

    for ax, dataset in zip(axes, DATASET_ORDER):
        dataset_methods = [method for method in methods if (method, dataset) in lookup]
        dataset_methods.sort(key=lambda method: float(lookup[(method, dataset)]["ARI_mean"]))
        values = [float(lookup[(method, dataset)]["ARI_mean"]) for method in dataset_methods]
        errors = [float(lookup[(method, dataset)]["ARI_std"]) for method in dataset_methods]
        labels = [DISPLAY.get(method, method) for method in dataset_methods]
        colors = ["#c44e52" if method == "iter2_repro" else "#4c72b0" for method in dataset_methods]
        ax.barh(labels, values, xerr=errors, color=colors, alpha=0.9, ecolor="#333333", capsize=2)
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlim(0, min(1.0, max(values) + 0.08))
        ax.set_xlabel("ARI")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle("Dataset-wise ARI comparison", fontsize=12, fontweight="bold")
    OUT_MAIN.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_MAIN / "fig3_datasetwise_ari_barplots.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    make_heatmap(rows)
    make_ari_bars(rows)


if __name__ == "__main__":
    main()
