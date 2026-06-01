from __future__ import annotations

from math import ceil
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import torch
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
SC_DIR = ROOT / 'scagcr'
import sys
sys.path.insert(0, str(SC_DIR))

from model import Model  # type: ignore
from config import config  # type: ignore

DATA_PATH = ROOT / 'data' / 'baron' / 'baron.h5ad'
CKPT = ROOT / 'results' / 'iter2_repro' / 'checkpoints' / 'baron_seed1.pt'
OUT = ROOT / 'generated_figures' / 'fig_case_baron_clusters_3col.pdf'
OUT_2COL = ROOT / 'generated_figures' / 'fig_case_baron_clusters_2col.pdf'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

plt.style.use(['science', 'nature', 'no-latex'])
plt.rcParams.update({'figure.dpi':200,'savefig.dpi':300,'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42})


def make_palette(n: int):
    base = plt.get_cmap('tab20')(np.linspace(0, 1, max(n, 20)))
    return ListedColormap(base[:n])


def align_cluster_colors(pred_labels, true_labels):
    true_cat = pd.Categorical(true_labels)
    true_codes = true_cat.codes
    mapping = {}
    for cluster in pd.unique(pred_labels):
        mask = pred_labels == cluster
        vals, cnts = np.unique(true_codes[mask], return_counts=True)
        mapping[cluster] = vals[np.argmax(cnts)]
    return np.array([mapping[c] for c in pred_labels]), len(true_cat.categories)


def build_model(input_dim: int, n_clusters: int):
    model = Model(
        input_dim, config['graph_head'], config['phi'], config['gcn_dim'], config['mlp_dim'],
        config['prob_feature'], config['prob_edge'], config['tau'], config['alpha'], config['beta'],
        config['dropout'], n_clusters, config['cluster_alpha']
    ).to(DEVICE)
    model.eval()
    return model


def infer_embeddings(model: Model, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.tensor(X[start:start+batch_size], dtype=torch.float32, device=DEVICE)
            z, _, _, _, _, _, _, _ = model(batch)
            out.append(z.cpu().numpy())
    return np.vstack(out)


def load_baron():
    adata = ad.read_h5ad(DATA_PATH)
    adata.obs_names_make_unique(); adata.var_names_make_unique()
    labels = adata.obs['cell_type'].astype(str).to_numpy()
    x_max = float(adata.X.max()) if hasattr(adata.X, 'max') else float(np.max(adata.X))
    if x_max > 30:
        import scanpy as sc
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    if adata.n_vars > 2000:
        import scanpy as sc
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')
        adata = adata[:, adata.var.highly_variable].copy()
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)
    return X.astype(np.float32), labels


def compute_scagcr_labels(X, labels):
    model = build_model(X.shape[1], len(np.unique(labels)))
    ckpt = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt['net'])
    emb = infer_embeddings(model, X)
    pred = KMeans(n_clusters=len(np.unique(labels)), random_state=1, n_init=20).fit_predict(emb)
    import umap
    layout = umap.UMAP(n_neighbors=20, min_dist=0.25, metric='euclidean', random_state=1).fit_transform(emb)
    return layout, pred


def compute_leiden_labels(X):
    import scanpy as sc
    import anndata as ad2
    adata = ad2.AnnData(X=X)
    sc.pp.neighbors(adata, use_rep='X', n_neighbors=15)
    sc.tl.leiden(adata, resolution=1.0, random_state=1)
    return adata.obs['leiden'].astype(int).to_numpy()


def draw_panel_figure(layout, labels, data, ncols, nrows, out_path):
    methods = list(data.keys())
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0*ncols, 3.1*nrows))
    axes = np.atleast_1d(axes).ravel()
    panel_tags = [f"({chr(ord('a') + i)})" for i in range(len(methods))]
    true_codes = pd.Categorical(labels).codes

    for ax in axes[len(methods):]:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.8)
            s.set_edgecolor('#888888')

    for idx, (ax, method) in enumerate(zip(axes, methods)):
        pred = data[method]
        if method == 'Ground truth':
            codes = true_codes
            n = len(np.unique(true_codes))
        else:
            codes, n = align_cluster_colors(np.asarray(pred), labels)
        ax.scatter(layout[:,0], layout[:,1], c=codes, s=6, cmap=make_palette(n), linewidths=0, alpha=0.88)
        ax.set_title(method)
        ax.text(0.02, 0.98, panel_tags[idx], transform=ax.transAxes, ha='left', va='top', fontsize=8, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.8)
            s.set_edgecolor('#888888')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def main():
    X, labels = load_baron()
    layout, scagcr = compute_scagcr_labels(X, labels)
    true_codes = pd.Categorical(labels).codes
    leiden = compute_leiden_labels(X)

    data = {
        'Ground truth': true_codes,
        'scAGCR': scagcr,
        'Leiden': leiden,
        'scDSC': pd.read_csv(ROOT/'results/baselines/baron_scdsc/types_120_pred.csv')['pred'].to_numpy(),
        'scDCC': pd.read_csv(ROOT/'results/baselines_batch1/baron_scdcc_rep1/types_120_pred.csv')['pred'].to_numpy(),
        'scMAE': pd.read_csv(ROOT/'results/baselines_5epoch/baron_scmae_rep3/types_5_pred.csv')['pred'].to_numpy(),
        'scGNN': pd.read_csv(ROOT/'results/tmp_scgnn_baron_smoke/types_1_pred.csv')['pred'].to_numpy(),
    }

    payload = torch.load(ROOT/'embeddings/baron_scgpt.pt', map_location='cpu', weights_only=False)
    emb = payload['embeddings'].cpu().numpy() if hasattr(payload['embeddings'], 'cpu') else np.asarray(payload['embeddings'])
    data['scGPT-KMeans'] = KMeans(n_clusters=len(np.unique(labels)), random_state=1, n_init=20).fit_predict(emb)
    try:
        import scanpy as sc
        import anndata as ad2
        adata = ad2.AnnData(X=emb)
        sc.pp.neighbors(adata, use_rep='X', n_neighbors=15)
        sc.tl.leiden(adata, resolution=1.0, random_state=1)
        data['scGPT-Leiden'] = adata.obs['leiden'].astype(int).to_numpy()
    except Exception:
        data['scGPT-Leiden'] = data['scGPT-KMeans']

    methods = list(data.keys())
    draw_panel_figure(layout, labels, data, ncols=3, nrows=3, out_path=OUT)
    draw_panel_figure(layout, labels, data, ncols=2, nrows=5, out_path=OUT_2COL)
    print(OUT)
    print(OUT_2COL)


if __name__ == '__main__':
    main()
