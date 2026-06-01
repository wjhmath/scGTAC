"""
run_scvi.py —— scVI baseline (遵守 common.py 契约)。

环境: bl_torch + pip install scvi-tools
scVI (Lopez 2018 NatMethods): 变分自编码器建模 scRNA-seq, 学隐变量表示 -> 聚类。
和图方法形成"无图深度 vs 有图深度"的对照。
需要 GPU 跑得快。

被 run.py 调用; 也可单独:
  python run_scvi.py --data .../baron/baron.h5ad --out results/baselines/scvi \
      --dataset baron --seed 1
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import scanpy as sc
import scipy.sparse as sps
from sklearn.cluster import KMeans

from common import load_dataset, save_outputs, set_seed


def get_counts(adata, X):
    """scVI 需要原始计数。"""
    cand, src = X, "X"
    if "counts" in getattr(adata, "layers", {}):
        cand, src = adata.layers["counts"], "layers['counts']"
    elif adata.raw is not None:
        cand, src = adata.raw.X, "raw.X"
    cand = cand.toarray() if sps.issparse(cand) else np.asarray(cand)
    cand = cand.astype("float32")
    if not np.allclose(cand, np.round(cand)):
        print(f"  [warn] {src} 非整数, 已四舍五入"); cand = np.round(cand)
    else:
        print(f"  [counts] 使用 {src}")
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", required=True); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label_key", default=None)
    ap.add_argument("--n_hvg", type=int, default=2000)
    ap.add_argument("--n_latent", type=int, default=30)
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()
    set_seed(args.seed); os.makedirs(args.out, exist_ok=True)

    import torch
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [device] {device}")

    try:
        import scvi
    except ImportError:
        sys.exit("scvi-tools 未安装。请运行:\n  pip install scvi-tools")

    scvi.settings.seed = args.seed

    adata0, X, y, k = load_dataset(args.data, args.label_key)
    counts = get_counts(adata0, X)

    # scVI 需要原始计数作输入, 内部自己建模 library size + dropout
    adata = sc.AnnData(counts)
    adata.obs["celltype"] = y

    # HVG 筛选(在计数上做)
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
    y_kept = adata.obs["celltype"].to_numpy().astype(int)
    if args.n_hvg and adata.n_vars > args.n_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg,
                                    flavor="seurat_v3")
        adata = adata[:, adata.var.highly_variable].copy()
    print(f"  [preprocess] cells={adata.n_obs} genes={adata.n_vars} k={k}")

    # 训练 scVI
    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(adata, n_latent=args.n_latent)
    print("  训练 scVI...")
    model.train(max_epochs=args.max_epochs, batch_size=args.batch_size,
                early_stopping=True, train_size=0.9)

    # 取隐变量表示 -> KMeans 聚类
    latent = model.get_latent_representation()
    print(f"  [scVI] 嵌入维度: {latent.shape}")
    km = KMeans(n_clusters=k, n_init=20, random_state=args.seed)
    y_pred = km.fit_predict(latent)

    save_outputs(args.out, args.dataset, "scvi", args.seed, y_pred, y_kept)


if __name__ == "__main__":
    main()
