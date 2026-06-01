"""
run_leiden.py —— Seurat/Leiden 基线(标准 scanpy pipeline)。
这个用你现有的 scagcr_env 就能直接跑, 不用建新环境, 是最先能出结果的一个。

用法:
  python run_leiden.py --data /path/baron/baron.h5ad --out results/baselines/leiden \
      --dataset baron --seed 1
"""
from __future__ import annotations
import argparse
import numpy as np
import scanpy as sc
from common import load_dataset, save_outputs, set_seed


def leiden_target_k(adata, k, seed, lo=0.05, hi=4.0, n_iter=25):
    """二分搜索分辨率, 让 Leiden 输出的簇数尽量等于 k(便于和已知 k 的方法公平比 ARI)。"""
    best = None
    for _ in range(n_iter):
        res = (lo + hi) / 2.0
        sc.tl.leiden(adata, resolution=res, random_state=seed, key_added="_lei")
        nc = adata.obs["_lei"].nunique()
        best = adata.obs["_lei"].astype(int).to_numpy()
        if nc == k:
            return best
        if nc < k:
            lo = res
        else:
            hi = res
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label_key", default=None)
    ap.add_argument("--n_hvg", type=int, default=2000)
    ap.add_argument("--n_pcs", type=int, default=50)
    ap.add_argument("--match_k", action="store_true",
                    help="二分分辨率匹配真实簇数(默认 res=1.0)")
    ap.add_argument("--no_preprocess", action="store_true",
                    help="数据已标准化时跳过 normalize/log1p")
    args = ap.parse_args()
    set_seed(args.seed)

    adata, X, y, k = load_dataset(args.data, args.label_key)
    adata.X = X

    if not args.no_preprocess:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    if args.n_hvg and adata.n_vars > args.n_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg)
        adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=min(args.n_pcs, adata.n_vars - 1), random_state=args.seed)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(args.n_pcs, adata.n_vars - 1),
                    random_state=args.seed)

    if args.match_k:
        y_pred = leiden_target_k(adata, k, args.seed)
    else:
        sc.tl.leiden(adata, resolution=1.0, random_state=args.seed, key_added="_lei")
        y_pred = adata.obs["_lei"].astype(int).to_numpy()

    save_outputs(args.out, args.dataset, "leiden", args.seed, y_pred, y)


if __name__ == "__main__":
    main()
