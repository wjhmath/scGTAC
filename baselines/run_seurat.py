"""
run_seurat.py —— Seurat baseline (遵守 common.py 契约)。

环境: bl_r (R 4.2 + Seurat)
标准 Seurat pipeline: Normalize -> HVG -> Scale -> PCA -> KNN -> Louvain/Leiden。
这是单细胞分析的"金标准"pipeline,审稿人必看的 baseline。
纯 CPU,不需要 GPU。

被 run.py 调用; 也可单独:
  python run_seurat.py --data .../baron/baron.h5ad --out results/baselines/seurat \
      --dataset baron --seed 1
"""
from __future__ import annotations
import argparse, os, sys, subprocess, tempfile
import numpy as np
import pandas as pd
import scipy.sparse as sps

from common import load_dataset, save_outputs, set_seed

R_SCRIPT = r"""
library(Seurat)
set.seed({seed})

cat("  [R] 读入计数矩阵...\n")
counts <- as.matrix(read.csv("{counts_csv}", row.names=1, check.names=FALSE))
counts <- t(counts)  # Seurat 要 genes x cells
cat(sprintf("  [R] %d genes x %d cells, k=%d\n", nrow(counts), ncol(counts), {k}))

obj <- CreateSeuratObject(counts=counts, min.cells=0, min.features=0)
obj <- NormalizeData(obj, verbose=FALSE)
obj <- FindVariableFeatures(obj, nfeatures={n_hvg}, verbose=FALSE)
obj <- ScaleData(obj, verbose=FALSE)
n_pcs <- min(50, ncol(obj)-1)
obj <- RunPCA(obj, npcs=n_pcs, verbose=FALSE)
obj <- FindNeighbors(obj, dims=1:n_pcs, verbose=FALSE)

# 二分搜索 resolution 匹配目标簇数
target_k <- {k}
lo <- 0.01; hi <- 5.0; best_res <- 0.8
for (i in 1:30) {{
    res <- (lo + hi) / 2
    obj <- FindClusters(obj, resolution=res, random.seed={seed}, verbose=FALSE)
    nc <- length(unique(Idents(obj)))
    best_res <- res
    if (nc == target_k) break
    if (nc < target_k) {{ lo <- res }} else {{ hi <- res }}
}}
labels <- as.integer(Idents(obj)) - 1L
write.csv(data.frame(pred=labels), "{pred_csv}", row.names=FALSE)
cat(sprintf("  [R] Seurat 完成, resolution=%.3f, clusters=%d\n", best_res, length(unique(labels))))
"""


def get_counts(adata, X):
    cand, src = X, "X"
    if "counts" in getattr(adata, "layers", {}):
        cand, src = adata.layers["counts"], "layers['counts']"
    elif adata.raw is not None:
        cand, src = adata.raw.X, "raw.X"
    cand = cand.toarray() if sps.issparse(cand) else np.asarray(cand)
    cand = cand.astype("float64")
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
    args = ap.parse_args()
    set_seed(args.seed); os.makedirs(args.out, exist_ok=True)

    rscript = os.path.join(os.path.dirname(sys.executable), "Rscript")
    if not os.path.isfile(rscript):
        sys.exit(f"找不到 Rscript: {rscript}\n请用 bl_r 环境的 Python")

    adata, X, y, k = load_dataset(args.data, args.label_key)
    counts = get_counts(adata, X)

    # Python 端先筛 HVG, 大幅缩小传给 R 的 CSV(20000→2000, 快 10 倍)
    if args.n_hvg > 0 and counts.shape[1] > args.n_hvg:
        var = counts.var(axis=0)
        top_idx = np.sort(np.argsort(var)[-args.n_hvg:])
        counts = counts[:, top_idx]
        print(f"  [hvg] 预筛 top {args.n_hvg} 基因")

    n_cells, n_genes = counts.shape
    print(f"  [seurat] cells={n_cells} genes={n_genes} k={k}")

    gene_names = [f"g{i}" for i in range(n_genes)]
    cell_names = [f"c{i}" for i in range(n_cells)]

    with tempfile.TemporaryDirectory() as tmpdir:
        counts_csv = os.path.join(tmpdir, "counts.csv")
        pred_csv = os.path.join(tmpdir, "pred.csv")
        r_path = os.path.join(tmpdir, "run_seurat.R")

        pd.DataFrame(counts, index=cell_names, columns=gene_names).to_csv(counts_csv)
        with open(r_path, "w") as f:
            f.write(R_SCRIPT.format(counts_csv=counts_csv.replace("\\", "/"),
                                    pred_csv=pred_csv.replace("\\", "/"),
                                    k=k, seed=args.seed, n_hvg=args.n_hvg))

        ret = subprocess.run([rscript, "--vanilla", r_path], capture_output=True, text=True)
        if ret.stdout: print(ret.stdout.rstrip())
        if ret.returncode != 0:
            print(f"  [R stderr] {ret.stderr}"); sys.exit(f"Seurat 失败")

        y_pred = pd.read_csv(pred_csv)["pred"].to_numpy().astype(int)

    save_outputs(args.out, args.dataset, "seurat", args.seed, y_pred, y)


if __name__ == "__main__":
    main()
