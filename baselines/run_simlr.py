"""
run_simlr.py —— SIMLR baseline (遵守 common.py 契约)。

环境: bl_r (R 4.2 + SIMLR)
SIMLR (Wang 2017 NatMethods): 多核相似性学习 + 聚类。
既不是深度学习也不是图方法, 是核方法范式的代表。
纯 CPU, 不需要 GPU; 大数据集会慢。

被 run.py 调用; 也可单独:
  python run_simlr.py --data .../baron/baron.h5ad --out results/baselines/simlr \
      --dataset baron --seed 1
"""
from __future__ import annotations
import argparse, os, sys, subprocess, tempfile
import numpy as np
import pandas as pd

from common import load_dataset, save_outputs, set_seed

R_SCRIPT = r"""
library(SIMLR)
set.seed({seed})

cat("  [R] 读入表达矩阵...\n")
expr <- as.matrix(read.csv("{expr_csv}", row.names=1, check.names=FALSE))
expr <- t(expr)  # SIMLR 要 genes x cells
cat(sprintf("  [R] %d genes x %d cells, k=%d\n", nrow(expr), ncol(expr), {k}))

cat("  [R] 运行 SIMLR (可能较慢)...\n")
result <- SIMLR(X=expr, c={k}, normalize=FALSE, cores.ratio=0)
labels <- result$y$cluster - 1L  # 0-indexed
write.csv(data.frame(pred=labels), "{pred_csv}", row.names=FALSE)
cat("  [R] SIMLR 完成\n")
"""


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

    # 纯 numpy 预处理(不依赖 scanpy): normalize -> log -> HVG
    X_raw = X.copy().astype("float64")
    lib_sizes = X_raw.sum(axis=1, keepdims=True)
    lib_sizes[lib_sizes == 0] = 1
    X_proc = X_raw / lib_sizes * 1e4
    X_proc = np.log1p(X_proc)
    if args.n_hvg > 0 and X_proc.shape[1] > args.n_hvg:
        var = X_proc.var(axis=0)
        top_idx = np.sort(np.argsort(var)[-args.n_hvg:])
        X_proc = X_proc[:, top_idx]
        print(f"  [hvg] 预筛 top {args.n_hvg} 基因")
    n_cells, n_genes = X_proc.shape
    print(f"  [simlr] cells={n_cells} genes={n_genes} k={k}")

    gene_names = [f"g{i}" for i in range(n_genes)]
    cell_names = [f"c{i}" for i in range(n_cells)]

    with tempfile.TemporaryDirectory() as tmpdir:
        expr_csv = os.path.join(tmpdir, "expr.csv")
        pred_csv = os.path.join(tmpdir, "pred.csv")
        r_path = os.path.join(tmpdir, "run_simlr.R")

        pd.DataFrame(X_proc, index=cell_names, columns=gene_names).to_csv(expr_csv)
        with open(r_path, "w") as f:
            f.write(R_SCRIPT.format(expr_csv=expr_csv.replace("\\", "/"),
                                    pred_csv=pred_csv.replace("\\", "/"),
                                    k=k, seed=args.seed))

        ret = subprocess.run([rscript, "--vanilla", r_path], capture_output=True, text=True)
        if ret.stdout: print(ret.stdout.rstrip())
        if ret.returncode != 0:
            print(f"  [R stderr] {ret.stderr}"); sys.exit(f"SIMLR 失败")

        y_pred = pd.read_csv(pred_csv)["pred"].to_numpy().astype(int)

    save_outputs(args.out, args.dataset, "simlr", args.seed, y_pred, y)


if __name__ == "__main__":
    main()
