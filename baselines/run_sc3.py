"""
run_sc3.py —— SC3 baseline (遵守 common.py 契约)。

环境: bl_r  (conda 装的 R + bioconductor-sc3, 须额外 pip install scikit-learn)
SC3 是 R 包, 本 runner 用 Python 把数据写成 .mtx, 调 Rscript 跑 SC3, 再读回标签。
不需要 GPU, 但 SC3 是 O(n²), 大数据集(>10k 细胞)会很慢。

被 run.py 调用; 也可单独:
  python run_sc3.py --data .../baron/baron.h5ad --out results/baselines/sc3 \
      --dataset baron --seed 1
"""
from __future__ import annotations
import argparse, os, sys, tempfile, subprocess
import numpy as np
import scipy.sparse as sp
import scipy.io
import pandas as pd
from common import load_dataset, save_outputs, set_seed

R_SCRIPT = r"""
library(SC3)
library(SingleCellExperiment)
library(Matrix)

args  <- commandArgs(trailingOnly = TRUE)
mtx_f <- args[1]
k     <- as.integer(args[2])
out_f <- args[3]
seed  <- as.integer(args[4])

set.seed(seed)
cat("  [R] reading mtx...\n")
counts <- readMM(mtx_f)
counts <- as(counts, "dgCMatrix")
cat(sprintf("  [R] genes=%d cells=%d k=%d\n", nrow(counts), ncol(counts), k))

sce <- SingleCellExperiment(assays = list(counts = counts))
logcounts(sce) <- log2(counts + 1)
rowData(sce)$feature_symbol <- paste0("G", seq_len(nrow(sce)))

cat("  [R] running SC3...\n")
sce <- sc3(sce, ks = k, biology = FALSE, n_cores = 1, rand_seed = seed)

col_name <- paste0("sc3_", k, "_clusters")
labels   <- as.integer(colData(sce)[[col_name]])
write.csv(data.frame(pred = labels), out_f, row.names = FALSE)
cat("  [R] done, wrote", out_f, "\n")
"""


def get_counts(adata, X):
    """取原始整数计数: 优先 layers['counts'] / raw.X, 否则 X。"""
    cand, src = X, "X"
    if "counts" in getattr(adata, "layers", {}):
        cand, src = adata.layers["counts"], "layers['counts']"
    elif adata.raw is not None:
        cand, src = adata.raw.X, "raw.X"
    cand = cand.toarray() if sp.issparse(cand) else np.asarray(cand)
    cand = cand.astype("float64")
    if not np.allclose(cand, np.round(cand)):
        print(f"  [warn] {src} 不是整数计数, SC3 按计数处理; 已四舍五入")
        cand = np.round(cand)
    else:
        print(f"  [counts] 使用 {src} 作为原始计数")
    return cand


def find_rscript():
    """在当前 conda 环境里找 Rscript。"""
    env_bin = os.path.dirname(sys.executable)
    rs = os.path.join(env_bin, "Rscript")
    if os.path.isfile(rs):
        return rs
    import shutil
    rs2 = shutil.which("Rscript")
    if rs2:
        return rs2
    raise FileNotFoundError("找不到 Rscript, 确认 bl_r 环境已激活或在 PATH 上")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label_key", default=None)
    ap.add_argument("--n_hvg", type=int, default=2000)
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    rscript = find_rscript()
    adata, X, y, k = load_dataset(args.data, args.label_key)
    counts = get_counts(adata, X)

    # Python 端预筛 HVG(缩小传给 R 的矩阵)
    if args.n_hvg > 0 and counts.shape[1] > args.n_hvg:
        var = counts.var(axis=0)
        top_idx = np.sort(np.argsort(var)[-args.n_hvg:])
        counts = counts[:, top_idx]
        print(f"  [hvg] 预筛 top {args.n_hvg} 基因")

    if counts.shape[0] > 15000:
        print(f"  [warn] {counts.shape[0]} 细胞, SC3 是 O(n²), 可能很慢/OOM; "
              f"考虑跳过此数据集或降采样")

    with tempfile.TemporaryDirectory() as tmpdir:
        mtx_path = os.path.join(tmpdir, "counts.mtx")
        r_path = os.path.join(tmpdir, "sc3_run.R")
        pred_path = os.path.join(tmpdir, "pred.csv")

        counts_sparse = sp.csc_matrix(counts.T)  # genes x cells
        scipy.io.mmwrite(mtx_path, counts_sparse)

        with open(r_path, "w") as f:
            f.write(R_SCRIPT)

        print(f"  [R] calling Rscript ... (k={k})")
        result = subprocess.run(
            [rscript, r_path, mtx_path, str(k), pred_path, str(args.seed)],
            capture_output=True, text=True, timeout=7200
        )
        print(result.stdout.strip())
        if result.returncode != 0:
            print("  [R stderr]", result.stderr[-500:] if result.stderr else "")
            raise RuntimeError(f"SC3 失败 (returncode={result.returncode})")

        df = pd.read_csv(pred_path)
        y_pred = df["pred"].values.astype(int)

    if y_pred.min() == 1:
        y_pred = y_pred - 1

    save_outputs(args.out, args.dataset, "sc3", args.seed, y_pred, y)


if __name__ == "__main__":
    main()
