"""
run_sctag.py —— scTAG baseline (遵守 common.py 契约)。

环境: bl_tf (TF2 + Spektral 0.6)   仓库: baselines/repos/scTAG (须先 git clone)
要点: scTAG 用固定 KNN 图 + 拓扑自适应 GCN(TAGConv),
      和 scAGCR 同属图家族——scAGCR 让图可学, scTAG 图固定, 是最关键的对照。
      需要 GPU(TensorFlow)。

被 run.py 调用; 也可单独:
  python run_sctag.py --data .../baron/baron.h5ad --out results/baselines/sctag \
      --dataset baron --seed 1
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repos", "scTAG")
sys.path.insert(0, REPO)

from common import load_dataset, save_outputs, set_seed


def get_counts(adata, X):
    """取原始整数计数(和 scDeepCluster/SC3 runner 同逻辑)。"""
    import scipy.sparse as sp
    cand, src = X, "X"
    if "counts" in getattr(adata, "layers", {}):
        cand, src = adata.layers["counts"], "layers['counts']"
    elif adata.raw is not None:
        cand, src = adata.raw.X, "raw.X"
    cand = cand.toarray() if sp.issparse(cand) else np.asarray(cand)
    cand = cand.astype("float64")
    if not np.allclose(cand, np.round(cand)):
        print(f"  [warn] {src} 非整数, 已四舍五入")
        cand = np.round(cand)
    else:
        print(f"  [counts] 使用 {src}")
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label_key", default=None)
    ap.add_argument("--highly_genes", type=int, default=500,
                    help="HVG 数(scTAG 默认 500)")
    ap.add_argument("--pretrain_epochs", type=int, default=1000)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import tensorflow as tf
    tf.random.set_seed(args.seed)
    set_seed(args.seed)

    import scanpy as sc
    from preprocess import normalize
    from graph_function import get_adj
    from sctag import SCTAG
    from sklearn.cluster import SpectralClustering

    print(f"  [device] GPU={args.gpu}  TF={tf.__version__}")

    adata0, X, y, k = load_dataset(args.data, args.label_key)
    counts = get_counts(adata0, X)
    counts = np.ceil(counts).astype(int)   # scTAG 的 train.py 也做了 ceil+int

    # 按仓库流程: 计数 -> AnnData -> normalize (含 HVG 筛选)
    adata = sc.AnnData(counts.astype("float64"))
    adata.obs["Group"] = y
    adata = normalize(adata, copy=True, highly_genes=args.highly_genes,
                      size_factors=True, normalize_input=True, logtrans_input=True)
    y_kept = adata.obs["Group"].to_numpy().astype(int)
    count = adata.X
    print(f"  [preprocess] cells={count.shape[0]} genes={count.shape[1]} k={k}")

    # 建 KNN 图(固定图——这正是和 scAGCR 的区别)
    adj, adj_n = get_adj(count, k=15, pca=50)

    # 建模 + 预训练
    os.makedirs(args.out, exist_ok=True)
    model = SCTAG(count, adj=adj, adj_n=adj_n)
    model.pre_train(epochs=args.pretrain_epochs)

    # 初始聚类(谱聚类 on 邻接矩阵)-> 质心
    Y = model.embedding(count, adj_n)
    init_labels = SpectralClustering(
        n_clusters=k, affinity="precomputed",
        assign_labels="discretize", random_state=args.seed
    ).fit_predict(adj)
    centers = np.array([Y[init_labels == i].mean(0) for i in range(k)])

    # 交替训练(y 传入但代码里没用到,纯无监督)
    model.alt_train(y_kept, epochs=args.maxiter, centers=centers)
    y_pred = model.y_pred

    save_outputs(args.out, args.dataset, "sctag", args.seed,
                 np.asarray(y_pred).astype(int), y_kept)


if __name__ == "__main__":
    main()
