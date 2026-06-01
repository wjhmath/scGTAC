"""
run_scdeepcluster.py —— scDeepCluster baseline (遵守 common.py 契约)。

环境: bl_torch   仓库: baselines/repos/scDeepCluster_pytorch (须先 git clone)
要点: scDeepCluster 强制要原始整数计数输入, 内部自做 ZINB 归一化。
      自动用 GPU(有则), 深度方法建议在 A100 上跑。

被 run.py 调用; 也可单独:
  python run_scdeepcluster.py --data .../baron/baron.h5ad --out results/baselines/scdeepcluster \
      --dataset baron --seed 1
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import scanpy as sc

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repos", "scDeepCluster_pytorch")
sys.path.insert(0, REPO)

from common import load_dataset, save_outputs, set_seed


def get_counts(adata, X):
    """取原始整数计数: 优先 layers['counts'] / raw.X, 否则 X; 非整数则四舍五入并警告。"""
    import scipy.sparse as sp
    cand, src = X, "X"
    if "counts" in getattr(adata, "layers", {}):
        cand, src = adata.layers["counts"], "layers['counts']"
    elif adata.raw is not None:
        cand, src = adata.raw.X, "raw.X"
    cand = cand.toarray() if sp.issparse(cand) else np.asarray(cand)
    cand = cand.astype("float64")
    if not np.allclose(cand, np.round(cand)):
        print(f"  [warn] {src} 不是整数计数, scDeepCluster 需要原始 counts; "
              f"已四舍五入(若数据集本身没存 counts, 结果可能偏差)")
        cand = np.round(cand)
    else:
        print(f"  [counts] 使用 {src} 作为原始计数")
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label_key", default=None)
    ap.add_argument("--pretrain_epochs", type=int, default=300)
    ap.add_argument("--maxiter", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--gamma", type=float, default=1.0)
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    import torch
    from scDeepCluster import scDeepCluster
    from preprocess import read_dataset, normalize
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [device] {device}")

    adata0, X, y, k = load_dataset(args.data, args.label_key)
    counts = get_counts(adata0, X)

    # 按仓库流程: 原始计数建 AnnData -> read_dataset(断言计数) -> normalize
    adata = sc.AnnData(counts)
    adata.obs["Group"] = y
    adata = read_dataset(adata, transpose=False, test_split=False, copy=True)
    adata = normalize(adata, size_factors=True, normalize_input=True, logtrans_input=True)
    # normalize 里 filter_cells 可能删个别细胞, y 跟着 adata.obs 对齐
    y_kept = adata.obs["Group"].to_numpy().astype(int)
    print(f"  [preprocess] cells={adata.n_obs} genes={adata.n_vars} n_clusters={k}")

    model = scDeepCluster(input_dim=adata.n_vars, z_dim=32,
                          encodeLayer=[256, 64], decodeLayer=[64, 256],
                          sigma=args.sigma, gamma=args.gamma, device=device)

    ae_file = os.path.join(args.out, f"{args.dataset}_seed{args.seed}_AE.pth.tar")
    save_dir = os.path.join(args.out, "_ckpt")
    os.makedirs(save_dir, exist_ok=True)

    model.pretrain_autoencoder(X=adata.X, X_raw=adata.raw.X,
                               size_factor=adata.obs.size_factors,
                               batch_size=args.batch_size,
                               epochs=args.pretrain_epochs, ae_weights=ae_file)

    y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X,
                                   size_factor=adata.obs.size_factors,
                                   n_clusters=k, init_centroid=None, y_pred_init=None,
                                   y=y_kept, batch_size=args.batch_size,
                                   num_epochs=args.maxiter, update_interval=1,
                                   tol=0.001, save_dir=save_dir)

    save_outputs(args.out, args.dataset, "scdeepcluster", args.seed,
                 np.asarray(y_pred).astype(int), y_kept)


if __name__ == "__main__":
    main()
