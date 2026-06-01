"""
common.py —— 所有 baseline 共用的后端:统一加载数据集、统一算指标、统一存预测。

设计契约(每个方法 runner 都遵守):
  输入: data/<dataset>/<dataset>.h5ad
  输出: results/baselines/<method>/<dataset>_seed<s>_pred.csv   (一列 pred, 一行一个细胞)
        results/baselines/<method>/<dataset>_seed<s>_metrics.json (ACC/NMI/ARI)

这样所有方法打分口径完全一致, 且 pred.csv 的格式正好能喂给 figures/ 里的
case-study 可视化脚本(它读 ['pred'])。
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
try:
    import scanpy as sc
    _read_h5ad = sc.read_h5ad
except ImportError:
    import anndata as ad
    _read_h5ad = ad.read_h5ad
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

# 常见的标签列名(你的数据集若用别的名字, 往这里加)
LABEL_KEYS = ["celltype", "cell_type", "CellType", "cell.type", "labels", "label",
              "y", "Group", "group", "cluster", "Cluster", "annotation", "louvain"]


def load_dataset(h5ad_path: str, label_key: str | None = None):
    """返回 (adata, X[dense], y_true[int codes], n_clusters)。"""
    adata = _read_h5ad(h5ad_path)
    key = label_key
    if key is None:
        for k in LABEL_KEYS:
            if k in adata.obs.columns:
                key = k
                break
    if key is None or key not in adata.obs.columns:
        raise ValueError(f"找不到标签列。obs 里有这些列: {list(adata.obs.columns)}  "
                         f"请用 --label_key 指定")
    y = pd.Categorical(adata.obs[key].astype(str)).codes.astype(int)
    X = adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    n_clusters = int(len(np.unique(y)))
    print(f"  [load] cells={X.shape[0]} genes={X.shape[1]} "
          f"n_clusters={n_clusters} label_key='{key}'")
    return adata, X.astype(np.float32), y, n_clusters


def cluster_acc(y_true, y_pred) -> float:
    """聚类准确率: 用匈牙利算法做最优标签匹配 (无监督聚类的标准 ACC)。"""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    D = max(int(y_pred.max()), int(y_true.max())) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row, col = linear_sum_assignment(w.max() - w)
    return float(w[row, col].sum()) / y_pred.size


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "ACC": round(cluster_acc(y_true, y_pred), 4),
        "NMI": round(float(normalized_mutual_info_score(y_true, y_pred)), 4),
        "ARI": round(float(adjusted_rand_score(y_true, y_pred)), 4),
        "n_pred_clusters": int(len(np.unique(y_pred))),
    }


def save_outputs(out_dir, dataset, method, seed, y_pred, y_true) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    y_pred = np.asarray(y_pred).astype(int)
    pd.DataFrame({"pred": y_pred, "true": np.asarray(y_true).astype(int)}).to_csv(
        os.path.join(out_dir, f"{dataset}_seed{seed}_pred.csv"), index=False)
    m = compute_metrics(y_true, y_pred)
    rec = {"dataset": dataset, "method": method, "seed": int(seed), **m}
    with open(os.path.join(out_dir, f"{dataset}_seed{seed}_metrics.json"), "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print(f"  [done] {method}/{dataset} seed{seed}  "
          f"ACC={m['ACC']:.3f} NMI={m['NMI']:.3f} ARI={m['ARI']:.3f}")
    return rec


def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
