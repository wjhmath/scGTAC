"""
run_scgnn.py —— scGNN baseline (遵守 common.py 契约)。

环境: bl_torch   不需要 clone 仓库(简化自包含实现)。
scGNN 的核心: 自编码器学嵌入 -> KNN 建图 -> 图自编码器精炼 -> 聚类 -> 迭代。
图是每轮从嵌入用 KNN 重建的(启发式), 不是端到端学出来的——和 scAGCR 的可学图形成对照。
需要 GPU 跑得快。

被 run.py 调用; 也可单独:
  python run_scgnn.py --data .../baron/baron.h5ad --out results/baselines/scgnn \
      --dataset baron --seed 1
"""
from __future__ import annotations
import argparse, os
import numpy as np
import scanpy as sc
import scipy.sparse as sps
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from common import load_dataset, save_outputs, set_seed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# ========== 模型组件 ==========

class FeatureAE(nn.Module):
    """特征自编码器: 学习细胞嵌入"""
    def __init__(self, input_dim, hidden=128, z_dim=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, z_dim))
        self.dec = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, input_dim))
    def forward(self, x):
        z = self.enc(x); return self.dec(z), z

class GCNLayer(nn.Module):
    def __init__(self, inf, outf):
        super().__init__(); self.W = nn.Linear(inf, outf)
    def forward(self, x, adj):
        return torch.spmm(adj, self.W(x))

class GraphAE(nn.Module):
    """图自编码器: 用图结构精炼嵌入"""
    def __init__(self, input_dim, hidden=64, z_dim=32):
        super().__init__()
        self.gc1 = GCNLayer(input_dim, hidden)
        self.gc2 = GCNLayer(hidden, z_dim)
    def encode(self, x, adj):
        h = F.relu(self.gc1(x, adj)); return self.gc2(h, adj)
    def decode(self, z):
        return torch.sigmoid(torch.mm(z, z.t()))


def build_sparse_adj(X_np, k=15):
    """从嵌入建 KNN 图, 返回归一化稀疏邻接(torch sparse)"""
    A = kneighbors_graph(X_np, k, mode="connectivity", include_self=True)
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)
    A = A + sps.eye(A.shape[0])
    # D^{-0.5} A D^{-0.5}
    D = np.array(A.sum(1)).flatten()
    D_inv_sqrt = np.power(D, -0.5); D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
    D_mat = sps.diags(D_inv_sqrt)
    A_norm = D_mat.dot(A).dot(D_mat)
    A_coo = sps.coo_matrix(A_norm, dtype=np.float32)
    idx = torch.LongTensor(np.vstack([A_coo.row, A_coo.col]))
    val = torch.FloatTensor(A_coo.data)
    return torch.sparse_coo_tensor(idx, val, A_coo.shape)


# ========== Runner ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", required=True); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label_key", default=None)
    ap.add_argument("--n_hvg", type=int, default=2000)
    ap.add_argument("--z_dim", type=int, default=32)
    ap.add_argument("--ae_epochs", type=int, default=300)
    ap.add_argument("--gae_epochs", type=int, default=200)
    ap.add_argument("--em_iters", type=int, default=5, help="EM 迭代轮数")
    ap.add_argument("--knn_k", type=int, default=15)
    args = ap.parse_args()
    set_seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [device] {device}"); os.makedirs(args.out, exist_ok=True)

    adata, X, y, k = load_dataset(args.data, args.label_key)
    adata.X = X
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    if args.n_hvg and adata.n_vars > args.n_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg)
        adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    X_proc = np.asarray(adata.X.toarray() if sps.issparse(adata.X) else adata.X, dtype=np.float32)
    n_cells, n_genes = X_proc.shape
    print(f"  [preprocess] cells={n_cells} genes={n_genes} k={k}")

    X_t = torch.tensor(X_proc).to(device)
    z_dim = args.z_dim

    # ========== EM 迭代 ==========
    y_pred = None
    for em in range(args.em_iters):
        print(f"  [EM iter {em+1}/{args.em_iters}]")

        # Step 1: 训练特征自编码器
        ae = FeatureAE(n_genes, hidden=128, z_dim=z_dim).to(device)
        opt_ae = Adam(ae.parameters(), lr=1e-3)
        for ep in range(args.ae_epochs):
            x_rec, z = ae(X_t); loss = F.mse_loss(x_rec, X_t)
            opt_ae.zero_grad(); loss.backward(); opt_ae.step()
        with torch.no_grad(): _, z_ae = ae(X_t)
        z_np = z_ae.cpu().numpy()
        print(f"    AE 完成, recon_loss={loss.item():.4f}")

        # Step 2: 从 AE 嵌入建 KNN 图
        adj = build_sparse_adj(z_np, k=args.knn_k).to(device)

        # Step 3: 训练图自编码器
        gae = GraphAE(z_dim, hidden=64, z_dim=z_dim).to(device)
        z_input = z_ae.detach()
        opt_gae = Adam(gae.parameters(), lr=1e-3)
        # 邻接矩阵做标签(稀疏 -> dense 会爆内存, 用采样近似)
        adj_dense = adj.to_dense() if n_cells < 5000 else None
        for ep in range(args.gae_epochs):
            z_gae = gae.encode(z_input, adj)
            if adj_dense is not None:
                A_rec = gae.decode(z_gae)
                loss = F.binary_cross_entropy(A_rec, adj_dense)
            else:
                # 大数据集: 用重建损失近似(避免 NxN dense)
                loss = F.mse_loss(z_gae, z_input) * 0.1
            opt_gae.zero_grad(); loss.backward(); opt_gae.step()

        with torch.no_grad(): z_final = gae.encode(z_input, adj).cpu().numpy()
        print(f"    GAE 完成")

        # Step 4: 聚类
        km = KMeans(n_clusters=k, n_init=20, random_state=args.seed)
        y_pred = km.fit_predict(z_final)

        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(y, y_pred)
        print(f"    聚类 ARI={ari:.4f}")

    save_outputs(args.out, args.dataset, "scgnn", args.seed,
                 np.asarray(y_pred), y)


if __name__ == "__main__":
    main()
