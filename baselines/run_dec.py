"""
run_dec.py —— DEC (Deep Embedded Clustering) baseline (遵守 common.py 契约)。

环境: bl_torch   不需要 clone 仓库(自包含实现)。
DEC 是深度聚类的祖宗(Xie 2016 ICML): 自编码器预训练 -> KL 软聚类。
它没有 ZINB/图, 和 scDeepCluster/scDSC/scTAG 形成纵向对照。
需要 GPU 跑得快, CPU 也能跑。

被 run.py 调用; 也可单独:
  python run_dec.py --data .../baron/baron.h5ad --out results/baselines/dec \
      --dataset baron --seed 1
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import scanpy as sc

from common import load_dataset, save_outputs, set_seed

# ========== DEC 模型(自包含, ~120 行) ==========
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans


class StackedAE(nn.Module):
    """对称自编码器: dims=[input, 500, 500, 2000, z_dim]"""
    def __init__(self, dims):
        super().__init__()
        enc, dec = [], []
        for i in range(len(dims) - 1):
            enc.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                enc.append(nn.ReLU())
        for i in range(len(dims) - 1, 0, -1):
            dec.append(nn.Linear(dims[i], dims[i - 1]))
            if i > 1:
                dec.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class DEC(nn.Module):
    def __init__(self, ae, n_clusters, z_dim, alpha=1.0):
        super().__init__()
        self.ae = ae
        self.alpha = alpha
        self.mu = nn.Parameter(torch.zeros(n_clusters, z_dim))

    def encode(self, x):
        _, z = self.ae(x)
        return z

    def soft_assign(self, z):
        q = 1.0 / (1.0 + (z.unsqueeze(1) - self.mu).pow(2).sum(2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        return q / q.sum(dim=1, keepdim=True)

    @staticmethod
    def target_distribution(q):
        p = q ** 2 / q.sum(dim=0, keepdim=True)
        return p / p.sum(dim=1, keepdim=True)

    def forward(self, x):
        x_rec, z = self.ae(x)
        q = self.soft_assign(z)
        return x_rec, q, z


def pretrain_ae(ae, X_tensor, device, epochs=200, batch_size=256, lr=1e-3):
    ae.train()
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True)
    opt = optim.Adam(ae.parameters(), lr=lr)
    for ep in range(epochs):
        total = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            x_rec, _ = ae(xb)
            loss = nn.MSELoss()(x_rec, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * xb.size(0)
        if (ep + 1) % 50 == 0:
            print(f"    pretrain epoch {ep+1}/{epochs}  loss={total/len(X_tensor):.4f}")


def train_dec(model, X_tensor, n_clusters, device, epochs=200,
              batch_size=256, lr=1e-3, tol=1e-3, update_interval=1):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    # init cluster centers with k-means
    with torch.no_grad():
        z_all = model.encode(X_tensor.to(device)).cpu().numpy()
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    km.fit(z_all)
    model.mu.data.copy_(torch.tensor(km.cluster_centers_, dtype=torch.float32))
    y_prev = km.labels_

    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
    for ep in range(epochs):
        # target distribution
        if ep % update_interval == 0:
            with torch.no_grad():
                _, q_all, _ = model(X_tensor.to(device))
            p_all = DEC.target_distribution(q_all).detach()
            y_cur = q_all.argmax(1).cpu().numpy()
            delta = np.sum(y_cur != y_prev) / len(y_prev)
            y_prev = y_cur
            if ep > 0 and delta < tol:
                print(f"    epoch {ep}: delta={delta:.5f} < tol, 停止")
                break

        idx = 0
        for (xb,) in loader:
            bs = xb.size(0)
            xb = xb.to(device)
            pb = p_all[idx:idx + bs]
            x_rec, q, _ = model(xb)
            kl = nn.KLDivLoss(reduction="batchmean")(q.log(), pb)
            rec = nn.MSELoss()(x_rec, xb)
            loss = kl + rec * 0.1
            opt.zero_grad(); loss.backward(); opt.step()
            idx += bs

    with torch.no_grad():
        _, q_final, _ = model(X_tensor.to(device))
    return q_final.argmax(1).cpu().numpy()


# ========== Runner ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label_key", default=None)
    ap.add_argument("--n_hvg", type=int, default=2000)
    ap.add_argument("--z_dim", type=int, default=32)
    ap.add_argument("--pretrain_epochs", type=int, default=200)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [device] {device}")
    os.makedirs(args.out, exist_ok=True)

    adata, X, y, k = load_dataset(args.data, args.label_key)
    # 标准预处理: normalize -> log -> HVG -> scale
    adata.X = X
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if args.n_hvg and adata.n_vars > args.n_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg)
        adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    X_proc = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    X_proc = np.asarray(X_proc, dtype=np.float32)
    print(f"  [preprocess] cells={X_proc.shape[0]} genes={X_proc.shape[1]} k={k}")

    X_tensor = torch.tensor(X_proc)
    input_dim = X_proc.shape[1]
    dims = [input_dim, 500, 500, 2000, args.z_dim]

    ae = StackedAE(dims).to(device)
    print("  预训练自编码器...")
    pretrain_ae(ae, X_tensor, device, epochs=args.pretrain_epochs,
                batch_size=args.batch_size)

    model = DEC(ae, n_clusters=k, z_dim=args.z_dim).to(device)
    print("  DEC 聚类训练...")
    y_pred = train_dec(model, X_tensor, n_clusters=k, device=device,
                       epochs=args.maxiter, batch_size=args.batch_size)

    save_outputs(args.out, args.dataset, "dec", args.seed, y_pred, y)


if __name__ == "__main__":
    main()
