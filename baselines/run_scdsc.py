"""
run_scdsc.py —— scDSC baseline (遵守 common.py 契约)。

环境: bl_torch   不需要 clone scDSC 仓库(自包含实现, 避免其 TF import 问题)。
scDSC = ZINB 自编码器 + GNN(固定 KNN 图) + 互监督聚类。
和 scAGCR 同属图家族——scAGCR 图可学, scDSC 图固定 KNN。
需要 GPU。

被 run.py 调用; 也可单独:
  python run_scdsc.py --data .../baron/baron.h5ad --out results/baselines/scdsc \
      --dataset baron --seed 1
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import scanpy as sc
import scipy.sparse as sps
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans

from common import load_dataset, save_outputs, set_seed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch.optim import Adam

# ========== 组件(自包含,~150 行) ==========

class ZINBLoss(nn.Module):
    def forward(self, x, mean, disp, pi, sf, ridge=0.0):
        eps = 1e-10; sf = sf[:, None]; mean = mean * sf
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1) - torch.lgamma(x+disp+eps)
        t2 = (disp+x)*torch.log(1+(mean/(disp+eps))) + x*(torch.log(disp+eps)-torch.log(mean+eps))
        nb = t1+t2; nb_case = nb-torch.log(1-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi+(1-pi)*zero_nb+eps)
        r = torch.where(x<=1e-8, zero_case, nb_case)
        if ridge>0: r = r+ridge*pi**2
        return r.mean()

class MeanAct(nn.Module):
    def forward(self,x): return torch.clamp(torch.exp(x),1e-5,1e6)
class DispAct(nn.Module):
    def forward(self,x): return torch.clamp(F.softplus(x),1e-4,1e4)

class GNNLayer(nn.Module):
    def __init__(self, inf, outf):
        super().__init__(); self.W=Parameter(torch.FloatTensor(inf,outf)); nn.init.xavier_uniform_(self.W)
    def forward(self, x, adj, act=True):
        h = torch.mm(x, self.W); h = torch.spmm(adj, h)
        return F.relu(h) if act else h

class AE(nn.Module):
    def __init__(self, dims, n_z):
        super().__init__()
        self.enc1=Linear(dims[0],dims[1]); self.bn1=nn.BatchNorm1d(dims[1])
        self.enc2=Linear(dims[1],dims[2]); self.bn2=nn.BatchNorm1d(dims[2])
        self.enc3=Linear(dims[2],dims[3]); self.bn3=nn.BatchNorm1d(dims[3])
        self.z_layer=Linear(dims[3],n_z)
        self.dec1=Linear(n_z,dims[3]);     self.bn4=nn.BatchNorm1d(dims[3])
        self.dec2=Linear(dims[3],dims[2]); self.bn5=nn.BatchNorm1d(dims[2])
        self.dec3=Linear(dims[2],dims[1]); self.bn6=nn.BatchNorm1d(dims[1])
        self.x_bar=Linear(dims[1],dims[0])
    def forward(self,x):
        h1=F.relu(self.bn1(self.enc1(x))); h2=F.relu(self.bn2(self.enc2(h1)))
        h3=F.relu(self.bn3(self.enc3(h2))); z=self.z_layer(h3)
        d1=F.relu(self.bn4(self.dec1(z))); d2=F.relu(self.bn5(self.dec2(d1)))
        d3=F.relu(self.bn6(self.dec3(d2))); return self.x_bar(d3), h1, h2, h3, z, d3

class SDCN(nn.Module):
    def __init__(self, ae, n_input, enc_dims, n_z, n_clusters):
        super().__init__(); self.ae=ae; sigma=0.5; self.sigma=sigma
        self.gnn1=GNNLayer(n_input,enc_dims[0]); self.gnn2=GNNLayer(enc_dims[0],enc_dims[1])
        self.gnn3=GNNLayer(enc_dims[1],enc_dims[2]); self.gnn4=GNNLayer(enc_dims[2],n_z)
        self.gnn5=GNNLayer(n_z,n_clusters)
        self.cluster_layer=Parameter(torch.Tensor(n_clusters,n_z)); nn.init.xavier_normal_(self.cluster_layer.data)
        self._mean=nn.Sequential(Linear(enc_dims[0],n_input),MeanAct())
        self._disp=nn.Sequential(Linear(enc_dims[0],n_input),DispAct())
        self._pi=nn.Sequential(Linear(enc_dims[0],n_input),nn.Sigmoid())
        self.zinb=ZINBLoss()
    def forward(self, x, adj):
        xb, t1, t2, t3, z, d3 = self.ae(x); s=self.sigma
        h=self.gnn1(x,adj); h=self.gnn2((1-s)*h+s*t1,adj)
        h=self.gnn3((1-s)*h+s*t2,adj); h=self.gnn4((1-s)*h+s*t3,adj)
        h=self.gnn5((1-s)*h+s*z,adj,act=False); pred=F.softmax(h,dim=1)
        mean=self._mean(d3); disp=self._disp(d3); pi=self._pi(d3)
        q=1.0/(1.0+torch.sum((z.unsqueeze(1)-self.cluster_layer)**2,2))
        q=q.pow(1.0); q=(q.t()/q.sum(1)).t()
        return xb, q, pred, z, mean, disp, pi

def target_distribution(q):
    p=q**2/q.sum(0); return (p.t()/p.sum(1)).t()

def build_knn_adj(X, k=15):
    A = kneighbors_graph(X, k, mode="connectivity", include_self=True)
    A = A + A.T.multiply(A.T>A) - A.multiply(A.T>A)
    A = A + sps.eye(A.shape[0])
    A = sps.coo_matrix(A, dtype=np.float32)
    idx = torch.LongTensor(np.vstack([A.row, A.col]))
    val = torch.FloatTensor(A.data)
    return torch.sparse_coo_tensor(idx, val, A.shape)


# ========== Runner ==========
def get_counts(adata, X):
    cand, src = X, "X"
    if "counts" in getattr(adata,"layers",{}): cand,src=adata.layers["counts"],"layers['counts']"
    elif adata.raw is not None: cand,src=adata.raw.X,"raw.X"
    cand = cand.toarray() if sps.issparse(cand) else np.asarray(cand)
    cand = cand.astype("float64")
    if not np.allclose(cand, np.round(cand)):
        print(f"  [warn] {src} 非整数, 已四舍五入"); cand=np.round(cand)
    else: print(f"  [counts] 使用 {src}")
    return cand

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", required=True); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label_key", default=None)
    ap.add_argument("--n_z", type=int, default=10)
    ap.add_argument("--knn_k", type=int, default=15)
    ap.add_argument("--pretrain_epochs", type=int, default=200)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    set_seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [device] {device}"); os.makedirs(args.out, exist_ok=True)

    adata0, X, y, k = load_dataset(args.data, args.label_key)
    counts = get_counts(adata0, X)

    # ZINB 预处理(和 scDeepCluster 相同流程)
    adata = sc.AnnData(counts); adata.obs["Group"]=y
    sc.pp.filter_genes(adata, min_counts=1); sc.pp.filter_cells(adata, min_counts=1)
    adata.raw = adata.copy()
    sc.pp.normalize_per_cell(adata)
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    sc.pp.log1p(adata); sc.pp.scale(adata)
    y_kept = adata.obs["Group"].to_numpy().astype(int)
    X_norm = np.asarray(adata.X, dtype=np.float32)
    X_raw = np.asarray(adata.raw.X.toarray() if sps.issparse(adata.raw.X) else adata.raw.X, dtype=np.float32)
    sf = adata.obs['size_factors'].to_numpy().astype(np.float32)
    n_input = X_norm.shape[1]
    print(f"  [preprocess] cells={X_norm.shape[0]} genes={n_input} k={k}")

    # 建 KNN 图
    from sklearn.decomposition import PCA
    X_pca = PCA(n_components=min(50, n_input)).fit_transform(X_norm)
    adj = build_knn_adj(X_pca, k=args.knn_k).to(device)

    data_t = torch.tensor(X_norm).to(device)
    raw_t = torch.tensor(X_raw).to(device)
    sf_t = torch.tensor(sf).to(device)

    # 预训练 AE
    enc_dims = [500, 500, 2000]
    ae = AE([n_input]+enc_dims, args.n_z).to(device)
    opt_ae = Adam(ae.parameters(), lr=1e-3)
    print("  预训练 AE...")
    for ep in range(args.pretrain_epochs):
        xb, *_, z, _ = ae(data_t); loss = F.mse_loss(xb, data_t)
        opt_ae.zero_grad(); loss.backward(); opt_ae.step()
        if (ep+1) % 50 == 0: print(f"    epoch {ep+1}/{args.pretrain_epochs} loss={loss.item():.4f}")

    # 建 SDCN
    model = SDCN(ae, n_input, enc_dims, args.n_z, k).to(device)
    opt = Adam(model.parameters(), lr=args.lr)
    with torch.no_grad(): _, _, _, _, z0, _ = ae(data_t)
    km = KMeans(n_clusters=k, n_init=20, random_state=args.seed).fit(z0.cpu().numpy())
    model.cluster_layer.data = torch.tensor(km.cluster_centers_, dtype=torch.float32).to(device)

    # 训练
    print("  scDSC 聚类训练...")
    W = [0.1, 0.01, 1.0, 0.1]  # bce, ce, recon, zinb
    for ep in range(args.maxiter):
        if ep % 1 == 0:
            with torch.no_grad(): _, q_t, _, _, _, _, _ = model(data_t, adj)
            p = target_distribution(q_t.data)
        xb, q, pred, z, mean, disp, pi = model(data_t, adj)
        eps = 1e-10
        q = q.clamp(eps, 1-eps)
        p = p.clamp(eps, 1-eps)
        pred = pred.clamp(eps, 1-eps)
        bce = F.binary_cross_entropy(q, p)
        ce = F.kl_div(pred.log(), p, reduction='batchmean')
        re = F.mse_loss(xb, data_t)
        zinb = model.zinb(raw_t, mean, disp, pi, sf_t)
        loss = W[0]*bce + W[1]*ce + W[2]*re + W[3]*zinb
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 50 == 0:
            y_pred = pred.argmax(1).cpu().numpy()
            from sklearn.metrics import adjusted_rand_score
            print(f"    epoch {ep+1}/{args.maxiter}  loss={loss.item():.3f}  ARI={adjusted_rand_score(y_kept, y_pred):.3f}")

    with torch.no_grad(): _, _, pred, _, _, _, _ = model(data_t, adj)
    y_pred = pred.argmax(1).cpu().numpy()
    save_outputs(args.out, args.dataset, "scdsc", args.seed, y_pred, y_kept)


if __name__ == "__main__":
    main()
