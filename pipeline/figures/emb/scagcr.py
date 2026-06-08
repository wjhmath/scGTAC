import os, sys, numpy as np, torch, scanpy as sc
from sklearn.cluster import KMeans
PROJ="/home/liyang/BioJiaheWang/scGTAC"
sys.path.insert(0, os.path.join(PROJ,"scgtac"))
from model import Model
from config import config
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA=f"{PROJ}/data/muraro_pancreas/muraro_pancreas.h5ad"
CKPT=f"{PROJ}/results/scagcr_final/muraro_pancreas/muraro_pancreas_seed1.pt"
OUT=f"{PROJ}/paper_figures/Figure3_umap"; os.makedirs(OUT,exist_ok=True)
SEED=1; NCL=10

data=sc.read_h5ad(DATA); data.obs_names_make_unique(); data.var_names_make_unique()
for col in ["cell_type","cell_type_label","CellType","celltype","label","labels","Group"]:
    if col in data.obs.columns: y=data.obs[col].astype(str).to_numpy(); break
else: y=data.obs.iloc[:,0].astype(str).to_numpy()
if data.n_obs>20000:
    np.random.seed(1); idx=np.random.choice(data.n_obs,20000,replace=False); data=data[idx].copy(); y=y[idx]
xmax=float(data.X.max()) if hasattr(data.X,"max") else float(np.max(data.X))
if xmax>30: sc.pp.normalize_total(data,target_sum=1e4); sc.pp.log1p(data)
if data.n_vars>2000:
    sc.pp.highly_variable_genes(data,n_top_genes=2000,flavor="seurat"); data=data[:,data.var.highly_variable].copy()
X=data.X; X=X.toarray() if hasattr(X,"toarray") else np.asarray(X); X=X.astype(np.float32)
N,input_dim=X.shape; print("cells",N,"genes",input_dim,"types",len(np.unique(y)))

model=Model(input_dim, config['graph_head'], config['phi'], config['gcn_dim'], config['mlp_dim'],
            config['prob_feature'], config['prob_edge'], config['tau'], config['alpha'], config['beta'],
            config['dropout'], NCL, config['cluster_alpha']).to(device)
ck=torch.load(CKPT, map_location=device, weights_only=False)
model.load_state_dict(ck['net']); model.eval(); print("best_ari in ckpt:", ck.get('best_ari'))

perm=np.random.RandomState(SEED).permutation(N)
Z=np.zeros((N, config['gcn_dim']), dtype=np.float32)
with torch.no_grad():
    for i in range(0,N,128):
        bidx=perm[i:i+128]
        bx=torch.tensor(X[bidx]).float().to(device)
        Z[bidx]=model(bx)[0].cpu().numpy()
pred=KMeans(n_clusters=NCL, random_state=SEED, n_init=20).fit_predict(Z)
np.savez(f"{OUT}/emb_muraro_scGTAC.npz", emb=Z, y=y, pred=pred)
print("saved scGTAC", Z.shape)
