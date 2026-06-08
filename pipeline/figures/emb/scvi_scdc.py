import os, sys, numpy as np, scanpy as sc, scipy.sparse as sps, torch
from sklearn.cluster import KMeans
PROJ="/home/liyang/BioJiaheWang/scGTAC"
sys.path.insert(0, f"{PROJ}/baselines")
from common import load_dataset, set_seed
DATA=f"{PROJ}/data/muraro_pancreas/muraro_pancreas.h5ad"
OUT=f"{PROJ}/paper_figures/Figure3_umap"; os.makedirs(OUT,exist_ok=True)
SEED=1
def get_counts(adata,X):
    cand,src=X,"X"
    if "counts" in getattr(adata,"layers",{}): cand=adata.layers["counts"]
    elif adata.raw is not None: cand=adata.raw.X
    cand=cand.toarray() if sps.issparse(cand) else np.asarray(cand)
    cand=cand.astype("float32")
    return np.round(cand) if not np.allclose(cand,np.round(cand)) else cand

adata0,X,y,k=load_dataset(DATA)
ylab=adata0.obs["cell_type"].astype(str).to_numpy()
counts=get_counts(adata0,X)

# ---- scVI ----
try:
    set_seed(SEED); import scvi; scvi.settings.seed=SEED
    ad=sc.AnnData(counts); ad.obs["ct"]=ylab
    sc.pp.filter_genes(ad,min_counts=1); sc.pp.filter_cells(ad,min_counts=1)
    yk=ad.obs["ct"].to_numpy()
    if ad.n_vars>2000:
        sc.pp.highly_variable_genes(ad,n_top_genes=2000,flavor="seurat_v3"); ad=ad[:,ad.var.highly_variable].copy()
    scvi.model.SCVI.setup_anndata(ad); m=scvi.model.SCVI(ad,n_latent=30)
    m.train(max_epochs=200,batch_size=256,early_stopping=True,train_size=0.9)
    lat=m.get_latent_representation()
    pred=KMeans(n_clusters=k,n_init=20,random_state=SEED).fit_predict(lat)
    np.savez(f"{OUT}/emb_muraro_scVI.npz", emb=lat, y=yk, pred=pred); print("saved scVI", lat.shape)
except Exception as e:
    print("scVI FAILED:", e)

# ---- scDeepCluster ----
try:
    set_seed(SEED)
    REPO=f"{PROJ}/baselines/repos/scDeepCluster_pytorch"; sys.path.insert(0,REPO)
    from scDeepCluster import scDeepCluster
    from preprocess import read_dataset, normalize
    dev="cuda" if torch.cuda.is_available() else "cpu"
    ad=sc.AnnData(counts); ad.obs["Group"]=y; ad.obs["ct"]=ylab
    ad=read_dataset(ad, transpose=False, test_split=False, copy=True)
    ad=normalize(ad, size_factors=True, normalize_input=True, logtrans_input=True)
    yk=ad.obs["ct"].to_numpy()
    model=scDeepCluster(input_dim=ad.n_vars, z_dim=32, encodeLayer=[256,64], decodeLayer=[64,256],
                        sigma=2.5, gamma=1.0, device=dev)
    model.pretrain_autoencoder(X=ad.X, X_raw=ad.raw.X, size_factor=ad.obs.size_factors,
                               batch_size=256, epochs=300, ae_weights=f"{OUT}/_scdc_ae.pth.tar")
    yp,_,_,_,_=model.fit(X=ad.X, X_raw=ad.raw.X, size_factor=ad.obs.size_factors,
                         n_clusters=k, init_centroid=None, y_pred_init=None, y=y.astype(int)[:ad.n_obs],
                         batch_size=256, num_epochs=2000, update_interval=1, tol=0.001, save_dir=f"{OUT}/_scdc")
    lat=model.encodeBatch(torch.tensor(np.asarray(ad.X),dtype=torch.float32))
    lat=lat.detach().cpu().numpy() if torch.is_tensor(lat) else np.asarray(lat)
    np.savez(f"{OUT}/emb_muraro_scDeepCluster.npz", emb=lat, y=yk, pred=np.asarray(yp).astype(int)); print("saved scDeepCluster", lat.shape)
except Exception as e:
    import traceback; print("scDeepCluster FAILED:", e); traceback.print_exc()
