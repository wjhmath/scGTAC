import os, sys, numpy as np, pandas as pd, scipy.sparse as sps, subprocess, tempfile
PROJ="/home/liyang/BioJiaheWang/scGTAC"
sys.path.insert(0, f"{PROJ}/baselines")
from common import load_dataset, set_seed
DATA=f"{PROJ}/data/muraro_pancreas/muraro_pancreas.h5ad"
OUT=f"{PROJ}/paper_figures/Figure3_umap"; os.makedirs(OUT,exist_ok=True)
SEED=1; NHVG=2000
set_seed(SEED)
adata,X,y,k=load_dataset(DATA); ylab=adata.obs["cell_type"].astype(str).to_numpy()
cand=X.toarray() if sps.issparse(X) else np.asarray(X); cand=cand.astype("float64")
if not np.allclose(cand,np.round(cand)): cand=np.round(cand)
if cand.shape[1]>NHVG:
    var=cand.var(axis=0); top=np.sort(np.argsort(var)[-NHVG:]); cand=cand[:,top]
n,g=cand.shape
R=r"""
library(Seurat); set.seed({seed})
counts<-as.matrix(read.csv("{cc}",row.names=1,check.names=FALSE)); counts<-t(counts)
obj<-CreateSeuratObject(counts=counts,min.cells=0,min.features=0)
obj<-NormalizeData(obj,verbose=FALSE); obj<-FindVariableFeatures(obj,nfeatures={nh},verbose=FALSE)
obj<-ScaleData(obj,verbose=FALSE); np<-min(50,ncol(obj)-1); obj<-RunPCA(obj,npcs=np,verbose=FALSE)
obj<-FindNeighbors(obj,dims=1:np,verbose=FALSE)
tk<-{k}; lo<-0.01; hi<-5.0; br<-0.8
for(i in 1:30){{r<-(lo+hi)/2; obj<-FindClusters(obj,resolution=r,random.seed={seed},verbose=FALSE)
 nc<-length(unique(Idents(obj))); br<-r; if(nc==tk)break; if(nc<tk)lo<-r else hi<-r}}
write.csv(data.frame(pred=as.integer(Idents(obj))-1L),"{pc}",row.names=FALSE)
write.csv(Embeddings(obj,"pca"),"{ec}",row.names=FALSE)
"""
rsc=os.path.join(os.path.dirname(sys.executable),"Rscript")
with tempfile.TemporaryDirectory() as td:
    cc=f"{td}/c.csv"; pc=f"{td}/p.csv"; ec=f"{td}/e.csv"; rp=f"{td}/r.R"
    pd.DataFrame(cand, index=[f"c{i}" for i in range(n)], columns=[f"g{j}" for j in range(g)]).to_csv(cc)
    open(rp,"w").write(R.format(cc=cc,pc=pc,ec=ec,k=k,seed=SEED,nh=NHVG))
    ret=subprocess.run([rsc,"--vanilla",rp],capture_output=True,text=True)
    print(ret.stdout[-500:] if ret.stdout else ""); 
    if ret.returncode!=0: print("R ERR:",ret.stderr[-800:]); sys.exit(1)
    pred=pd.read_csv(pc)["pred"].to_numpy().astype(int)
    emb=pd.read_csv(ec).to_numpy().astype(np.float32)
np.savez(f"{OUT}/emb_muraro_Seurat.npz", emb=emb, y=ylab, pred=pred); print("saved Seurat", emb.shape)
