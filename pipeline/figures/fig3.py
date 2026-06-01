#!/usr/bin/env python3
"""Figure 3 (Muraro): a) UMAP grid  b) per-type recovery  c) marker dotplot."""
import os, glob, numpy as np, pandas as pd, scanpy as sc
import matplotlib.pyplot as plt, matplotlib as mpl, matplotlib.lines as ml
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

METHOD="scAGCR"
PROJ="/home/liyang/BioJiaheWang/scAGCR"; OUT=f"{PROJ}/paper_figures/Figure3_umap"
DATA=f"{PROJ}/data/muraro_pancreas/muraro_pancreas.h5ad"; CACHE=f"{OUT}/_umap_cache.npz"
BL=[("scvi","scVI"),("scgnn","scGNN"),("seurat","Seurat"),("dec","DEC"),("scdeepcluster","scDeepCluster"),("scdsc","scDSC")]

if os.path.exists(CACHE):
    z=np.load(CACHE,allow_pickle=True); coords=z["coords"]; true_str=z["true_str"].astype(str)
else:
    ad=sc.read_h5ad(DATA); ad.obs_names_make_unique(); ad.var_names_make_unique()
    true_str=ad.obs['cell_type'].astype(str).values
    if float(ad.X.max())>30: sc.pp.normalize_total(ad,target_sum=1e4); sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad,n_top_genes=2000,flavor="seurat"); ad=ad[:,ad.var.highly_variable].copy()
    sc.pp.scale(ad,max_value=10); sc.pp.pca(ad,n_comps=50); sc.pp.neighbors(ad,n_neighbors=15); sc.tl.umap(ad,random_state=0,min_dist=0.3)
    coords=ad.obsm["X_umap"]; np.savez(CACHE,coords=coords,true_str=true_str)
N=coords.shape[0]; cats=sorted(set(true_str)); c2i={c:i for i,c in enumerate(cats)}
true_code=np.array([c2i[s] for s in true_str]); cnt={c:int((true_str==c).sum()) for c in cats}

def match(pred,true):
    pred=np.asarray(pred).astype(int); true=np.asarray(true).astype(int)
    D=max(pred.max(),true.max())+1; w=np.zeros((D,D),int)
    for p,t in zip(pred,true): w[p,t]+=1
    r,c=linear_sum_assignment(w.max()-w); m=dict(zip(r,c))
    return np.array([m.get(p,p) for p in pred])

scagcr_pred=np.load(f"{OUT}/emb_muraro_{METHOD}.npz",allow_pickle=True)["pred"]
preds={METHOD:scagcr_pred}
for key,name in BL:
    cd=glob.glob(f"{PROJ}/results/baselines*/{key}/*muraro*pred.csv")
    if cd:
        pr=pd.read_csv(sorted(cd)[0])["pred"].to_numpy()
        if len(pr)==N: preds[name]=pr
order_m=[METHOD]+[n for _,n in BL if n in preds]
panels=[("Ground truth",true_code,None)]+[(n,match(preds[n],true_code),adjusted_rand_score(true_code,preds[n])) for n in order_m]
def recall(pred):
    m=match(pred,true_code); return [ (m[true_code==i]==i).mean() if (true_code==i).sum() else np.nan for i in range(len(cats))]
R=np.array([recall(preds[n]) for n in order_m]).T
ro=np.argsort([cnt[c] for c in cats]); R=R[ro]; rowcats=[cats[i] for i in ro]

adx=sc.read_h5ad(DATA); adx.var_names_make_unique()
if float(adx.X.max())>30: sc.pp.normalize_total(adx,target_sum=1e4); sc.pp.log1p(adx)
fn=adx.var['feature_name'].astype(str).to_numpy(); sym2ens={}
for ens,nm in zip(adx.var_names.to_numpy(),fn): sym2ens.setdefault(nm,ens)
MARK=["GCG","INS","SST","PPY","GHRL","PRSS1","CPA1","KRT19","CFTR","COL1A1","PECAM1","VWF","CHGA"]
genes=[(g,sym2ens[g]) for g in MARK if g in sym2ens]; ens=[e for _,e in genes]
TYPEORDER=["pancreatic A cell","type B pancreatic cell","pancreatic D cell","pancreatic PP cell",
           "pancreatic epsilon cell","pancreatic acinar cell","pancreatic ductal cell","mesenchymal cell","endothelial cell","pancreatic endocrine cell"]
SHORT={"pancreatic A cell":"A/α","type B pancreatic cell":"B/β","pancreatic D cell":"D/δ","pancreatic PP cell":"PP",
       "pancreatic epsilon cell":"ε","pancreatic acinar cell":"acinar","pancreatic ductal cell":"ductal",
       "mesenchymal cell":"mesen.","endothelial cell":"endoth.","pancreatic endocrine cell":"endo."}
clusters=sorted(set(scagcr_pred)); dom={c:pd.Series(true_str[scagcr_pred==c]).mode()[0] for c in clusters}
clord=sorted(clusters,key=lambda c:TYPEORDER.index(dom[c]) if dom[c] in TYPEORDER else 99)
sub=adx[:,ens].X; sub=sub.toarray() if hasattr(sub,"toarray") else np.asarray(sub)
ncl,ng=len(clord),len(genes); pct=np.zeros((ncl,ng)); me=np.zeros((ncl,ng))
for i,c in enumerate(clord):
    m=scagcr_pred==c; s=sub[m]; pct[i]=(s>0).mean(0)*100; me[i]=s.mean(0)
scld=(me-me.min(0))/(me.max(0)-me.min(0)+1e-9)

PAL=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3","#937860","#DA8BC3","#7F7F7F","#CCB974","#64B5CD","#B07AA1","#E15759"]
col={i:PAL[i%len(PAL)] for i in range(len(cats))}
HEAT=LinearSegmentedColormap.from_list("h",["#EFEAE1","#D2D4C6","#AFBCAC","#869B8B","#5E7367"])
DOT=LinearSegmentedColormap.from_list("d",["#EFEAE1","#AFBCAC","#5E7367"])
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":7,
    "axes.linewidth":0.5,"pdf.fonttype":42,"savefig.bbox":"tight","savefig.dpi":600})

fig=plt.figure(figsize=(10,7.8))
gsa=fig.add_gridspec(2,4,left=0.045,right=0.80,top=0.965,bottom=0.545,wspace=0.10,hspace=0.30)
axa=[fig.add_subplot(gsa[r,c]) for r in range(2) for c in range(4)]
for ax,(name,lab,ari) in zip(axa,panels):
    for i in range(len(cats)):
        mk=(lab==i)
        if mk.sum(): ax.scatter(coords[mk,0],coords[mk,1],s=2.5,c=col[i],linewidths=0,rasterized=True)
    ax.set_title(name if ari is None else f"{name} ({ari:.3f})",fontsize=7.5,fontweight="bold" if name in(METHOD,"Ground truth") else "normal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.5)
for ax in axa[len(panels):]: ax.axis("off")
h=[ml.Line2D([0],[0],marker='o',ls='',ms=5,mfc=col[i],mec='none',label=cats[i]) for i in range(len(cats))]
fig.legend(handles=h,loc="center left",bbox_to_anchor=(0.805,0.755),frameon=False,fontsize=6.3,handletextpad=0.4)

axb=fig.add_axes([0.075,0.085,0.30,0.40]); caxb=fig.add_axes([0.385,0.085,0.012,0.40])
im=axb.imshow(R,cmap=HEAT,vmin=0,vmax=1,aspect="auto")
axb.set_xticks(range(len(order_m))); axb.set_xticklabels(order_m,rotation=35,ha="right"); axb.get_xticklabels()[0].set_fontweight("bold")
axb.set_yticks(range(len(rowcats))); axb.set_yticklabels([f"{SHORT.get(c,c)} (n={cnt[c]})" for c in rowcats],fontsize=6.3)
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if not np.isnan(R[i,j]): axb.text(j,i,f"{R[i,j]:.2f}",ha="center",va="center",fontsize=5.4,color="white" if R[i,j]>0.55 else "#3A3A38")
axb.add_patch(Rectangle((-0.5,-0.5),1,R.shape[0],fill=False,edgecolor="#A8625A",linewidth=1.2)); axb.tick_params(length=2,width=0.5)
cb=fig.colorbar(im,cax=caxb); cb.set_label("Per-type recall",fontsize=6.5); cb.ax.tick_params(labelsize=6,width=0.5); cb.outline.set_linewidth(0.5)

# (c) dotplot
axc=fig.add_axes([0.55,0.105,0.245,0.38])
for i in range(ncl):
    for j in range(ng):
        axc.scatter(j,i,s=pct[i,j]*1.7,c=[DOT(scld[i,j])],edgecolors="#9a9a92",linewidths=0.3,zorder=3)
axc.set_xticks(range(ng)); axc.set_xticklabels([g for g,_ in genes],rotation=45,ha="right",fontsize=6.3)
axc.set_yticks(range(ncl)); axc.set_yticklabels([f"c{c}·{SHORT.get(dom[c],dom[c])}" for c in clord],fontsize=6.3)
axc.set_xlim(-0.6,ng-0.4); axc.set_ylim(-0.6,ncl-0.4); axc.invert_yaxis()
axc.set_axisbelow(True); axc.grid(True,lw=0.3,color="#E8E4DB",zorder=0)
for s in axc.spines.values(): s.set_linewidth(0.5)
axc.tick_params(length=2,width=0.5)

# ---- (c) 图例: 圆圈与色柱居中同一 x, 标题都横排 ----
CX=0.86
axl=fig.add_axes([0.81,0.30,0.10,0.175]); axl.set_xlim(0,1); axl.set_ylim(0,1); axl.axis("off")
axl.text(0.5,1.08,"% expr",ha="center",va="bottom",fontsize=6.3)
for p,yy in zip([25,50,75,100],[0.85,0.60,0.35,0.12]):
    axl.scatter(0.5,yy,s=p*1.7,c="#B8B3A8",edgecolors="#9a9a92",linewidths=0.3)
    axl.text(0.82,yy,f"{p}",va="center",ha="left",fontsize=6)
caxc=fig.add_axes([CX-0.007,0.105,0.014,0.15]); sm=plt.cm.ScalarMappable(cmap=DOT); sm.set_array([0,1])
cc=fig.colorbar(sm,cax=caxc); cc.ax.set_title("Scaled expr",fontsize=6.3,pad=4)
cc.ax.tick_params(labelsize=6,width=0.5); cc.set_ticks([0,0.5,1]); cc.outline.set_linewidth(0.5)

fig.text(0.03,0.965,"a",fontsize=12,fontweight="bold",va="top")
fig.text(0.03,0.51,"b",fontsize=12,fontweight="bold",va="top")
fig.text(0.50,0.51,"c",fontsize=12,fontweight="bold",va="top")

for ext in ["pdf","svg","png"]: fig.savefig(f"{OUT}/Figure3_full.{ext}")
print("done -> Figure3_full")
