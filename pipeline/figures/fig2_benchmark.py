#!/usr/bin/env python3
"""Figure 2 (15 datasets). (a) ARI heatmap (b) three-metric means (c) rank mosaic.
 Cool-warm palette (high=warm red). 改名后改 METHOD_NAME。"""
import os, re, glob, json, sys
from collections import defaultdict
import statistics as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from scgtac_palette import (HEATMAP_CMAP, METRIC_COLORS, rank_colors,
                                 WARM, COOL, NEUTRAL, apply_nature_style)
except Exception:
    HEATMAP_CMAP='RdBu_r'; COOL='#2f5597'; NEUTRAL='#b5b5b5'; WARM='#a4262c'
    METRIC_COLORS=[COOL,NEUTRAL,WARM]
    def rank_colors(n=7,best='warm'):
        c=plt.get_cmap('RdBu' if best=='warm' else 'RdBu_r'); return [c(i/(n-1)) for i in range(n)]
    def apply_nature_style(): pass

METHOD_NAME="scGTAC"
PROJ="/home/liyang/BioJiaheWang/scGTAC"
OUT=f"{PROJ}/paper_figures_final/Fig2_benchmark"; os.makedirs(OUT,exist_ok=True)

LINEUP=["muraro_pancreas","baron","GSE103354","GSE103322","Tonsil","GSE150580_Mammary",
        "GSE119531","GSE194122_PBMC_Bench_1","multiome","GSE159115_ccRCC",
        "GSE194122_PBMC_Test","Goolam","Crohn","68kPBMC","GSE123516_labeled"]
DISP={"muraro_pancreas":"Muraro","baron":"Baron","GSE103354":"Airway","GSE103322":"Puram",
      "Tonsil":"Tonsil","GSE150580_Mammary":"Mammary","GSE119531":"UUO kidney",
      "GSE194122_PBMC_Bench_1":"BMMC-B1","multiome":"Multiome","GSE159115_ccRCC":"Kidney ccRCC",
      "GSE194122_PBMC_Test":"BMMC-test","Goolam":"Goolam","Crohn":"Crohn",
      "68kPBMC":"68k PBMC","GSE123516_labeled":"Intestine"}
BASE=["dec","scvi","scgnn","seurat","scdeepcluster","scdsc"]
BL={"dec":"DEC","scvi":"scVI","scgnn":"scGNN","seurat":"Seurat","scdeepcluster":"scDeepCluster","scdsc":"scDSC"}
ALIAS={"muraro":"muraro_pancreas"}

ACCENT="#1A1A1A"; BOX="#C8860D"; HEAT=HEATMAP_CMAP
MCOL={"ARI":COOL,"NMI":NEUTRAL,"ACC":WARM}
RANKCOL=rank_colors(7,'warm')

PATS={"ARI":"'ARI'","NMI":"'NMI'","ACC":"'CA'"}
sca=defaultdict(dict); raw=defaultdict(lambda:defaultdict(list))
for lg in glob.glob(f"{PROJ}/results/scagcr_final/*/run_seed*.log"):
    ds=os.path.basename(os.path.dirname(lg)); t=open(lg).read().strip()
    if not t: continue
    for mk,p in PATS.items():
        m=re.search(rf"{p}:\s*(?:np\.float64\()?([0-9.]+)",t.split("\n")[-1])
        if m: raw[ds][mk].append(float(m.group(1)))
for ds,dd in raw.items():
    for mk,v in dd.items():
        if v: sca[ds][mk]=st.mean(v)

blraw=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
for f in glob.glob(f"{PROJ}/results/baselines/*/*_metrics.json")+glob.glob(f"{PROJ}/results/baselines_chuli/*/*_metrics.json"):
    me=f.split("/")[-2]
    if me=="leiden": continue
    ds=re.sub(r"_seed\d+_metrics\.json$","",os.path.basename(f)); ds=ALIAS.get(ds,ds)
    try: d=json.load(open(f))
    except: continue
    for mk in ["ARI","NMI","ACC"]:
        if d.get(mk) is not None: blraw[ds][mk][me].append(float(d[mk]))
bl=defaultdict(lambda:defaultdict(dict))
for ds,dd in blraw.items():
    for mk,md in dd.items():
        for me,v in md.items(): bl[ds][mk][me]=st.mean(v)

def bmean(me,mk):
    vs=[(sca.get(d,{}).get(mk) if me==METHOD_NAME else bl.get(d,{}).get(mk,{}).get(me)) for d in LINEUP]
    vs=[v for v in vs if v is not None]; return np.mean(vs) if vs else 0
base_sorted=sorted(BASE,key=lambda m:-bmean(m,"ARI"))
methods=[METHOD_NAME]+base_sorted

M=np.full((len(methods),len(LINEUP)),np.nan)
for j,d in enumerate(LINEUP):
    for i,me in enumerate(methods):
        v=sca.get(d,{}).get("ARI") if me==METHOD_NAME else bl.get(d,{}).get("ARI",{}).get(me)
        if v is not None: M[i,j]=v

apply_nature_style()
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":7,"axes.linewidth":0.5,"xtick.labelsize":6.5,"ytick.labelsize":7,
    "xtick.major.width":0.5,"ytick.major.width":0.5,"xtick.major.size":2.5,"ytick.major.size":2.5,
    "legend.fontsize":6,"pdf.fonttype":42,"ps.fonttype":42,"savefig.bbox":"tight","savefig.dpi":600})

fig=plt.figure(figsize=(7.5,6.4))
gs=fig.add_gridspec(2,2,height_ratios=[1.25,1.45],hspace=0.35,wspace=0.5,
                    left=0.12,right=0.94,top=0.95,bottom=0.1)
axa=fig.add_subplot(gs[0,:]); axb=fig.add_subplot(gs[1,0]); axc=fig.add_subplot(gs[1,1])

im=axa.imshow(M,cmap=HEAT,vmin=0,vmax=1,aspect="auto")
axa.set_xticks(range(len(LINEUP))); axa.set_xticklabels([DISP[d] for d in LINEUP],rotation=35,ha="right")
axa.set_yticks(range(len(methods))); axa.set_yticklabels([METHOD_NAME]+[BL[m] for m in base_sorted])
axa.get_yticklabels()[0].set_fontweight("bold"); axa.tick_params(top=False,right=False)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v=M[i,j]
        if np.isnan(v): axa.text(j,i,"—",ha="center",va="center",fontsize=5.5,color="#6b6b66")
        else:
            tc="white" if (v>0.72 or v<0.28) else "#222222"
            axa.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=5.6,color=tc)
for j in range(M.shape[1]):
    col=M[:,j]
    if np.all(np.isnan(col)): continue
    ib=int(np.nanargmax(col)); axa.add_patch(Rectangle((j-0.46,ib-0.46),0.92,0.92,fill=False,edgecolor=BOX,linewidth=1.1))
axa.add_patch(Rectangle((-0.5,-0.5),M.shape[1],1.0,fill=False,edgecolor=ACCENT,linewidth=1.2))
cb=plt.colorbar(im,ax=axa,fraction=0.015,pad=0.01); cb.set_label("ARI",fontsize=7)
cb.ax.tick_params(labelsize=6,width=0.5); cb.outline.set_linewidth(0.5)

metrics=["ARI","NMI","ACC"]; bh=0.26; yc=np.arange(len(methods))
for k,mk in enumerate(metrics):
    axb.barh(yc+(1-k)*bh,[bmean(me,mk) for me in methods],height=bh,color=MCOL[mk],label=mk,edgecolor="white",linewidth=0.3)
axb.add_patch(Rectangle((0,yc[0]-0.5),1.0,1.0,color=ACCENT,alpha=0.08,zorder=0))
for k,mk in enumerate(metrics):
    sm=bmean(METHOD_NAME,mk); bb=max(bmean(me,mk) for me in BASE); imp=(sm-bb)*100
    axb.text(sm+0.012,yc[0]+(1-k)*bh,f"+{imp:.0f}%",va="center",ha="left",fontsize=5.6,fontweight="bold",color=MCOL[mk])
axb.set_yticks(yc); axb.set_yticklabels([METHOD_NAME]+[BL[m] for m in base_sorted])
axb.get_yticklabels()[0].set_fontweight("bold"); axb.invert_yaxis()
axb.set_xlim(0,1.05); axb.set_xticks([0,0.2,0.4,0.6,0.8,1.0]); axb.set_xlabel("Mean score (15 datasets)")
axb.spines["top"].set_visible(False); axb.spines["right"].set_visible(False)
axb.legend(loc="lower right",frameon=False,ncol=3,columnspacing=0.7,handlelength=0.9,bbox_to_anchor=(1.02,-0.3))

RANKS=defaultdict(lambda:defaultdict(int))
for ds in LINEUP:
    for mk in metrics:
        sc={}
        s=sca.get(ds,{}).get(mk)
        if s is not None: sc[METHOD_NAME]=s
        for me in BASE:
            v=bl.get(ds,{}).get(mk,{}).get(me)
            if v is not None: sc[me]=v
        for r,(me,_) in enumerate(sorted(sc.items(),key=lambda x:-x[1]),1):
            RANKS[me][r]+=1
methods_c=sorted([METHOD_NAME]+BASE,key=lambda m:-RANKS[m][1])
for i,me in enumerate(methods_c):
    left=0
    for r in range(1,8):
        w=RANKS[me].get(r,0)
        if w==0: continue
        axc.barh(i,w,left=left,color=RANKCOL[r-1],edgecolor="white",linewidth=0.4)
        left+=w
axc.text(RANKS[METHOD_NAME].get(1,0)/2,0,f"{RANKS[METHOD_NAME].get(1,0)}× #1",
         va="center",ha="center",fontsize=5.8,fontweight="bold",color="white")
axc.set_yticks(range(len(methods_c))); axc.set_yticklabels([m if m==METHOD_NAME else BL[m] for m in methods_c])
axc.get_yticklabels()[0].set_fontweight("bold"); axc.invert_yaxis()
axc.set_xlabel("Rank count (15 datasets × 3 metrics = 45)")
axc.spines["top"].set_visible(False); axc.spines["right"].set_visible(False)
handles=[Patch(facecolor=RANKCOL[r],label=f"{r+1}") for r in range(7)]
axc.legend(handles=handles,title="Rank",loc="lower right",frameon=False,ncol=7,
           fontsize=5.2,title_fontsize=6,columnspacing=0.4,handlelength=0.8,handletextpad=0.3,bbox_to_anchor=(1.03,-0.32))

fig.text(0.01, 0.98, "a", fontsize=13, fontweight="bold", va="top", ha="left")
fig.text(0.01, 0.52, "b", fontsize=13, fontweight="bold", va="top", ha="left")
fig.text(0.50, 0.52, "c", fontsize=13, fontweight="bold", va="top", ha="left")

for ext in ["pdf","svg","png"]: fig.savefig(f"{OUT}/Figure2.{ext}"); print("saved",ext)
plt.close(fig); print("done ->",OUT)
