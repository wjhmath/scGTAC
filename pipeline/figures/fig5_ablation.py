"""Fig.5 Ablation — clean delta bars, no error bars"""
import numpy as np, matplotlib.pyplot as plt, glob, re, os

PROJ="/home/liyang/BioJiaheWang/scAGCR"
OUT=f"{PROJ}/paper_figures_final/Fig5_ablation"; os.makedirs(OUT,exist_ok=True)

def collect(var, ds):
    data={"ARI":[],"NMI":[],"CA":[]}
    for f in glob.glob(f"{PROJ}/results/ablation/{var}/{ds}/run_seed*.log"):
        t=open(f,errors="ignore").read().strip().splitlines()
        if not t: continue
        ln=t[-1]
        for mk,pat in [("ARI",r"'ARI':\s*(?:np\.float64\()?([0-9.]+)"),
                        ("NMI",r"'NMI':\s*(?:np\.float64\()?([0-9.]+)"),
                        ("CA", r"'CA':\s*(?:np\.float64\()?([0-9.]+)")]:
            m=re.search(pat,ln)
            if m: data[mk].append(float(m.group(1)))
    return {k:np.array(v) for k,v in data.items()}

DATASETS=["GSE119531","GSE103322","GSE159115_ccRCC"]
MODULES=[("wo_cl","w/o CL"),("wo_feature_aug","w/o Feat. aug."),("wo_edge_aug","w/o Edge aug.")]
METRICS=["ARI","NMI","CA"]; MTITLES=["ARI","NMI","ACC"]
BAR_COL=["#7E9486","#A3B899","#C4B08E"]

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6,
    "pdf.fonttype":42,"savefig.dpi":600})

fig,axes=plt.subplots(1,3,figsize=(10,3.5))
x=np.arange(len(MODULES)); w=0.55

for ax,metric,mtitle in zip(axes,METRICS,MTITLES):
    means=[]
    for var,label in MODULES:
        ds_deltas=[]
        for ds in DATASETS:
            fu=collect("full",ds)[metric]
            ab=collect(var,ds)[metric]
            if fu.size and ab.size: ds_deltas.append(fu.mean()-ab.mean())
        means.append(np.mean(ds_deltas) if ds_deltas else 0)
    
    bars=ax.bar(x,means,w,color=BAR_COL,edgecolor="white",linewidth=0.5,alpha=0.9,zorder=2)
    ax.axhline(0,color="#ccc",linewidth=0.5,linestyle="-",zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _,l in MODULES],fontsize=8.5)
    ax.set_title(mtitle,fontsize=11,fontweight="bold")
    ax.set_ylabel("Δ Score" if ax==axes[0] else "",fontsize=10)
    ax.set_ylim(0, max(means)*1.2)
    ax.tick_params(length=3,width=0.6)

for ax,lab in zip(axes,["a","b","c"]):
    ax.text(-0.02,1.10,lab,transform=ax.transAxes,fontsize=12,fontweight="bold",va="top")
plt.tight_layout()
for ext in ["pdf","svg","png"]:
    fig.savefig(f"{OUT}/Figure5.{ext}",dpi=300,bbox_inches="tight")
print(f"done -> {OUT}")
