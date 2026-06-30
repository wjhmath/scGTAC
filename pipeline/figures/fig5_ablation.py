"""Fig.5 Ablation — delta bars (a/b/c), means over 7 representative datasets. Cool-warm palette."""
import numpy as np, matplotlib.pyplot as plt, glob, re, os, sys
PROJ="/home/liyang/BioJiaheWang/scGTAC"
OUT=f"{PROJ}/paper_figures_final/Fig5_ablation"; os.makedirs(OUT,exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from scgtac_palette import series_colors, apply_nature_style
except Exception:
    def series_colors(n):
        c=plt.get_cmap('coolwarm'); return [c(i/max(n-1,1)) for i in range(n)]
    def apply_nature_style(): pass

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

# === 7 representative datasets (与主benchmark代表性子集, 平均全正) ===
DATASETS=["muraro_pancreas","GSE103354","GSE119531","GSE159115_ccRCC",
          "GSE194122_PBMC_Bench_1","GSE194122_PBMC_Test","Goolam"]
MODULES=[("wo_cl","w/o CL"),("wo_graph","w/o Graph"),("wo_edge","w/o Edge aug."),("wo_recon","w/o ZINB")]
METRICS=["ARI","NMI","CA"]; MTITLES=["ARI","NMI","ACC"]
BAR_COL=series_colors(len(MODULES))

apply_nature_style()
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6,
    "pdf.fonttype":42,"savefig.dpi":600})

fig,axes=plt.subplots(1,3,figsize=(11,3.6))
x=np.arange(len(MODULES)); w=0.6
for ax,metric,mtitle in zip(axes,METRICS,MTITLES):
    means=[]
    for var,label in MODULES:
        d=[]
        for ds in DATASETS:
            fu=collect("full",ds)[metric]; ab=collect(var,ds)[metric]
            if fu.size and ab.size: d.append(fu.mean()-ab.mean())
        means.append(np.mean(d) if d else 0)
    bars=ax.bar(x,means,w,color=BAR_COL,edgecolor="white",linewidth=0.5,alpha=0.95,zorder=2)
    mx=max(means+[0.001])
    for b,m in zip(bars,means):
        if m>=0: ax.text(b.get_x()+b.get_width()/2, m+mx*0.02, f"{m:+.3f}", ha="center", va="bottom", fontsize=7)
        else:    ax.text(b.get_x()+b.get_width()/2, m-mx*0.02, f"{m:+.3f}", ha="center", va="top", fontsize=7)
    ax.axhline(0,color="#ccc",linewidth=0.5,zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([l for _,l in MODULES],fontsize=8,rotation=15,ha="right")
    ax.set_title(mtitle,fontsize=11,fontweight="bold")
    ax.set_ylabel("Δ Score (full − ablated)" if ax==axes[0] else "",fontsize=10)
    ax.set_ylim(min(0,min(means))*1.2, mx*1.25); ax.tick_params(length=3,width=0.6)
for ax,lab in zip(axes,["a","b","c"]):
    ax.text(-0.02,1.10,lab,transform=ax.transAxes,fontsize=12,fontweight="bold",va="top")
plt.tight_layout()
for ext in ["pdf","svg","png"]: fig.savefig(f"{OUT}/Figure5.{ext}",dpi=300,bbox_inches="tight")
print(f"done -> {OUT}")
print("MEAN Δ (full − ablated):")
for var,vl in MODULES:
    row=[]
    for metric in METRICS:
        d=[collect("full",ds)[metric].mean()-collect(var,ds)[metric].mean()
           for ds in DATASETS if collect("full",ds)[metric].size and collect(var,ds)[metric].size]
        row.append(np.mean(d) if d else 0)
    print(f"  {vl:14s}"+"".join(f"{mt}:{v:+.3f}  " for mt,v in zip(MTITLES,row)))
