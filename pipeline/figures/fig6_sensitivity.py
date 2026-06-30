"""Fig6: 参数敏感性 tau/lambda_cl/dropout. Cool-warm palette."""
import numpy as np, matplotlib.pyplot as plt, glob, re, os, sys
PROJ="/home/liyang/BioJiaheWang/scGTAC"
OUT=f"{PROJ}/paper_figures_final/Fig6_sensitivity"; os.makedirs(OUT,exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from scgtac_palette import COOL, NEUTRAL, WARM, apply_nature_style
except Exception:
    COOL='#2f5597'; NEUTRAL='#b5b5b5'; WARM='#a4262c'
    def apply_nature_style(): pass

def collect(tag):
    data={"ARI":[],"NMI":[],"CA":[]}
    for f in glob.glob(f"{PROJ}/results/param_sweep/{tag}/baron/run_seed*.log"):
        t=open(f,errors="ignore").read().strip().splitlines()
        if not t: continue
        ln=t[-1]
        for mk,pat in [("ARI",r"'ARI':\s*(?:np\.float64\()?([0-9.]+)"),
                        ("NMI",r"'NMI':\s*(?:np\.float64\()?([0-9.]+)"),
                        ("CA", r"'CA':\s*(?:np\.float64\()?([0-9.]+)")]:
            m=re.search(pat,ln)
            if m: data[mk].append(float(m.group(1)))
    return {k:np.array(v) for k,v in data.items()}

PANELS=[("tau",[0.2,0.4,0.6,0.8,1.0],0.8,r"$\tau$"),
        ("lambda_cl",[0.1,0.3,0.5,0.7,0.9],0.7,r"$\lambda_{cl}$"),
        ("dropout",[0.0,0.1,0.2,0.3,0.5],0.3,"Dropout")]
METRICS=["ARI","NMI","CA"]; MLABEL={"ARI":"ARI","NMI":"NMI","CA":"ACC"}
MCOL={"ARI":COOL,"NMI":NEUTRAL,"CA":WARM}

apply_nature_style()
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6,
    "pdf.fonttype":42,"savefig.dpi":600})

fig,axes=plt.subplots(1,3,figsize=(11,3.5),sharey=True)
for ax,(param,vals,default,xlabel) in zip(axes,PANELS):
    for mk in METRICS:
        means,stds=[],[]
        for v in vals:
            d=collect(f"{param}_{v}")[mk]
            means.append(d.mean() if d.size else np.nan)
            stds.append(d.std() if d.size>1 else 0)
        means,stds=np.array(means),np.array(stds)
        ax.plot(vals,means,"-o",color=MCOL[mk],ms=5,lw=1.5,label=MLABEL[mk],zorder=3)
        ax.fill_between(vals,means-stds,means+stds,color=MCOL[mk],alpha=0.15,zorder=2)
    ax.axvline(default,color="#ccc",linewidth=1,linestyle="--",zorder=1)
    ax.text(default,ax.get_ylim()[0]+0.01,"default",ha="center",va="bottom",fontsize=7,color="#999")
    ax.set_xlabel(xlabel,fontsize=11); ax.tick_params(length=3,width=0.6)
axes[0].set_ylabel("Score",fontsize=11); axes[0].set_ylim(0.70,0.95)
axes[-1].legend(frameon=False,fontsize=8,loc="lower right")
for ax,lab in zip(axes,["a","b","c"]):
    ax.text(-0.02,1.10,lab,transform=ax.transAxes,fontsize=12,fontweight="bold",va="top")
plt.tight_layout()
for ext in ["pdf","svg","png"]: fig.savefig(f"{OUT}/Figure6.{ext}",dpi=300,bbox_inches="tight")
print(f"done -> {OUT}")
