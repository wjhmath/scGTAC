"""Fig.8: Scalability (a) Simulated (b) Real. Cool-warm palette."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os, sys
from scipy.optimize import curve_fit
PROJ="/home/liyang/BioJiaheWang/scGTAC"
OUT=f"{PROJ}/paper_figures_final/Fig8_scalability"; os.makedirs(OUT,exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from scgtac_palette import WARM, NEUTRAL, apply_nature_style
except Exception:
    WARM='#a4262c'; NEUTRAL='#b5b5b5'
    def apply_nature_style(): pass

def power(x,a,b): return a*np.power(x,b)
def linear(x,a,b): return a*x+b
SHORT={"muraro_pancreas":"Muraro","GSE123516_labeled":"Intestine","GSE150580_Mammary":"Mammary",
    "GSE159115_ccRCC":"Kidney ccRCC","GSE194122_PBMC_Bench_1":"BMMC-B1",
    "GSE194122_PBMC_Test":"BMMC-test","10X_PBMC":"10X PBMC","GSE103354":"Airway",
    "GSE103322":"Puram","GSE119531":"UUO kidney","68kPBMC":"68k PBMC"}

apply_nature_style()
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6,"pdf.fonttype":42,"savefig.dpi":600})
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
syn=pd.read_csv(f"{PROJ}/results/scalability/synthetic_timing.csv")
syn["minutes"]=syn["seconds"]/60
ax=axes[0]
ax.scatter(syn["n_cells"],syn["minutes"],s=55,c=WARM,edgecolors="white",linewidths=0.6,zorder=4)
popt,_=curve_fit(linear,syn["n_cells"],syn["minutes"])
xs=np.linspace(0,syn["n_cells"].max()*1.05,200)
r2=np.corrcoef(syn["n_cells"],syn["minutes"])[0,1]**2
ax.plot(xs,linear(xs,*popt),color=NEUTRAL,linewidth=1.5,linestyle="--",zorder=2,label=f"Linear fit (R$^2$={r2:.3f})")
for _,r in syn.iterrows():
    lb=f"{int(r['n_cells']/1000)}k" if r["n_cells"]>=1000 else str(int(r["n_cells"]))
    ax.annotate(lb,(r["n_cells"],r["minutes"]),textcoords="offset points",xytext=(6,4),fontsize=7.5,color="#555")
ax.set_xlabel("Number of cells",fontsize=10); ax.set_ylabel("Training time (minutes)",fontsize=10)
ax.set_title("Simulated data",fontsize=10,fontweight="bold"); ax.legend(frameon=False,fontsize=8,loc="upper left")
real=pd.read_csv(f"{PROJ}/results/scalability/timing.csv").sort_values("n_cells")
real["minutes"]=real["seconds"]/60
real["label"]=real["dataset"].map(lambda x:SHORT.get(x,x))
ax=axes[1]
ax.scatter(real["n_cells"],real["minutes"],s=55,c=WARM,edgecolors="white",linewidths=0.6,zorder=4)
popt2,_=curve_fit(power,real["n_cells"],real["minutes"],p0=[0.01,0.5])
xs2=np.linspace(real["n_cells"].min(),real["n_cells"].max(),200)
ax.plot(xs2,power(xs2,*popt2),color=NEUTRAL,linewidth=1.5,linestyle="--",zorder=2,label=rf"Fit: t $\propto$ N$^{{{popt2[1]:.2f}}}$")
try:
    from adjustText import adjust_text
    texts=[ax.text(r["n_cells"],r["minutes"],r["label"],fontsize=6.5,color="#555") for _,r in real.iterrows()]
    adjust_text(texts,ax=ax,arrowprops=dict(arrowstyle="-",color="#ccc",lw=0.4))
except:
    for _,r in real.iterrows(): ax.annotate(r["label"],(r["n_cells"],r["minutes"]),textcoords="offset points",xytext=(5,3),fontsize=6,color="#555")
ax.set_xlabel("Number of cells",fontsize=10); ax.set_ylabel("Training time (minutes)",fontsize=10)
ax.set_title("Real datasets",fontsize=10,fontweight="bold"); ax.legend(frameon=False,fontsize=8,loc="upper left")
for ax,lab in zip(axes,["a","b"]):
    ax.tick_params(length=3,width=0.6)
    ax.text(-0.08,1.05,lab,transform=ax.transAxes,fontsize=12,fontweight="bold",va="top")
plt.tight_layout()
for ext in ["pdf","svg","png"]: fig.savefig(f"{OUT}/Figure8.{ext}",dpi=300,bbox_inches="tight")
plt.close()
print(f"done -> {OUT}")
