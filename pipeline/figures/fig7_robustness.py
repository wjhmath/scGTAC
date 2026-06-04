"""鲁棒性: down-sampling ARI 折线图"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt, os

PROJ="/home/liyang/BioJiaheWang/scAGCR"
OUT=f"{PROJ}/paper_figures_final/Fig7_robustness"; os.makedirs(OUT,exist_ok=True)
df=pd.read_csv(f"{PROJ}/results/robustness/downsample.csv")
df=df.dropna(subset=["ARI"])  # 去掉缺失行

DS_LABEL={"baron":"Baron (8,569)","GSE103322":"GSE103322 (3,363)","GSE119531":"GSE119531 (6,147)"}
COLORS={"baron":"#5E7367","GSE103322":"#90A0AE","GSE119531":"#C49A8E"}
MARKERS={"baron":"o","GSE103322":"s","GSE119531":"D"}

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6,
    "pdf.fonttype":42})

fig,ax=plt.subplots(figsize=(6.5,4))
for ds in ["baron","GSE103322","GSE119531"]:
    sub=df[df["dataset"]==ds]
    grp=sub.groupby("ratio")["ARI"]
    means=grp.mean(); stds=grp.std()
    ax.plot(means.index,means.values,"-",marker=MARKERS[ds],color=COLORS[ds],
            ms=6,lw=1.8,label=DS_LABEL[ds],zorder=3)
    ax.fill_between(means.index,means.values-stds.values,means.values+stds.values,
                    color=COLORS[ds],alpha=0.15,zorder=2)

ax.set_xlabel("Down-sampling ratio",fontsize=10)
ax.set_ylabel("ARI",fontsize=10)
ax.set_xticks([0.1,0.3,0.5,0.7,0.9,1.0])
ax.set_xlim(0.05,1.05); ax.set_ylim(0.3,1.0)
ax.legend(frameon=False,fontsize=8,loc="lower right")
ax.tick_params(length=3,width=0.6)
ax.axhline(y=0.5,color="#ddd",linewidth=0.5,linestyle="--",zorder=1)
plt.tight_layout()
for ext in ["pdf","svg","png"]:
    fig.savefig(f"{OUT}/Figure7.{ext}",dpi=300,bbox_inches="tight")
print(f"done -> {OUT}")
