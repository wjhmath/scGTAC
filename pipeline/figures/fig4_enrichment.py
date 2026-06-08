import scanpy as sc, numpy as np, matplotlib.pyplot as plt, pandas as pd
import matplotlib.gridspec as gridspec, matplotlib.image as mpimg
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os, warnings
warnings.filterwarnings("ignore")
PROJ = "/home/liyang/BioJiaheWang/scGTAC"
OUT = PROJ + "/paper_figures_final/Fig4_bio_enrichment"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"font.size":10,"pdf.fonttype":42})
adata = sc.read_h5ad(PROJ+"/data/muraro_pancreas/muraro_pancreas.h5ad")
adata.raw = None
names = adata.var["feature_name"].fillna(pd.Series(adata.var_names, index=adata.var.index))
adata.var_names = pd.Index(names.values); adata.var_names_make_unique()
adata.obs["celltype"] = adata.obs["cell_type"].astype(str)
sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
adata.raw = adata.copy()
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata = adata[:, adata.var.highly_variable].copy()
sc.pp.scale(adata, max_value=10); sc.tl.pca(adata, n_comps=30)
sc.pp.neighbors(adata, n_pcs=30); sc.tl.umap(adata)
MARKERS = ["GCG","INS","SST","PPY","REG1A","ESAM"]
TITLES = ["\u03b1: GCG","\u03b2: INS","\u03b4: SST","PP: PPY","Acinar: REG1A","Endoth.: ESAM"]
fig_f, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, gene, title in zip(axes.flatten(), MARKERS, TITLES):
    if gene in adata.raw.var_names:
        sc.pl.umap(adata, color=gene, ax=ax, show=False, title=title, frameon=False, size=15, cmap="RdYlBu_r", vmin=0, colorbar_loc="right")
        ax.set_title(title, fontsize=11, fontweight="bold")
    else: ax.axis("off")
plt.tight_layout(h_pad=1.5, w_pad=1.0)
fig_f.savefig(OUT+"/_tf.png", dpi=300, bbox_inches="tight"); plt.close()
print("[ok] feature")
df = pd.read_csv(PROJ+"/paper_figures/Figure_marker/enrichment_results.csv")
df["Count"] = df["Overlap"].str.split("/").str[0].astype(int)
df["neglog10p"] = -np.log10(df["Adjusted P-value"].clip(1e-30))
df["Term_clean"] = df["Term"].str.split("(").str[0].str.strip().str[:40]
CT_ORDER = ["A/alpha","B/beta","D/delta","Acinar","Ductal"]
CT_DISPLAY = {"A/alpha":"\u03b1 cell","B/beta":"\u03b2 cell","D/delta":"\u03b4 cell","Acinar":"Acinar","Ductal":"Ductal"}
for lib, title, cmap, pn in [("GO","GO Biological Process","YlGn","b"),("KEGG","KEGG Pathway","YlOrRd","c")]:
    sub = df[df["library"]==lib]
    if len(sub)==0: continue
    top_t = pd.concat([sub[sub["celltype"]==ct].nlargest(4,"neglog10p") for ct in CT_ORDER], ignore_index=True)
    at = []
    for ct in CT_ORDER:
        for t in top_t[top_t["celltype"]==ct]["Term_clean"].tolist():
            if t not in at: at.append(t)
    t2y = {t:i for i,t in enumerate(at)}; c2x = {c:i for i,c in enumerate(CT_ORDER)}
    fig_e, ax = plt.subplots(figsize=(7, len(at)*0.42+2))
    norm = Normalize(vmin=0, vmax=top_t["neglog10p"].max())
    sm = ScalarMappable(cmap=cmap, norm=norm)
    for _, row in top_t.iterrows():
        ax.scatter(c2x[row["celltype"]], t2y[row["Term_clean"]], s=row["Count"]*20+30, c=[sm.to_rgba(row["neglog10p"])], edgecolors="#666", linewidths=0.5, zorder=3)
    ax.set_xticks(range(len(CT_ORDER))); ax.set_xticklabels([CT_DISPLAY[c] for c in CT_ORDER], fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(at))); ax.set_yticklabels(at, fontsize=9); ax.invert_yaxis()
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    for i in range(len(at)): ax.axhline(i, color="#f0f0f0", linewidth=0.4, zorder=1)
    for i in range(len(CT_ORDER)): ax.axvline(i, color="#f0f0f0", linewidth=0.4, zorder=1)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(left=False, bottom=False); ax.set_xlim(-0.5, len(CT_ORDER)-0.5)
    cb = plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.02, aspect=12)
    cb.set_label("-log/home/liyang/BioJiaheWang/scGTAC{10}", fontsize=9)
    sizes = [3, 10, 20]
    for sv in sizes: ax.scatter([],[],s=sv*20+30,c="gray",alpha=0.4,edgecolors="#666",linewidths=0.5,label=str(sv))
    ax.legend(title="Count",loc="upper right",frameon=True,fontsize=8,title_fontsize=9,labelspacing=2.0,handletextpad=1.5,borderpad=1.5,fancybox=True,edgecolor="#ddd",bbox_to_anchor=(1.28,1.0))
    plt.tight_layout()
    fig_e.savefig(OUT+"/_t"+pn+".png", dpi=300, bbox_inches="tight"); plt.close()
    print("[ok] "+lib)
feat = mpimg.imread(OUT+"/_tf.png")
go = mpimg.imread(OUT+"/_tb.png")
kegg = mpimg.imread(OUT+"/_tc.png")
fig = plt.figure(figsize=(16, 18))
gs = gridspec.GridSpec(2, 2, height_ratios=[0.8, 1], wspace=0.05, hspace=0.08)
ax_a = fig.add_subplot(gs[0, :]); ax_a.imshow(feat); ax_a.axis("off"); ax_a.text(0.01,0.98,"a",transform=ax_a.transAxes,fontsize=18,fontweight="bold",va="top")
ax_b = fig.add_subplot(gs[1, 0]); ax_b.imshow(go); ax_b.axis("off"); ax_b.text(0.01,0.98,"b",transform=ax_b.transAxes,fontsize=18,fontweight="bold",va="top")
ax_c = fig.add_subplot(gs[1, 1]); ax_c.imshow(kegg); ax_c.axis("off"); ax_c.text(0.01,0.98,"c",transform=ax_c.transAxes,fontsize=18,fontweight="bold",va="top")
for ext in ["png","pdf","svg"]: fig.savefig(OUT+"/Figure4."+ext, dpi=300, bbox_inches="tight")
plt.close()
for f in ["_tf.png","_tb.png","_tc.png"]: os.remove(OUT+"/"+f) if os.path.exists(OUT+"/"+f) else None
print("done -> "+OUT+"/Figure4.png/.pdf/.svg")
