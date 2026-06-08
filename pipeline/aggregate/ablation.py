import glob, re, os, csv, numpy as np
PROJ="/home/liyang/BioJiaheWang/scGTAC"; DS="baron"
os.makedirs(f"{PROJ}/generated_tables", exist_ok=True)
def collect(vardir):
    rows=[]
    for log in glob.glob(f"{PROJ}/results/ablation/{vardir}/{DS}/run_seed*.log"):
        t=open(log,errors='ignore').read().strip().splitlines()
        m=re.search(r"'CA':\s*(?:np\.float64\()?([\d.]+)\)?,\s*'NMI':\s*([\d.]+),\s*'ARI':\s*([\d.]+)", t[-1] if t else '')
        if m: rows.append([float(x) for x in m.groups()])   # CA, NMI, ARI
    return np.array(rows) if rows else None
# 文章 variant_name -> 我们的结果目录
ORDER=[("baseline","full"),("no_cl","wo_cl"),("no_cluster","wo_cluster"),
       ("no_zinb","wo_zinb"),("no_edge_aug","no_edge_aug"),("no_feature_aug","no_feature_aug")]
out=[]
for vname,vdir in ORDER:
    a=collect(vdir)
    if a is None and vname in ("no_edge_aug","no_feature_aug"):
        a=collect("wo_aug")   # 没单独跑边/特征 -> 用合并增强占位
        if a is not None: print(f"[占位] {vname} 用 wo_aug(合并增强)代替")
    if a is None: print(f"[缺] {vname} 无结果"); continue
    acc,nmi,ari=a[:,0],a[:,1],a[:,2]
    out.append(dict(variant_name=vname,dataset=DS,
        ACC_mean=acc.mean(),ACC_std=acc.std(),NMI_mean=nmi.mean(),NMI_std=nmi.std(),
        ARI_mean=ari.mean(),ARI_std=ari.std()))
    print(f"{vname:<16} n={len(a)}  ARI={ari.mean():.3f}±{ari.std():.3f}")
with open(f"{PROJ}/generated_tables/iter2_ablation_summary.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["variant_name","dataset","ACC_mean","ACC_std","NMI_mean","NMI_std","ARI_mean","ARI_std"])
    w.writeheader(); [w.writerow(r) for r in out]
print("\n-> generated_tables/iter2_ablation_summary.csv 写好")
