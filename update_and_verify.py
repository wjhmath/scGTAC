#!/usr/bin/env python3
import os,re,glob,json,sys,csv
from collections import defaultdict
import statistics as st
import numpy as np

ROOT=sys.argv[sys.argv.index('--root')+1] if '--root' in sys.argv else '/home/liyang/BioJiaheWang/scAGCR'
DISP_NEW={"muraro_pancreas":"Muraro","baron":"Baron","GSE103354":"Airway","GSE103322":"Puram","Tonsil":"Tonsil","GSE150580_Mammary":"Mammary","GSE119531":"UUO kidney","GSE194122_PBMC_Bench_1":"BMMC-B1","multiome":"Multiome","GSE159115_ccRCC":"Kidney ccRCC","GSE194122_PBMC_Test":"BMMC-test","Goolam":"Goolam","Crohn":"Crohn","68kPBMC":"68k PBMC","GSE123516_labeled":"Intestine","10X_PBMC":"10X PBMC","zheng68k":"zheng68k"}
LINEUP=["muraro_pancreas","baron","GSE103354","GSE103322","Tonsil","GSE150580_Mammary","GSE119531","GSE194122_PBMC_Bench_1","multiome","GSE159115_ccRCC","GSE194122_PBMC_Test","Goolam","Crohn","68kPBMC","GSE123516_labeled"]
BASE=["dec","scvi","scgnn","seurat","scdeepcluster","scdsc"]
BL={"dec":"DEC","scvi":"scVI","scgnn":"scGNN","seurat":"Seurat","scdeepcluster":"scDeepCluster","scdsc":"scDSC"}
ALIAS={"muraro":"muraro_pancreas"}
PATS={"ARI":"'ARI'","NMI":"'NMI'","ACC":"'CA'"}
METHOD="scGTAC"

sca=defaultdict(dict);raw=defaultdict(lambda:defaultdict(list))
for lg in glob.glob(f"{ROOT}/results/scagcr_final/*/run_seed*.log"):
    ds=os.path.basename(os.path.dirname(lg));t=open(lg,errors='ignore').read().strip()
    if not t:continue
    for mk,p in PATS.items():
        m=re.search(rf"{p}:\s*(?:np\.float64\()?([0-9.]+)",t.split("\n")[-1])
        if m:raw[ds][mk].append(float(m.group(1)))
for ds,dd in raw.items():
    for mk,v in dd.items():
        if v:sca[ds][mk]=st.mean(v)
blraw=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
for f in glob.glob(f"{ROOT}/results/baselines/*/*_metrics.json")+glob.glob(f"{ROOT}/results/baselines_chuli/*/*_metrics.json"):
    me=f.split("/")[-2]
    if me=="leiden":continue
    ds=re.sub(r"_seed\d+_metrics\.json$","",os.path.basename(f));ds=ALIAS.get(ds,ds)
    try:d=json.load(open(f))
    except:continue
    for mk in["ARI","NMI","ACC"]:
        if d.get(mk) is not None:blraw[ds][mk][me].append(float(d[mk]))
bl=defaultdict(lambda:defaultdict(dict))
for ds,dd in blraw.items():
    for mk,md in dd.items():
        for me,v in md.items():bl[ds][mk][me]=st.mean(v)
def bmean(me,mk):
    vs=[(sca.get(d,{}).get(mk) if me==METHOD else bl.get(d,{}).get(mk,{}).get(me)) for d in LINEUP]
    vs=[v for v in vs if v is not None];return np.mean(vs) if vs else 0
metrics=["ARI","NMI","ACC"]
methods_sorted=[METHOD]+sorted(BASE,key=lambda m:-bmean(m,"ARI"))

print("="*70)
print("图 2: 热图 ARI (每个数据集)")
print("="*70)
print(f"{'Display':<14}",end="")
for me in methods_sorted:
    label=METHOD if me==METHOD else BL[me]
    print(f" {label:>10}",end="")
print()
for ds in LINEUP:
    print(f"{DISP_NEW.get(ds,'?'):<14}",end="")
    for me in methods_sorted:
        v=sca.get(ds,{}).get("ARI") if me==METHOD else bl.get(ds,{}).get("ARI",{}).get(me)
        print(f" {v:>10.3f}" if v is not None else f" {'--':>10}",end="")
    print()

print(f"\n{'='*70}")
print("表 2: 均值")
print("="*70)
print(f"{'Method':<18} {'ARI':>8} {'NMI':>8} {'ACC':>8}")
print("-"*44)
for me in methods_sorted:
    vals={mk:bmean(me,mk) for mk in metrics}
    label=f"{METHOD} (ours)" if me==METHOD else BL[me]
    print(f"{label:<18} {vals['ARI']:>8.3f} {vals['NMI']:>8.3f} {vals['ACC']:>8.3f}")

best_base={mk:max(bmean(m,mk) for m in BASE) for mk in metrics}
our={mk:bmean(METHOD,mk) for mk in metrics}
print(f"\n--- 增益 ---")
for mk in metrics:
    print(f"  {mk}: +{(our[mk]-best_base[mk])*100:.1f}pp (scGTAC {our[mk]:.3f} vs best {best_base[mk]:.3f})")
RANKS=defaultdict(lambda:defaultdict(int))
for ds in LINEUP:
    for mk in metrics:
        sc={};s=sca.get(ds,{}).get(mk)
        if s is not None:sc[METHOD]=s
        for me in BASE:
            v=bl.get(ds,{}).get(mk,{}).get(me)
            if v is not None:sc[me]=v
        for r,(me2,_) in enumerate(sorted(sc.items(),key=lambda x:-x[1]),1):RANKS[me2][r]+=1
print(f"  Rank #1: {RANKS[METHOD].get(1,0)}/45")

print(f"\n{'='*70}")
print("图 3: Muraro 各种子")
print("="*70)
for log in sorted(glob.glob(f"{ROOT}/results/scagcr_final/muraro_pancreas/run_seed*.log")):
    t=open(log,errors='ignore').read().strip()
    m=re.search(r"'ARI':\s*(?:np\.float64\()?([0-9.]+)",t.split("\n")[-1])
    seed=re.search(r'seed(\d+)',log)
    if m and seed:print(f"  seed {seed.group(1)}: ARI={float(m.group(1)):.4f}")

print(f"\n{'='*70}")
print("图 5: 消融")
print("="*70)
ABL_DS=["GSE119531","GSE103322","GSE159115_ccRCC"]
for vname,vdirs in [("full",["full","baseline"]),("wo_cl",["wo_cl","no_cl"]),("wo_feature_aug",["wo_feature_aug","no_feature_aug"]),("wo_edge_aug",["wo_edge_aug","no_edge_aug"])]:
    aris=[];nmis=[];accs=[]
    for vdir in vdirs:
        for ds in ABL_DS:
            for log in glob.glob(f"{ROOT}/results/ablation/{vdir}/{ds}/run_seed*.log"):
                t=open(log,errors='ignore').read().strip()
                if not t:continue
                last=t.split("\n")[-1]
                ma=re.search(r"'ARI':\s*(?:np\.float64\()?([0-9.]+)",last)
                mn=re.search(r"'NMI':\s*([0-9.]+)",last)
                mc=re.search(r"'CA':\s*(?:np\.float64\()?([0-9.]+)",last)
                if ma:aris.append(float(ma.group(1)))
                if mn:nmis.append(float(mn.group(1)))
                if mc:accs.append(float(mc.group(1)))
    if aris:print(f"  {vname}: ARI={np.mean(aris):.4f} NMI={np.mean(nmis):.4f} ACC={np.mean(accs):.4f} (n={len(aris)})")
    else:print(f"  {vname}: [未找到]")

print(f"\n{'='*70}")
print("图 6: 参数敏感性")
print("="*70)
for param in['tau','lambda_cl','dropout']:
    vals={}
    for d in glob.glob(f"{ROOT}/results/param_sweep/{param}_*"):
        vm=re.search(rf'{param}_([\d.]+)',d)
        if vm:
            val=vm.group(1);aris=[]
            for log in glob.glob(f"{d}/baron/run_seed*.log")+glob.glob(f"{d}/*/run_seed*.log"):
                t=open(log,errors='ignore').read().strip()
                if not t:continue
                ma=re.search(r"'ARI':\s*(?:np\.float64\()?([0-9.]+)",t.split("\n")[-1])
                if ma:aris.append(float(ma.group(1)))
            if aris:vals[val]=np.mean(aris)
    if vals:
        print(f"  {param}: ",end="")
        for v in sorted(vals.keys(),key=float):print(f"{v}={vals[v]:.3f} ",end="")
        print()

print(f"\n{'='*70}")
print("图 7: 下采样")
print("="*70)
ds_csv=f"{ROOT}/results/robustness/downsample.csv"
if os.path.exists(ds_csv):
    dd=defaultdict(lambda:defaultdict(list))
    with open(ds_csv,errors='ignore') as f:
        reader=csv.DictReader(f)
        for row in reader:
            try:dd[row['dataset']][row['ratio']].append(float(row['ARI']))
            except:continue
    for dataset in['baron','GSE103322','GSE119531']:
        print(f"  {DISP_NEW.get(dataset,dataset)}: ",end="")
        for r in sorted(dd.get(dataset,{}).keys(),key=lambda x:float(x)):
            print(f"{r}={np.mean(dd[dataset][r]):.3f} ",end="")
        print()

print(f"\n{'='*70}")
print("汇总: 正文关键数值")
print("="*70)
print(f"  mean ARI = {our['ARI']:.2f}, mean NMI = {our['NMI']:.2f}, mean ACC = {our['ACC']:.2f}")
print(f"  ARI gain = +{(our['ARI']-best_base['ARI'])*100:.0f}pp")
print(f"  Rank #1 = {RANKS[METHOD].get(1,0)}/45")
print(f"\n  数据集命名:")
for ds in LINEUP:print(f"    {ds:<30} -> {DISP_NEW[ds]}")
