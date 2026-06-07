#!/usr/bin/env python3
"""
update_and_verify.py
1. 更新所有画图脚本中的数据集显示名 (DISP dict)
2. 重新生成所有图
3. 输出每张图对应的论文正文所需精确数值
用法: python update_and_verify.py --root /home/liyang/BioJiaheWang/scAGCR
"""
import os, re, glob, json, sys, csv
from collections import defaultdict
import statistics as st
import numpy as np

ROOT = sys.argv[sys.argv.index('--root')+1] if '--root' in sys.argv else '/home/liyang/BioJiaheWang/scAGCR'

# ========== 1. 统一命名映射 ==========
# 内部文件名 -> 论文展示名
DISP_NEW = {
    "muraro_pancreas": "Muraro",
    "baron":           "Baron",
    "GSE103354":       "Airway",
    "GSE103322":       "Puram",
    "Tonsil":          "Tonsil",
    "GSE150580_Mammary":"Mammary",
    "GSE119531":       "UUO kidney",
    "GSE194122_PBMC_Bench_1": "BMMC-B1",
    "multiome":        "Multiome",
    "GSE159115_ccRCC": "Kidney ccRCC",
    "GSE194122_PBMC_Test": "BMMC-test",
    "Goolam":          "Goolam",
    "Crohn":           "Crohn",
    "68kPBMC":         "68k PBMC",
    "GSE123516_labeled":"Intestine",
    # scalability 额外数据集
    "10X_PBMC":        "10X PBMC",
    "zheng68k":        "zheng68k",
}

LINEUP = ["muraro_pancreas","baron","GSE103354","GSE103322","Tonsil","GSE150580_Mammary",
          "GSE119531","GSE194122_PBMC_Bench_1","multiome","GSE159115_ccRCC",
          "GSE194122_PBMC_Test","Goolam","Crohn","68kPBMC","GSE123516_labeled"]
BASE = ["dec","scvi","scgnn","seurat","scdeepcluster","scdsc"]
BL = {"dec":"DEC","scvi":"scVI","scgnn":"scGNN","seurat":"Seurat","scdeepcluster":"scDeepCluster","scdsc":"scDSC"}
ALIAS = {"muraro":"muraro_pancreas"}
PATS = {"ARI":"'ARI'","NMI":"'NMI'","ACC":"'CA'"}
METHOD = "scGTAC"

# ========== 2. 更新画图脚本中的 DISP ==========
def update_disp_in_script(filepath):
    """替换脚本中的 DISP={...} 字典"""
    if not os.path.exists(filepath):
        print(f"  [跳过] {filepath} 不存在")
        return
    with open(filepath) as f:
        content = f.read()
    # 构建新的 DISP 字符串
    disp_items = []
    for k, v in DISP_NEW.items():
        disp_items.append(f'"{k}":"{v}"')
    new_disp = "DISP={" + ",\n      ".join(disp_items) + "}"
    # 用正则替换旧的 DISP={...}
    pattern = r'DISP\s*=\s*\{[^}]+\}'
    if re.search(pattern, content):
        content = re.sub(pattern, new_disp, content, count=1)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  [更新] {filepath}")
    else:
        print(f"  [无DISP] {filepath}")

print("=" * 70)
print("步骤 1: 更新画图脚本中的数据集显示名")
print("=" * 70)
for script in ['fig2_benchmark.py', 'fig7_robustness.py', 'fig8_scalability.py']:
    update_disp_in_script(f"{ROOT}/pipeline/figures/{script}")

# ========== 3. 读取实验数据 (与 fig2 完全相同的逻辑) ==========
print("\n" + "=" * 70)
print("步骤 2: 读取所有实验数据")
print("=" * 70)

sca = defaultdict(dict)
raw = defaultdict(lambda: defaultdict(list))
for lg in glob.glob(f"{ROOT}/results/scagcr_final/*/run_seed*.log"):
    ds = os.path.basename(os.path.dirname(lg))
    t = open(lg, errors='ignore').read().strip()
    if not t: continue
    for mk, p in PATS.items():
        m = re.search(rf"{p}:\s*(?:np\.float64\()?([0-9.]+)", t.split("\n")[-1])
        if m: raw[ds][mk].append(float(m.group(1)))
for ds, dd in raw.items():
    for mk, v in dd.items():
        if v: sca[ds][mk] = st.mean(v)

blraw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for f in glob.glob(f"{ROOT}/results/baselines/*/*_metrics.json") + \
         glob.glob(f"{ROOT}/results/baselines_chuli/*/*_metrics.json"):
    me = f.split("/")[-2]
    if me == "leiden": continue
    ds = re.sub(r"_seed\d+_metrics\.json$", "", os.path.basename(f))
    ds = ALIAS.get(ds, ds)
    try: d = json.load(open(f))
    except: continue
    for mk in ["ARI", "NMI", "ACC"]:
        if d.get(mk) is not None:
            blraw[ds][mk][me].append(float(d[mk]))
bl = defaultdict(lambda: defaultdict(dict))
for ds, dd in blraw.items():
    for mk, md in dd.items():
        for me, v in md.items():
            bl[ds][mk][me] = st.mean(v)

def bmean(me, mk):
    vs = [(sca.get(d, {}).get(mk) if me == METHOD else bl.get(d, {}).get(mk, {}).get(me))
          for d in LINEUP]
    vs = [v for v in vs if v is not None]
    return np.mean(vs) if vs else 0

print(f"  scGTAC: 找到 {len(sca)} 个数据集的结果")
print(f"  基线: 找到 {len(blraw)} 个数据集的结果")

# ========== 4. 图 2 输出 ==========
print("\n" + "=" * 70)
print("图 2: 主实验 Benchmark")
print("=" * 70)

metrics = ["ARI", "NMI", "ACC"]
methods_sorted = [METHOD] + sorted(BASE, key=lambda m: -bmean(m, "ARI"))

print(f"\n--- 热图数据 (ARI, 每个数据集) ---")
print(f"{'Dataset':<18} {'Display':<14}", end="")
for me in methods_sorted:
    label = METHOD if me == METHOD else BL[me]
    print(f" {label:>8}", end="")
print()
for ds in LINEUP:
    print(f"{ds:<18} {DISP_NEW.get(ds,'?'):<14}", end="")
    for me in methods_sorted:
        v = sca.get(ds, {}).get("ARI") if me == METHOD else bl.get(ds, {}).get("ARI", {}).get(me)
        print(f" {v:>8.2f}" if v is not None else f" {'—':>8}", end="")
    print()

print(f"\n--- 表 2: 均值 ---")
print(f"{'Method':<18} {'ARI':>8} {'NMI':>8} {'ACC':>8}")
print("-" * 44)
for me in methods_sorted:
    vals = {mk: bmean(me, mk) for mk in metrics}
    label = f"{METHOD} (ours)" if me == METHOD else BL[me]
    print(f"{label:<18} {vals['ARI']:>8.3f} {vals['NMI']:>8.3f} {vals['ACC']:>8.3f}")
print("-" * 44)

best_base = {mk: max(bmean(m, mk) for m in BASE) for mk in metrics}
our = {mk: bmean(METHOD, mk) for mk in metrics}
print(f"\n--- 增益 (正文/摘要用) ---")
for mk in metrics:
    bb_name = [BL[m] for m in BASE if bmean(m, mk) == best_base[mk]][0]
    print(f"  {mk}: +{(our[mk]-best_base[mk])*100:.1f}pp (scGTAC {our[mk]:.3f} vs {bb_name} {best_base[mk]:.3f})")

RANKS = defaultdict(lambda: defaultdict(int))
for ds in LINEUP:
    for mk in metrics:
        sc = {}
        s = sca.get(ds, {}).get(mk)
        if s is not None: sc[METHOD] = s
        for me in BASE:
            v = bl.get(ds, {}).get(mk, {}).get(me)
            if v is not None: sc[me] = v
        for r, (me2, _) in enumerate(sorted(sc.items(), key=lambda x: -x[1]), 1):
            RANKS[me2][r] += 1
print(f"\n--- Rank #1 ---")
print(f"  scGTAC: {RANKS[METHOD].get(1,0)} / 45")

# ========== 5. 图 3 输出 ==========
print("\n" + "=" * 70)
print("图 3: Muraro 生物学验证")
print("=" * 70)

muraro_seeds = sorted(glob.glob(f"{ROOT}/results/scagcr_final/muraro_pancreas/run_seed*.log"))
print(f"\n--- Muraro 各种子 (训练 log) ---")
for log in muraro_seeds:
    t = open(log, errors='ignore').read().strip()
    m = re.search(r"'ARI':\s*(?:np\.float64\()?([0-9.]+)", t.split("\n")[-1])
    seed = re.search(r'seed(\d+)', log)
    if m and seed:
        print(f"  seed {seed.group(1)}: ARI = {float(m.group(1)):.4f}")
print(f"  (UMAP 中显示的 ARI 来自 emb/scagcr.py 提取的 embedding, 可能不同)")

# ========== 6. 图 5 输出 ==========
print("\n" + "=" * 70)
print("图 5: 消融实验")
print("=" * 70)

ABL_DS = ["GSE119531", "GSE103322", "GSE159115_ccRCC"]
ABL_VARIANTS = {
    "full":             ["full", "baseline"],
    "wo_cl":            ["wo_cl", "no_cl"],
    "wo_feature_aug":   ["wo_feature_aug", "no_feature_aug"],
    "wo_edge_aug":      ["wo_edge_aug", "no_edge_aug"],
}
abl_results = {}
for vname, vdirs in ABL_VARIANTS.items():
    aris = []
    nmis = []
    accs = []
    for vdir in vdirs:
        for ds in ABL_DS:
            for log in glob.glob(f"{ROOT}/results/ablation/{vdir}/{ds}/run_seed*.log"):
                t = open(log, errors='ignore').read().strip()
                if not t: continue
                last = t.split("\n")[-1]
                ma = re.search(r"'ARI':\s*(?:np\.float64\()?([0-9.]+)", last)
                mn = re.search(r"'NMI':\s*([0-9.]+)", last)
                mc = re.search(r"'CA':\s*(?:np\.float64\()?([0-9.]+)", last)
                if ma: aris.append(float(ma.group(1)))
                if mn: nmis.append(float(mn.group(1)))
                if mc: accs.append(float(mc.group(1)))
    if aris:
        abl_results[vname] = {'ARI': np.mean(aris), 'NMI': np.mean(nmis), 'ACC': np.mean(accs), 'n': len(aris)}

if 'full' in abl_results:
    full = abl_results['full']
    print(f"  Full: ARI={full['ARI']:.4f}, NMI={full['NMI']:.4f}, ACC={full['ACC']:.4f} (n={full['n']})")
    for vname in ['wo_cl', 'wo_feature_aug', 'wo_edge_aug']:
        if vname in abl_results:
            v = abl_results[vname]
            print(f"  {vname}: ARI={v['ARI']:.4f}, Δ={full['ARI']-v['ARI']:.4f}; "
                  f"NMI={v['NMI']:.4f}, Δ={full['NMI']-v['NMI']:.4f}; "
                  f"ACC={v['ACC']:.4f}, Δ={full['ACC']-v['ACC']:.4f} (n={v['n']})")
        else:
            print(f"  {vname}: [未找到结果]")
else:
    print("  [缺] full model 结果未找到")

# ========== 7. 图 6 输出 ==========
print("\n" + "=" * 70)
print("图 6: 参数敏感性 (Baron)")
print("=" * 70)

for param in ['tau', 'lambda_cl', 'dropout']:
    vals = {}
    for d in glob.glob(f"{ROOT}/results/param_sweep/{param}_*"):
        val_match = re.search(rf'{param}_([\d.]+)', d)
        if val_match:
            val = val_match.group(1)
            aris, nmis, accs = [], [], []
            for log in glob.glob(f"{d}/baron/run_seed*.log") + glob.glob(f"{d}/*/run_seed*.log"):
                t = open(log, errors='ignore').read().strip()
                if not t: continue
                last = t.split("\n")[-1]
                ma = re.search(r"'ARI':\s*(?:np\.float64\()?([0-9.]+)", last)
                mn = re.search(r"'NMI':\s*([0-9.]+)", last)
                mc = re.search(r"'CA':\s*(?:np\.float64\()?([0-9.]+)", last)
                if ma: aris.append(float(ma.group(1)))
                if mn: nmis.append(float(mn.group(1)))
                if mc: accs.append(float(mc.group(1)))
            if aris:
                vals[val] = {'ARI': np.mean(aris), 'NMI': np.mean(nmis), 'ACC': np.mean(accs)}
    if vals:
        print(f"\n  {param}:")
        print(f"    {'value':<8} {'ARI':>8} {'NMI':>8} {'ACC':>8}")
        for v in sorted(vals.keys(), key=float):
            d = vals[v]
            print(f"    {v:<8} {d['ARI']:>8.4f} {d['NMI']:>8.4f} {d['ACC']:>8.4f}")

# ========== 8. 图 7 输出 ==========
print("\n" + "=" * 70)
print("图 7: 下采样鲁棒性")
print("=" * 70)

ds_csv = f"{ROOT}/results/robustness/downsample.csv"
if os.path.exists(ds_csv):
    ds_data = defaultdict(lambda: defaultdict(list))
    with open(ds_csv, errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ds_data[row['dataset']][row['ratio']].append(float(row['ARI']))
            except (ValueError, KeyError):
                continue
    for dataset in ['baron', 'GSE103322', 'GSE119531']:
        disp = DISP_NEW.get(dataset, dataset)
        print(f"\n  {disp} ({dataset}):")
        for ratio in sorted(ds_data.get(dataset, {}).keys(), key=lambda x: float(x)):
            aris = ds_data[dataset][ratio]
            print(f"    ratio={ratio}: ARI = {np.mean(aris):.4f} ± {np.std(aris):.4f} (n={len(aris)})")
else:
    print("  downsample.csv 不存在")

# ========== 9. 图 8 输出 ==========
print("\n" + "=" * 70)
print("图 8: 可扩展性")
print("=" * 70)

for csv_name, label in [('synthetic_timing.csv', '模拟数据'), ('timing.csv', '真实数据')]:
    csv_path = f"{ROOT}/results/scalability/{csv_name}"
    if os.path.exists(csv_path):
        print(f"\n  {label} ({csv_name}):")
        with open(csv_path, errors='ignore') as f:
            for i, line in enumerate(f):
                if i < 30:
                    print(f"    {line.strip()}")
    else:
        print(f"\n  {label}: {csv_path} 不存在")

# ========== 10. 总结 ==========
print("\n" + "=" * 70)
print("正文需要的关键数值汇总")
print("=" * 70)
print(f"""
Abstract / Introduction:
  mean ARI  = {our['ARI']:.2f} (精确 {our['ARI']:.3f})
  mean NMI  = {our['NMI']:.2f} (精确 {our['NMI']:.3f})
  mean ACC  = {our['ACC']:.2f} (精确 {our['ACC']:.3f})
  ARI gain  = +{(our['ARI']-best_base['ARI'])*100:.0f}pp
  NMI gain  = +{(our['NMI']-best_base['NMI'])*100:.0f}pp
  ACC gain  = +{(our['ACC']-best_base['ACC'])*100:.0f}pp
  Rank #1   = {RANKS[METHOD].get(1,0)}/45

Section 3.4 (实验设置):
  d = 256, d_c = 128 (论文标准化后)
  T = 200, T_pre = 20
  
Section 3.5 (总体性能):
  见上方表 2 数值

数据集展示名映射:
""")
for ds in LINEUP:
    print(f"  {ds:<30} → {DISP_NEW[ds]}")

print("\n完成。请用以上数值更新论文正文。")