import scanpy as sc, glob, os, subprocess, re, sys

CHULI = "/home/liyang/BioJiaheWang/RARECELL/data/chuli"
MAIN  = "scagcr/main.py"
EPOCHS = 5
CANDS = ["cell_type","cell_type_label","CellType","celltype","label","labels","Group"]

results = []
files = sorted(glob.glob(f"{CHULI}/*.h5ad"))
print(f"共找到 {len(files)} 个数据集，开始扫描 (epochs={EPOCHS})...\n")

for i, f in enumerate(files):
    name = os.path.basename(f).replace('.h5ad','')
    try:
        a = sc.read_h5ad(f, backed='r')
    except Exception as e:
        print(f"[{i+1}/{len(files)}] {name}: 读取失败 {e}"); continue
    col = next((c for c in CANDS if c in a.obs.columns), None)
    if col is None:
        print(f"[{i+1}/{len(files)}] {name}: 无标签列，跳过"); continue
    k = a.obs[col].astype(str).nunique()
    n_cells = a.n_obs
    if k < 2 or k > 50:
        print(f"[{i+1}/{len(files)}] {name}: k={k} 不适合聚类，跳过"); continue

    ckpt = f"/tmp/scan_{name}.pt"
    cmd = ["python", MAIN, "--data_path", f, "--n_clusters", str(k),
           "--epochs", str(EPOCHS), "--save_model_path", ckpt]
    print(f"[{i+1}/{len(files)}] {name} (cells={n_cells}, k={k})...", end=" ", flush=True)
    try:
        ret = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        last = ret.stdout.strip().split('\n')[-1] if ret.stdout.strip() else ''
        m_ca = re.search(r"'CA':\s*(?:np\.float64\()?([\d.]+)", last)
        m_nmi = re.search(r"'NMI':\s*([\d.]+)", last)
        m_ari = re.search(r"'ARI':\s*([\d.]+)", last)
        if m_ari:
            ca, nmi, ari = float(m_ca.group(1)), float(m_nmi.group(1)), float(m_ari.group(1))
            print(f"ACC={ca:.3f} NMI={nmi:.3f} ARI={ari:.3f}")
            results.append({"name": name, "cells": n_cells, "k": k, "ACC": ca, "NMI": nmi, "ARI": ari})
        else:
            err = ret.stderr.strip().split('\n')[-1] if ret.stderr.strip() else '未知错误'
            print(f"失败: {err[:80]}")
    except subprocess.TimeoutExpired:
        print("超时(>10min)，跳过")
    except Exception as e:
        print(f"异常: {e}")

# 按 ARI 排序输出
results.sort(key=lambda x: -x['ARI'])
print(f"\n{'='*85}")
print(f"{'排名':>4} {'数据集':<28} {'细胞数':>7} {'k':>4} {'ACC':>7} {'NMI':>7} {'ARI':>7}")
print(f"{'-'*85}")
for i, r in enumerate(results):
    flag = "★" if r['ARI'] >= 0.6 else "  "
    print(f"{flag}{i+1:>3} {r['name']:<28} {r['cells']:>7} {r['k']:>4} {r['ACC']:>7.3f} {r['NMI']:>7.3f} {r['ARI']:>7.3f}")
print(f"{'='*85}")
print(f"共 {len(results)} 个成功，其中 ARI>=0.6 的有 {sum(1 for r in results if r['ARI']>=0.6)} 个")

import json
json.dump(results, open("results/chuli_scan_results.json","w"), indent=2, ensure_ascii=False)
print(f"\n完整结果已保存: results/chuli_scan_results.json")
