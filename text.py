#!/usr/bin/env python3
"""
verify_figures.py — 逐图核对论文中的数值是否与实际实验输出一致
在服务器上运行: python verify_figures.py --root /home/liyang/BioJiaheWang/scAGCR
"""
import argparse, glob, json, os, re, sys
import numpy as np
from collections import defaultdict

def parse_log_last_line(logfile):
    """从 scAGCR log 最后一行提取 ARI/NMI/CA"""
    with open(logfile, errors='ignore') as f:
        lines = f.read().strip().splitlines()
    if not lines:
        return None
    last = lines[-1]
    m = re.search(r"'CA':\s*(?:np\.float64\()?([\d.]+)\)?,\s*'NMI':\s*([\d.]+),\s*'ARI':\s*([\d.]+)", last)
    if m:
        return {'ACC': float(m.group(1)), 'NMI': float(m.group(2)), 'ARI': float(m.group(3))}
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/home/liyang/BioJiaheWang/scAGCR')
    args = ap.parse_args()
    ROOT = args.root

    print("=" * 70)
    print("图 2 验证: scGTAC (scAGCR) 各数据集 ARI + 基线均值")
    print("=" * 70)

    # --- scAGCR 结果 ---
    DATASETS_15 = [
        'muraro_pancreas', 'baron', 'GSE103354', 'GSE103322', 'Tonsil',
        'GSE150580_Mammary', 'GSE119531', 'GSE194122_PBMC_Bench_1',
        'multiome', 'GSE159115_ccRCC', 'GSE194122_PBMC_Test',
        'Goolam', 'Crohn', '68kPBMC', 'GSE123516_labeled'
    ]

    # 图 2a 热图中读到的值 (左→右, 2位小数)
    HEATMAP_ARI = {
        'muraro_pancreas': 0.93, 'baron': 0.78, 'GSE103354': 0.77,
        'GSE103322': 0.73, 'Tonsil': 0.70, 'GSE150580_Mammary': 0.67,
        'GSE119531': 0.63, 'GSE194122_PBMC_Bench_1': 0.63,
        'multiome': 0.58, 'GSE159115_ccRCC': 0.56,
        'GSE194122_PBMC_Test': 0.54, 'Goolam': 0.49,
        'Crohn': 0.47, '68kPBMC': 0.45, 'GSE123516_labeled': 0.43
    }

    scagcr_results = {}
    for ds in DATASETS_15:
        # 尝试多种路径格式
        patterns = [
            f'{ROOT}/results/scagcr_final/{ds}/run_seed1.log',
            f'{ROOT}/results/scagcr_final/{ds}/run_seed*.log',
        ]
        found = False
        for pat in patterns:
            logs = glob.glob(pat)
            if logs:
                res = parse_log_last_line(logs[0])
                if res:
                    scagcr_results[ds] = res
                    found = True
                    break
        if not found:
            print(f"  [缺] {ds}: 找不到结果文件")

    print(f"\n{'数据集':<30} {'实际ARI':>8} {'热图ARI':>8} {'差值':>8} {'一致?':>6}")
    print("-" * 70)
    ari_list = []
    for ds in DATASETS_15:
        actual = scagcr_results.get(ds, {}).get('ARI', None)
        heatmap = HEATMAP_ARI.get(ds, None)
        if actual is not None:
            ari_list.append(actual)
            diff = abs(actual - heatmap) if heatmap else None
            ok = '✓' if (diff is not None and diff < 0.015) else '✗'
            print(f"  {ds:<28} {actual:>8.4f} {heatmap:>8.2f} {diff:>8.4f} {ok:>6}")
        else:
            print(f"  {ds:<28} {'N/A':>8} {heatmap:>8.2f} {'':>8} {'':>6}")

    if ari_list:
        mean_ari = np.mean(ari_list)
        print(f"\n  15数据集平均 ARI = {mean_ari:.4f}")
        print(f"  论文表 2 写的 = 0.610")
        print(f"  图 2b 标注的 = +15% (相对 Seurat 0.459)")
        print(f"  实际增益 = +{(mean_ari - 0.459)*100:.1f}pp")

    # --- 基线结果 ---
    print("\n" + "=" * 70)
    print("基线方法均值")
    print("=" * 70)

    BASELINES = ['seurat', 'scdeepcluster', 'scvi', 'scgnn', 'dec', 'scdsc']
    TABLE2_MEANS = {
        'seurat': {'ARI': 0.459, 'NMI': 0.660, 'ACC': 0.583},
        'scdeepcluster': {'ARI': 0.456, 'NMI': 0.639, 'ACC': 0.583},
        'scvi': {'ARI': 0.455, 'NMI': 0.662, 'ACC': 0.581},
        'scgnn': {'ARI': 0.418, 'NMI': 0.578, 'ACC': 0.570},
        'dec': {'ARI': 0.400, 'NMI': 0.569, 'ACC': 0.540},
        'scdsc': {'ARI': 0.207, 'NMI': 0.352, 'ACC': 0.385},
    }

    for method in BASELINES:
        jsons = glob.glob(f'{ROOT}/results/baselines/{method}/*_metrics.json')
        if not jsons:
            # 尝试其他路径
            jsons = glob.glob(f'{ROOT}/results/baselines_chuli/{method}/*_metrics.json')
            jsons += glob.glob(f'{ROOT}/results/baselines/{method}/**/*_metrics.json', recursive=True)

        if jsons:
            all_metrics = defaultdict(list)
            for fp in jsons:
                try:
                    d = json.load(open(fp))
                    for k in ['ARI', 'NMI', 'ACC']:
                        if k in d:
                            all_metrics[k].append(d[k])
                except:
                    pass
            if all_metrics:
                actual_mean = {k: np.mean(v) for k, v in all_metrics.items()}
                table_mean = TABLE2_MEANS.get(method, {})
                print(f"\n  {method}:")
                for k in ['ARI', 'NMI', 'ACC']:
                    a = actual_mean.get(k, None)
                    t = table_mean.get(k, None)
                    if a is not None and t is not None:
                        ok = '✓' if abs(a - t) < 0.01 else '✗'
                        print(f"    {k}: 实际={a:.4f}, 表2={t:.3f}, 差={abs(a-t):.4f} {ok}")
            else:
                print(f"\n  {method}: 文件存在但无法解析")
        else:
            print(f"\n  {method}: 找不到结果文件")

    # --- 图 3 验证: Muraro UMAP ARI ---
    print("\n" + "=" * 70)
    print("图 3 验证: Muraro UMAP 中显示的 ARI = 0.887")
    print("=" * 70)
    muraro_seeds = glob.glob(f'{ROOT}/results/scagcr_final/muraro_pancreas/run_seed*.log')
    for log in sorted(muraro_seeds):
        res = parse_log_last_line(log)
        if res:
            seed = re.search(r'seed(\d+)', log)
            s = seed.group(1) if seed else '?'
            print(f"  seed {s}: ARI={res['ARI']:.4f}, NMI={res['NMI']:.4f}, ACC={res['ACC']:.4f}")
    print(f"  图 3a 标注的 ARI = 0.887")
    print(f"  注意: 三种子中没有 0.887, 可能来自不同运行或不同 epoch")

    # --- 图 5 验证: 消融实验 ---
    print("\n" + "=" * 70)
    print("图 5 验证: 消融实验")
    print("=" * 70)

    ABLATION_VARIANTS = ['full', 'wo_cl', 'no_feature_aug', 'no_edge_aug']
    ABLATION_DS = ['GSE119531', 'GSE103322', 'GSE159115_ccRCC']

    # 图 5 中读到的 Δ ARI
    FIG5_DELTA = {'wo_cl': 0.10, 'no_feature_aug': 0.023, 'no_edge_aug': 0.015}

    ablation_results = defaultdict(list)
    for var in ABLATION_VARIANTS:
        for ds in ABLATION_DS:
            patterns = [
                f'{ROOT}/results/ablation/{var}/{ds}/run_seed*.log',
                f'{ROOT}/results/ablation/{var}/{ds}/*.log',
            ]
            for pat in patterns:
                for log in glob.glob(pat):
                    res = parse_log_last_line(log)
                    if res:
                        ablation_results[var].append(res['ARI'])

    if ablation_results.get('full'):
        full_mean = np.mean(ablation_results['full'])
        print(f"  Full model mean ARI = {full_mean:.4f}")
        for var in ['wo_cl', 'no_feature_aug', 'no_edge_aug']:
            if ablation_results.get(var):
                var_mean = np.mean(ablation_results[var])
                delta = full_mean - var_mean
                fig_delta = FIG5_DELTA.get(var, None)
                ok = '✓' if (fig_delta and abs(delta - fig_delta) < 0.01) else '?'
                print(f"  w/o {var:<20}: mean ARI={var_mean:.4f}, Δ={delta:.4f}, 图5 Δ={fig_delta} {ok}")
            else:
                print(f"  w/o {var:<20}: 找不到结果")
    else:
        print("  [缺] full model 结果未找到")

    # --- 图 6 验证: 参数敏感性 ---
    print("\n" + "=" * 70)
    print("图 6 验证: 参数敏感性 (Baron)")
    print("=" * 70)

    for param in ['tau', 'lambda_cl', 'dropout']:
        sweep_dir = f'{ROOT}/results/param_sweep'
        vals = {}
        for d in glob.glob(f'{sweep_dir}/{param}_*'):
            val_match = re.search(rf'{param}_([\d.]+)', d)
            if val_match:
                val = val_match.group(1)
                logs = glob.glob(f'{d}/baron/run_seed*.log') + glob.glob(f'{d}/*/run_seed*.log')
                aris = []
                for log in logs:
                    res = parse_log_last_line(log)
                    if res:
                        aris.append(res['ARI'])
                if aris:
                    vals[val] = np.mean(aris)
        if vals:
            print(f"\n  {param}:")
            for v in sorted(vals.keys(), key=float):
                print(f"    {param}={v}: mean ARI = {vals[v]:.4f}")
        else:
            print(f"\n  {param}: 找不到扫描结果")

    # --- 图 7 验证: 下采样 ---
    print("\n" + "=" * 70)
    print("图 7 验证: 下采样鲁棒性")
    print("=" * 70)

    ds_csv = f'{ROOT}/results/robustness/downsample.csv'
    if os.path.exists(ds_csv):
        import csv
        with open(ds_csv) as f:
            reader = csv.DictReader(f)
            ds_data = defaultdict(lambda: defaultdict(list))
            for row in reader:
                ds_data[row['dataset']][row['ratio']].append(float(row['ARI']))
        for dataset in ['baron', 'GSE103322', 'GSE119531']:
            print(f"\n  {dataset}:")
            for ratio in sorted(ds_data.get(dataset, {}).keys(), key=float):
                aris = ds_data[dataset][ratio]
                print(f"    ratio={ratio}: ARI = {np.mean(aris):.4f} (n={len(aris)})")
    else:
        # 尝试从目录结构读取
        for dataset in ['baron', 'GSE103322', 'GSE119531']:
            print(f"\n  {dataset}:")
            for ratio in ['0.1', '0.3', '0.5', '0.7', '0.9', '1.0']:
                logs = glob.glob(f'{ROOT}/results/robustness/{dataset}/*ratio{ratio}*/run_seed*.log')
                if not logs:
                    logs = glob.glob(f'{ROOT}/results/robustness/*{ratio}*/{dataset}/*.log')
                aris = []
                for log in logs:
                    res = parse_log_last_line(log)
                    if res:
                        aris.append(res['ARI'])
                if aris:
                    print(f"    ratio={ratio}: ARI = {np.mean(aris):.4f} (n={len(aris)})")

    # --- 图 8 验证: 可扩展性 ---
    print("\n" + "=" * 70)
    print("图 8 验证: 可扩展性 (训练时间)")
    print("=" * 70)

    for csv_name, label in [('synthetic_timing.csv', '模拟数据'), ('timing.csv', '真实数据')]:
        csv_path = f'{ROOT}/results/scalability/{csv_name}'
        if os.path.exists(csv_path):
            print(f"\n  {label} ({csv_name}):")
            with open(csv_path) as f:
                for line in f:
                    print(f"    {line.strip()}")
        else:
            print(f"\n  {label}: {csv_path} 不存在")

    print("\n" + "=" * 70)
    print("验证完成。请逐项检查标 ✗ 的条目。")
    print("=" * 70)

if __name__ == '__main__':
    main()