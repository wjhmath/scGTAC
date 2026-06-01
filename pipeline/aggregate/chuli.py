"""跑完后执行：python aggregate_chuli.py"""
import json, glob, os, csv

OUT = "results/baselines_chuli"
rows = []
for f in sorted(glob.glob(os.path.join(OUT, "*", "*_metrics.json"))):
    try:
        rows.append(json.load(open(f)))
    except: pass

if not rows:
    print("暂无结果"); exit()

# 写 CSV
summary = os.path.join(OUT, "summary_chuli.csv")
keys = ["dataset","method","seed","ACC","NMI","ARI"]
with open(summary, "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    w.writerows(sorted(rows, key=lambda x: (x.get("dataset",""), -x.get("ARI",0))))

# 打印每个数据集的最佳方法
from collections import defaultdict
by_ds = defaultdict(list)
for r in rows:
    by_ds[r["dataset"]].append(r)

print(f"{'数据集':<28} {'最佳方法':<16} {'ARI':>7} {'NMI':>7} {'方法数':>6}")
print("-"*70)
for ds in sorted(by_ds):
    best = max(by_ds[ds], key=lambda x: x.get("ARI",0))
    print(f"{ds:<28} {best['method']:<16} {best['ARI']:>7.3f} {best['NMI']:>7.3f} {len(by_ds[ds]):>6}")

print(f"\n共 {len(rows)} 条结果，{len(by_ds)} 个数据集")
print(f"CSV: {summary}")
