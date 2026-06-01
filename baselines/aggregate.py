"""
aggregate.py —— 扫描 results/baselines/ 下所有 *_metrics.json,
汇总成 (dataset x method) 的 mean±std 对比表, 存成 CSV。
这张表就是 Fig 2 的数据源。

用法:
  python aggregate.py --root results/baselines --out results/baselines/summary.csv
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/baselines")
    ap.add_argument("--out", default="results/baselines/summary.csv")
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.root, "**", "*_metrics.json"), recursive=True)
    if not files:
        print(f"没找到任何 metrics.json (在 {args.root} 下)"); return

    rows = []
    for fp in files:
        try:
            rows.append(json.load(open(fp)))
        except Exception as e:
            print(f"跳过 {fp}: {e}")
    df = pd.DataFrame(rows)

    # 按 (dataset, method) across seeds 求 mean/std
    out = []
    for (ds, me), g in df.groupby(["dataset", "method"]):
        rec = {"dataset": ds, "method": me, "n_seeds": len(g)}
        for m in ["ACC", "NMI", "ARI"]:
            rec[f"{m}_mean"] = round(float(g[m].mean()), 4)
            rec[f"{m}_std"] = round(float(g[m].std(ddof=0)), 4)
        out.append(rec)
    summary = pd.DataFrame(out).sort_values(["dataset", "ARI_mean"],
                                            ascending=[True, False])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summary.to_csv(args.out, index=False)

    # 打印一个好看的透视(ARI 为例)
    piv = summary.pivot(index="method", columns="dataset", values="ARI_mean")
    print("\n=== ARI mean (method x dataset) ===")
    print(piv.round(3).to_string())
    print(f"\n完整表已存: {args.out}")
    print("把 scAGCR 自己的 ACC/NMI/ARI 追加进这张表, 即为 Fig 2 的数据源。")


if __name__ == "__main__":
    main()
