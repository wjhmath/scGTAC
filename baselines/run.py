"""
run.py —— 一键入口。在 VSCode 里打开本文件, 点右上角绿色 ▷ 运行即可。
不需要命令行、不需要 sbatch。

它会:遍历下面 DATASETS 里所有数据集 × SEEDS, 跑 METHOD 指定的方法,
最后自动汇总成对比表 results/baselines/summary.csv。

== 改这里就够了 ==============================================
  METHOD : 跑哪个方法(先用 "leiden" 把流程跑通)
  SEEDS  : 几个随机种子(给 Fig2 误差棒)
  DATASETS : 你的 7 个数据集名字(按 data/ 下实际名字改)
=============================================================

VSCode 注意:运行前左下角选对 Python 解释器——
  跑 leiden -> 选 scagcr_env
  跑 desc   -> 选 bl_tf
  跑 sctag  -> 选 bl_torch
"""
import os
import sys
import subprocess

# ====================== 配置(改这里) ======================
METHOD   = "leiden"
SEEDS    = [1, 2, 3]
DATASETS = ["baron", "multiome", "muraro_pancreas", "zheng68k",
            "cellxgene_sample_small", "tabula_muris_kidney",
            "tabula_sapiens_ear_utricle"]
EXTRA_ARGS = ["--match_k"]      # leiden 专用;换别的方法时清空成 []
LABEL_KEY  = None               # 若自动找不到标签列, 填你的列名, 如 "celltype"
# ===========================================================

HERE = os.path.dirname(os.path.abspath(__file__))          # .../scAGCR/baselines
PROJ = os.path.dirname(HERE)                               # .../scAGCR
DATA_DIR = os.path.join(PROJ, "data")
OUT_BASE = os.path.join(PROJ, "results", "baselines")
OUT_DIR  = os.path.join(OUT_BASE, METHOD)


def find_h5ad(ds: str):
    """同时支持 data/<ds>/<ds>.h5ad 和 data/<ds>.h5ad 两种布局。"""
    for p in (os.path.join(DATA_DIR, ds, f"{ds}.h5ad"),
              os.path.join(DATA_DIR, f"{ds}.h5ad")):
        if os.path.isfile(p):
            return p
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    runner = os.path.join(HERE, f"run_{METHOD}.py")
    if not os.path.isfile(runner):
        sys.exit(f"找不到 {runner} —— 这个方法的 runner 还没写。")

    print(f"===== 方法={METHOD}  数据集={len(DATASETS)}  种子={SEEDS} =====")
    print(f"  Python: {sys.executable}\n")
    ok = fail = skip = 0
    for ds in DATASETS:
        h5ad = find_h5ad(ds)
        if h5ad is None:
            print(f"[跳过] 找不到 {ds} 的 .h5ad(在 {DATA_DIR} 下)"); skip += 1; continue
        for s in SEEDS:
            done = os.path.join(OUT_DIR, f"{ds}_seed{s}_metrics.json")
            if os.path.isfile(done):
                print(f"[跳过] 已完成 {ds} seed{s}"); skip += 1; continue
            cmd = [sys.executable, runner, "--data", h5ad, "--out", OUT_DIR,
                   "--dataset", ds, "--seed", str(s)] + EXTRA_ARGS
            if LABEL_KEY:
                cmd += ["--label_key", LABEL_KEY]
            print(f">>> {METHOD}  {ds}  seed{s}")
            r = subprocess.run(cmd)
            if r.returncode == 0:
                ok += 1
            else:
                print(f"[失败] {ds} seed{s}(returncode={r.returncode}),继续"); fail += 1

    print(f"\n===== 完成: 成功{ok} 失败{fail} 跳过{skip} =====")
    print("汇总中...")
    subprocess.run([sys.executable, os.path.join(HERE, "aggregate.py"),
                    "--root", OUT_BASE,
                    "--out", os.path.join(OUT_BASE, "summary.csv")])


if __name__ == "__main__":
    main()
