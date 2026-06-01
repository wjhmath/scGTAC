"""
run_scgpt.py —— scGPT baseline (遵守 common.py 契约)。

环境: bl_torch + pip install scgpt
需要预训练模型:
  1. 从 https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y
     下载 whole-human 模型文件夹(含 best_model.pt, vocab.json, args.json)
  2. 放到 baselines/scgpt_model/ 下

scGPT 是预训练 Transformer 基础模型: 加载预训练权重 -> 提细胞嵌入 -> 聚类。
不需要训练(零样本), 需要 GPU。

被 run.py 调用; 也可单独:
  python run_scgpt.py --data .../baron/baron.h5ad --out results/baselines/scgpt \
      --dataset baron --seed 1 --model_dir baselines/scgpt_model
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans

from common import load_dataset, save_outputs, set_seed

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repos", "scGPT")
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label_key", default=None)
    ap.add_argument("--model_dir", default=os.path.join(HERE, "scgpt_model"),
                    help="scGPT 预训练模型目录(含 best_model.pt, vocab.json, args.json)")
    ap.add_argument("--gene_col", default="index",
                    help="adata.var 中基因名所在列(默认 'index' 即 var.index)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=1200)
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # 检查模型目录
    for needed in ["best_model.pt", "vocab.json", "args.json"]:
        fp = os.path.join(args.model_dir, needed)
        if not os.path.isfile(fp):
            sys.exit(f"缺少 {fp}\n请下载 scGPT 预训练模型到 {args.model_dir}/\n"
                     f"下载地址: https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [device] {device}")

    # 导入 scGPT
    try:
        from scgpt.tasks.cell_emb import embed_data
    except ImportError as e:
        sys.exit(f"scGPT 导入失败: {e}\n"
                 f"请确认 repos/scGPT 已 clone,且 bl_torch 里有 torch/scanpy/anndata")

    adata, X, y, k = load_dataset(args.data, args.label_key)
    print(f"  [scGPT] cells={adata.n_obs} genes={adata.n_vars} k={k}")

    # scGPT 需要原始计数(非归一化)和基因名
    # embed_data 内部会做 gene-vocab 匹配
    print("  [scGPT] 提取细胞嵌入(零样本,不训练)...")
    adata_emb = embed_data(
        adata,
        model_dir=args.model_dir,
        gene_col=args.gene_col,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        use_fast_transformer=False,  # 不需要 flash-attention
        return_new_adata=False,
    )
    embeddings = adata_emb.obsm["X_scGPT"]
    print(f"  [scGPT] 嵌入维度: {embeddings.shape}")

    # 在嵌入上聚类
    km = KMeans(n_clusters=k, n_init=20, random_state=args.seed)
    y_pred = km.fit_predict(embeddings)

    save_outputs(args.out, args.dataset, "scgpt", args.seed, y_pred, y)


if __name__ == "__main__":
    main()
