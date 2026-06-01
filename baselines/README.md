# Baseline 对比实验(VSCode 一键运行版)

把整个 `baselines/` 放到 `/home/liyang/BioJiaheWang/scAGCR/baselines/`,
然后在 VSCode 里打开 **`run.py`**,左下角选好 Python 解释器,点右上角绿色 ▷ 运行即可。
不需要命令行、不需要 sbatch。

## 一键运行步骤

1. 把 `baselines/` 整个目录传到 `…/scAGCR/baselines/`。
2. VSCode 打开 `run.py`。
3. 左下角选解释器:跑 leiden 选 `scagcr_env`(已装好,无需建环境)。
4. 点绿色 ▷ 运行。它会自动:遍历 7 个数据集 × 3 个种子 → 跑方法 → 汇总成
   `results/baselines/summary.csv`。

## 只改 run.py 顶部几行

```python
METHOD   = "leiden"      # 先用 leiden 跑通; 之后改 desc / sctag / ...
SEEDS    = [1, 2, 3]
DATASETS = [...]         # 按你 data/ 下实际名字改
EXTRA_ARGS = ["--match_k"]   # leiden 专用; 换别的方法清空成 []
LABEL_KEY  = None        # 自动找不到标签列时, 填列名如 "celltype"
```

数据路径两种布局都支持:`data/<ds>/<ds>.h5ad` 或 `data/<ds>.h5ad`,自动识别。

## 跑别的方法(以后)

每个方法一个 `run_<method>.py`,遵守同一套输入输出契约。换方法只需:
1. VSCode 左下角把解释器换成对应环境(desc→`bl_tf`,sctag→`bl_torch`);
2. `run.py` 顶部把 `METHOD` 改成对应名字、`EXTRA_ARGS` 清空;
3. 再点运行。

建新环境(登录节点有网时跑一次):`bash setup_baseline_envs.sh`

## 统一契约(为什么结果可信)

所有方法只负责吐出每个细胞的簇标号,ACC/NMI/ARI 全部由 `common.py` 用同一套代码算
(ACC 用匈牙利算法最优匹配),口径绝对一致。输出:
- `results/baselines/<method>/<ds>_seed<s>_pred.csv`(一列 pred 一列 true)
- `results/baselines/<method>/<ds>_seed<s>_metrics.json`(ACC/NMI/ARI)

pred.csv 的格式也正好能喂给 figures/ 里的 case-study 可视化脚本。

## 文件清单

| 文件 | 作用 |
|---|---|
| `run.py` | **一键入口**,VSCode 点运行就跑全部 |
| `common.py` | 统一加载+指标+存盘(后端,别动) |
| `run_leiden.py` | Leiden/Seurat runner(现有环境即可跑) |
| `run_scdeepcluster.py` | scDeepCluster runner(bl_torch + clone 仓库) |
| `run_sc3.py` | SC3 runner(bl_r, Python 调 R 脚本) |
| `run_sc3.py` | SC3 runner(bl_r, Python 调 R) |
| `run_dec.py` | DEC runner(自包含, bl_torch) |
| `run_scdsc.py` | scDSC runner(自包含, bl_torch) |
| `run_scgnn.py` | scGNN runner(自包含, bl_torch) |
| `run_scgpt.py` | scGPT runner(bl_torch + scgpt库 + 预训练模型) |
| `run_seurat.py` | Seurat runner(bl_r, R subprocess) |
| `run_scvi.py` | scVI runner(bl_torch, scvi-tools) |
| `run_simlr.py` | SIMLR runner(bl_r, R subprocess) |
| `run_sctag.py` | scTAG runner(bl_tf + Spektral, clone 仓库) |
| `aggregate.py` | 汇总所有结果成对比表 |
| `setup_baseline_envs.sh` | 建其余方法的 conda 环境 |

## 已有 vs 需新跑

- 复用(frozen benchmark 已有):Leiden、DEC、scDCC、scDSC、scGNN、scGPT
- 需新跑:**SC3、scDeepCluster、DESC、scTAG**(runner 待补,补好后同样一键跑)
