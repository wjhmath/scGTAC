#!/bin/bash
#SBATCH --job-name=scAGCR_baselines
#SBATCH --output=/home/liyang/BioJiaheWang/scAGCR/log/baselines_%j.out
#SBATCH --error=/home/liyang/BioJiaheWang/scAGCR/log/baselines_%j.err
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --qos=a100g1
#SBATCH --gres=gpu:a100:1
#SBATCH -p a100
# ══════════════════════════════════════════════════════════════
# run_baselines.sh — 一键跑全部 10 个对比算法
#
# 用法:
#   sbatch run_baselines.sh              # 后台提交, 跑全部
#   sbatch run_baselines.sh seurat dec   # 只跑指定方法
#   bash   run_baselines.sh dec          # 前台测试
# ══════════════════════════════════════════════════════════════
set -uo pipefail

# ── 路径 ──
PROJ="/home/liyang/BioJiaheWang/scAGCR"
BL="${PROJ}/baselines"
DATA="${PROJ}/data"
OUT="${PROJ}/results/baselines"
CONDA_SH="/home/liyang/BioJiaheWang/miniconda3/etc/profile.d/conda.sh"

# ── Python 解释器(显式路径, 不依赖 PATH) ──
P_TORCH="${PROJ}/envs/bl_torch/bin/python"
P_R="${PROJ}/envs/bl_r/bin/python"
P_TF="${PROJ}/envs/bl_tf/bin/python"

# ── 数据集 & 种子 ──
SEEDS=(1 2 3)
DATASETS=(baron multiome muraro_pancreas zheng68k \
          cellxgene_sample_small tabula_muris_kidney tabula_sapiens_ear_utricle)

# ── 10 个方法定义 ──
ALL_METHODS=(seurat dec scdeepcluster scvi scdsc scgnn sctag scgpt simlr sc3)

# 方法 -> 环境名 -> Python 路径
get_env() {
    case $1 in
        seurat|simlr|sc3) echo "bl_r" ;;
        sctag)            echo "bl_tf" ;;
        *)                echo "bl_torch" ;;
    esac
}
get_py() {
    case $1 in
        seurat|simlr|sc3) echo "$P_R" ;;
        sctag)            echo "$P_TF" ;;
        *)                echo "$P_TORCH" ;;
    esac
}

# ── 环境初始化 ──
source "${CONDA_SH}"
unset CC CXX
mkdir -p "${PROJ}/log"

# ── 支持命令行指定部分方法 ──
if [ $# -gt 0 ]; then
    METHODS=("$@")
else
    METHODS=("${ALL_METHODS[@]}")
fi

# ── 查找 h5ad ──
find_h5ad() {
    local ds=$1
    for p in "${DATA}/${ds}/${ds}.h5ad" "${DATA}/${ds}.h5ad"; do
        [ -f "$p" ] && echo "$p" && return
    done
}

# ══════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════════════════"
echo "scAGCR Baselines — ${#METHODS[@]} methods × ${#DATASETS[@]} datasets × ${#SEEDS[@]} seeds"
echo "方法: ${METHODS[*]}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "时间: $(date)"
echo "════════════════════════════════════════════════════════"

total_ok=0; total_fail=0; total_skip=0

for method in "${METHODS[@]}"; do
    echo ""
    echo "══════════════════ ${method} ══════════════════"
    env_name=$(get_env "$method")
    py=$(get_py "$method")
    conda activate "$env_name" 2>/dev/null || true

    ok=0; fail=0; skip=0
    for ds in "${DATASETS[@]}"; do
        h5ad=$(find_h5ad "$ds")
        if [ -z "$h5ad" ]; then
            echo "  [SKIP] ${ds}: h5ad 不存在"
            skip=$((skip+1)); continue
        fi
        for s in "${SEEDS[@]}"; do
            out_dir="${OUT}/${method}"
            done_f="${out_dir}/${ds}_seed${s}_metrics.json"
            if [ -f "$done_f" ]; then
                echo "  [SKIP] ${ds} seed${s}: 已完成"
                skip=$((skip+1)); continue
            fi
            mkdir -p "$out_dir"
            echo "  >>> ${method}  ${ds}  seed${s}  $(date +%H:%M:%S)"
            T0=$(date +%s)
            if "$py" "${BL}/run_${method}.py" \
                --data "$h5ad" --out "$out_dir" \
                --dataset "$ds" --seed "$s" 2>&1; then
                T1=$(date +%s)
                echo "  [OK] ${ds} seed${s} ($((T1-T0))s)"
                ok=$((ok+1))
            else
                echo "  [FAIL] ${ds} seed${s}"
                fail=$((fail+1))
            fi
        done
    done
    echo "  ── ${method} 小计: 成功=${ok} 失败=${fail} 跳过=${skip}"
    total_ok=$((total_ok+ok))
    total_fail=$((total_fail+fail))
    total_skip=$((total_skip+skip))
    conda deactivate 2>/dev/null || true
done

# ── 汇总 ──
echo ""
echo "════════════════════════════════════════════════════════"
echo "全部完成: 成功=${total_ok} 失败=${total_fail} 跳过=${total_skip}"
echo "时间: $(date)"
echo "════════════════════════════════════════════════════════"

echo "汇总中..."
"${P_TORCH}" "${BL}/aggregate.py" --root "${OUT}" --out "${OUT}/summary.csv" 2>&1 || true

echo ""
echo "结果在: ${OUT}/summary.csv"
echo "日志在: ${PROJ}/log/baselines_${SLURM_JOB_ID:-local}.out"
