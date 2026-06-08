#!/bin/bash
#SBATCH --job-name=bl_chuli
#SBATCH --output=/home/liyang/BioJiaheWang/scGTAC/log/bl_chuli_%j.out
#SBATCH --error=/home/liyang/BioJiaheWang/scGTAC/log/bl_chuli_%j.err
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --qos=a100g1
#SBATCH --gres=gpu:a100:1
#SBATCH -p a100

PROJ="/home/liyang/BioJiaheWang/scGTAC"
CHULI="/home/liyang/BioJiaheWang/RARECELL/data/chuli"
CONDA_SH="/home/liyang/BioJiaheWang/miniconda3/etc/profile.d/conda.sh"
OUT="${PROJ}/results/baselines_chuli"

source "${CONDA_SH}"
cd "${PROJ}/baselines"
mkdir -p "${OUT}"

PY_TORCH="${PROJ}/envs/bl_torch/bin/python"
PY_R="${PROJ}/envs/bl_r/bin/python"

METHODS=(
  "dec|run_dec.py|${PY_TORCH}"
  "scdeepcluster|run_scdeepcluster.py|${PY_TORCH}"
  "scvi|run_scvi.py|${PY_TORCH}"
  "scgnn|run_scgnn.py|${PY_TORCH}"
  "scdsc|run_scdsc.py|${PY_TORCH}"
  "seurat|run_seurat.py|${PY_R}"
)

SEED=1
success=0; fail=0; skip=0

echo "════════════════════════════════════════════════════════"
echo "Baselines x chuli  $(date)"
echo "════════════════════════════════════════════════════════"

for H5AD in ${CHULI}/*.h5ad; do
    [ -f "${H5AD}" ] || continue
    DS=$(basename "${H5AD}" .h5ad)

    for entry in "${METHODS[@]}"; do
        IFS='|' read -r METHOD RUNNER PY <<< "${entry}"
        OUTDIR="${OUT}/${METHOD}"
        mkdir -p "${OUTDIR}"
        METRIC="${OUTDIR}/${DS}_seed${SEED}_metrics.json"

        if [ -f "${METRIC}" ]; then
            skip=$((skip + 1)); continue
        fi

        echo ">>> ${METHOD} | ${DS} | $(date +%H:%M:%S)"
        T0=$(date +%s)
        if timeout 1800 ${PY} "${RUNNER}" --data "${H5AD}" --out "${OUTDIR}" --dataset "${DS}" --seed "${SEED}" 2>&1; then
            T1=$(date +%s)
            if [ -f "${METRIC}" ]; then
                echo "[OK] $((T1-T0))s"
                success=$((success + 1))
            else
                echo "[FAIL] 无输出"
                fail=$((fail + 1))
            fi
        else
            echo "[FAIL] 超时/报错"
            fail=$((fail + 1))
        fi
    done
done

echo ""
echo "完成: 成功=${success} 失败=${fail} 跳过=${skip}"
echo "════════════════════════════════════════════════════════"

cd "${PROJ}"
python aggregate_chuli.py
