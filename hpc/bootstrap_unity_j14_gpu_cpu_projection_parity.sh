#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "GPU_CPU_PARITY_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CLOSE=${DGFS_CLOSEOUT:-$ROOT/j14novclose_20260826_211439}
STEADY=${DGFS_STEADY_DIR:-$CLOSE/m16_steady_20260831_125816}
SRC=${DGFS_SOLVER_SRC:-$STEADY/src}
REF=${DGFS_REF:-a6ee593391bcdb7eefd09510188ffa7c9f04224b}
OUT=${DGFS_OUTPUT_DIR:-$PWD}
M6=$CLOSE/final_runs/run_M6_raw

mkdir -p "$OUT"
test -x "$ENV_DIR/bin/python"
test -d "$SRC"
test -s "$M6/p3b_M6_raw.ini"

SNAP=$({ find "$M6" -maxdepth 2 -type f -name 'dist_p3b_M6_raw-*.frfss' -printf '%p\n' 2>/dev/null || true; } | sort -V | tail -1)
[[ -s "$SNAP" ]] || { echo "PARITY_SNAPSHOT_NOT_FOUND"; exit 2; }

PY="$OUT/j14_gpu_cpu_projection_parity.py"
curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/DGFS-BE-Solver/$REF/cases/jcp2019_fig14b_normal_shock/novelty/j14_gpu_cpu_projection_parity.py" -o "$PY"
"$ENV_DIR/bin/python" -m py_compile "$PY"

SLURM="$OUT/dgfs-proj-parity.slurm"
cat > "$SLURM" <<SLURM
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="v100&x86_64"
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=dgfs-parity
#SBATCH --output=$OUT/dgfs-parity-%j.out
set -Eeuo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load cuda/12.6
module load openmpi/5.0.3-cuda12.6
module load conda/latest
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH="$SRC\${PYTHONPATH:+:\$PYTHONPATH}"
export CUDA_CACHE_PATH="$OUT/.cuda-cache-parity-\${SLURM_JOB_ID}"
mkdir -p "\$CUDA_CACHE_PATH"
cd "$OUT"
nvidia-smi --query-gpu=name --format=csv,noheader
"$ENV_DIR/bin/python" "$PY" \
  --config "$M6/p3b_M6_raw.ini" \
  --snapshot "$SNAP" \
  --output "$OUT/GPU_CPU_PROJECTION_PARITY.json" \
  --tol 5e-12
rm -rf "\$CUDA_CACHE_PATH" || true
SLURM

JOB=$(sbatch --parsable "$SLURM"); JOB=${JOB%%;*}
echo "GPU_CPU_PARITY_JOB=$JOB"
echo "GPU_CPU_PARITY_OUTPUT=$OUT/GPU_CPU_PROJECTION_PARITY.json"
echo "GPU_CPU_PARITY_BOOTSTRAP_COMPLETE"
