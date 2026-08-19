#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --job-name=dgfs-p3-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=vram16
#SBATCH --mem=24G
#SBATCH --time=01:00:00

# Phase-3 audit: quadrature sensitivity (32:6 baseline, 16:16 proposed, 16:24 reference)
# x {raw, euclidean, f-weighted, fplus, maxwellian} x 24 DG points + GPU projection timing.
# Expects in $(pwd): case.ini, snapshot.frfss, mesh.frfsm, p3/ (this package's p3 directory).
set -Eeuo pipefail
trap 'rc=$?; rm -f P3_AUDIT_SUCCESS; echo "P3_AUDIT_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

DGFS_ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
DGFS_ENV=${DGFS_ENV:-$DGFS_ROOT/dgfs_py310}
RUN_DIR=$(pwd)
SOLVER_SRC=${DGFS_SOLVER_SRC:-$RUN_DIR/source/DGFS-BE-Solver}
QUADS=${DGFS_P3_QUADRATURES:-32:6,16:16,16:24}

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load cuda/12.6
module load openmpi/5.0.3-cuda12.6
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$DGFS_ENV"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_CACHE_PATH="$RUN_DIR/.cuda-cache"
export PYTHONPATH="$SOLVER_SRC:$RUN_DIR/p3${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CUDA_CACHE_PATH"

test -s case.ini
test -s snapshot.frfss
test -s mesh.frfsm
test -s p3/audit_quadrature_projection.py
nvidia-smi

python p3/audit_quadrature_projection.py \
    --config case.ini --snapshot snapshot.frfss --mesh mesh.frfsm \
    --quadratures "$QUADS" \
    --repeats "${DGFS_P3_REPEATS:-3}" --projection-repeats "${DGFS_P3_PROJ_REPEATS:-5}" \
    --gpu-solve "${DGFS_P3_GPU_SOLVE:-device}" --skip-gpu-weighting f \
    --output-dir . \
    --output-json p3_quadrature_projection_audit.json \
    --output-csv p3_quadrature_projection_audit.csv 2>&1 | tee p3_audit.log

test -s p3_quadrature_projection_audit.json
sha256sum snapshot.frfss > SNAPSHOT_SHA256.txt
{
    echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
    echo "solver_commit=$(git -C "$SOLVER_SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "quadratures=$QUADS"
    echo "gpu_solve=${DGFS_P3_GPU_SOLVE:-device}"
} > RUN_METADATA.txt
touch P3_AUDIT_SUCCESS
sha256sum p3_quadrature_projection_audit.json p3_quadrature_projection_audit.csv p3_audit.log \
    p3/*.py case.ini mesh.frfsm RUN_METADATA.txt SNAPSHOT_SHA256.txt > SHA256SUMS.txt

ZIP="$DGFS_ROOT/p3a_${SLURM_JOB_ID:-manual}.zip"
zip -q -9 -r "$ZIP" p3_quadrature_projection_audit.json p3_quadrature_projection_audit.csv \
    p3_partial_*.json p3_audit.log p3 case.ini mesh.frfsm RUN_METADATA.txt SNAPSHOT_SHA256.txt \
    SHA256SUMS.txt P3_AUDIT_SUCCESS "slurm-${SLURM_JOB_ID:-manual}.out" 2>/dev/null || true
zip -T "$ZIP"
sha256sum "$ZIP" > "$ZIP.sha256.txt"
echo "P3_AUDIT_COMPLETE"
echo "UPLOAD_ZIP=$ZIP"
echo "UPLOAD_SHA=$ZIP.sha256.txt"
