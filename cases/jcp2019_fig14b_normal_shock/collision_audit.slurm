#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --job-name=dgfs-q-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=vram16
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -Eeuo pipefail
trap 'rc=$?; rm -f COLLISION_AUDIT_SUCCESS; echo "COLLISION_AUDIT_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

DGFS_ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
DGFS_ENV=${DGFS_ENV:-$DGFS_ROOT/dgfs_py310}
RUN_DIR=$(pwd)
SOLVER_SRC=${DGFS_SOLVER_SRC:-$RUN_DIR/source/DGFS-BE-Solver}

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
export PYTHONPATH="$SOLVER_SRC${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CUDA_CACHE_PATH"

test -s case.ini
test -s snapshot.frfss
test -s mesh.frfsm
test -s audit_collision.py
nvidia-smi

python audit_collision.py \
    --config case.ini \
    --snapshot snapshot.frfss \
    --mesh mesh.frfsm \
    --output-json collision_audit.json \
    --output-csv collision_audit.csv \
    --repeats "${DGFS_COLLISION_REPEATS:-3}" \
    --max-points "${DGFS_COLLISION_MAX_POINTS:-0}" \
    --relative-tolerance "${DGFS_COLLISION_TOLERANCE:-1e-8}"

test -s collision_audit.json
test -s collision_audit.csv
sha256sum snapshot.frfss > SNAPSHOT_SHA256.txt
{
    echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
    echo "solver_commit=$(git -C "$SOLVER_SRC" rev-parse HEAD)"
    echo "source_snapshot=$(cut -d= -f2- SNAPSHOT_SOURCE.txt)"
    echo "repeats=${DGFS_COLLISION_REPEATS:-3}"
    echo "max_points=${DGFS_COLLISION_MAX_POINTS:-0}"
    echo "relative_tolerance=${DGFS_COLLISION_TOLERANCE:-1e-8}"
} > RUN_METADATA.txt

touch COLLISION_AUDIT_SUCCESS
sha256sum collision_audit.json collision_audit.csv audit_collision.py \
    case.ini mesh.frfsm RUN_METADATA.txt SNAPSHOT_SOURCE.txt SNAPSHOT_SHA256.txt \
    COLLISION_AUDIT_SUCCESS > SHA256SUMS.txt

ZIP="$DGFS_ROOT/collision_audit_${SLURM_JOB_ID:-manual}.zip"
zip -q -j -9 "$ZIP" collision_audit.json collision_audit.csv \
    audit_collision.py case.ini mesh.frfsm RUN_METADATA.txt SNAPSHOT_SOURCE.txt SNAPSHOT_SHA256.txt \
    SHA256SUMS.txt COLLISION_AUDIT_SUCCESS "slurm-${SLURM_JOB_ID}.out"
zip -T "$ZIP"
sha256sum "$ZIP" > "$ZIP.sha256.txt"

echo "COLLISION_AUDIT_COMPLETE"
echo "UPLOAD_ZIP=$ZIP"
echo "UPLOAD_SHA=$ZIP.sha256.txt"
ls -lh "$ZIP" "$ZIP.sha256.txt"
