#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --job-name=dgfs-kinetic-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=vram16
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -Eeuo pipefail
trap 'rc=$?; echo "DGFS_KINETIC_SMOKE_FAILED rc=${rc} line=${LINENO}"; exit "$rc"' ERR

DGFS_ROOT=/project/pi_roohie_umass_edu/DGFS_BE
DGFS_ENV="$DGFS_ROOT/dgfs_py310"
DGFS_SRC="${DGFS_SRC:-$DGFS_ROOT/DGFS-BE-Solver}"
RUN_DIR="$(pwd)"
RESTART_FILE="${RESTART_FILE:-dist_dgfs_2d_normalShock-6.0.frfss}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load cuda/12.6
module load openmpi/5.0.3-cuda12.6
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$DGFS_ENV"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_CACHE_PATH="$RUN_DIR/.cuda-cache"
mkdir -p "$CUDA_CACHE_PATH"

test -s mesh.frfsm
test -s "$RESTART_FILE"
test -s normal_shock_M1p592_kinetic_diagnostics.ini
test -s "$DGFS_SRC/tools/validate_kinetic_diagnostics.py"

python -c 'from frfs.plugins.dgfskineticdiagnosticsstd import DGFSKineticDiagnosticsStdPlugin; print("KINETIC_PLUGIN_IMPORT_OK")'
python -c 'import pycuda.driver as cuda; cuda.init(); print("CUDA_PREFLIGHT_OK", cuda.Device(0).name())'

frfs restart mesh.frfsm "$RESTART_FILE" \
  normal_shock_M1p592_kinetic_diagnostics.ini -b cuda

python "$DGFS_SRC/tools/validate_kinetic_diagnostics.py" \
  kinetic_diagnostics.csv --minimum-samples 5 --require-troubled-cells

test -s kinetic_cells_rank00000.csv
sha256sum normal_shock_M1p592_kinetic_diagnostics.ini \
  kinetic_diagnostics.csv kinetic_cells_rank00000.csv \
  kinetic_smoke_residual.csv > KINETIC_SMOKE_SHA256SUMS.txt

touch DGFS_KINETIC_SMOKE_SUCCESS
echo "DGFS_KINETIC_SMOKE_COMPLETE"
echo "RUN_DIR=$RUN_DIR"
