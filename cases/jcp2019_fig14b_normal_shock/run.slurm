#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --job-name=dgfs-jcp-f14b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=vram16
#SBATCH --mem=16G
#SBATCH --time=03:50:00

set -Eeuo pipefail
trap 'rc=$?; echo "DGFS_JCP2019_FIG14B_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

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

python verify_case.py
nvidia-smi
python -m frfs import mesh_fig14b_8elem.msh mesh.frfsm
python -m frfs run mesh.frfsm dgfs_fig14b.ini -b cuda 2>&1 | tee solver.log

test -s dist_dgfs_fig14b-10.0.frfss
test -s bulksol_dgfs_fig14b-10.0.frfss
test -s kinetic_residual.csv

python -m frfs export mesh.frfsm bulksol_dgfs_fig14b-10.0.frfss fig14b_t10.vtu -d 24
test -s fig14b_t10.vtu

FINAL_RESIDUAL=$(awk -F, 'NR > 2 {v=$3} END {print v}' kinetic_residual.csv)
python - "$FINAL_RESIDUAL" <<'PY'
import sys
value = float(sys.argv[1])
print(f"FINAL_PAPER_NORMALIZED_RESIDUAL={value:.12e}")
if not value < 2.0e-5:
    raise SystemExit("paper convergence threshold 2e-5 was not reached")
PY

if python -c 'import matplotlib' >/dev/null 2>&1; then
    python plot_fig14b.py --run-dir . --output fig14b_raw_and_cell_average.png
else
    echo "PLOT_SKIPPED: matplotlib is not installed in $DGFS_ENV"
fi

{
    echo "paper=Jaiswal_Alexeenko_Hu_JCP378_2019_Figure14b"
    echo "slurm_job_id=${SLURM_JOB_ID:-not-slurm}"
    echo "solver_commit=$(git -C "$SOLVER_SRC" rev-parse HEAD)"
    echo "physical_domain_m=-0.015,0.015"
    echo "spatial_elements=8"
    echo "polynomial_degree=2"
    echo "velocity_grid=32^3"
    echo "velocity_domain=-7,7"
    echo "final_paper_normalized_residual=$FINAL_RESIDUAL"
} > RUN_METADATA.txt

CHECKSUM_FILES=(dgfs_fig14b.ini mesh_fig14b_8elem.msh mesh.frfsm
    dist_dgfs_fig14b-10.0.frfss bulksol_dgfs_fig14b-10.0.frfss
    fig14b_t10.vtu kinetic_residual.csv RUN_METADATA.txt run.slurm)
if [[ -s fig14b_raw_and_cell_average.png ]]; then
    CHECKSUM_FILES+=(fig14b_raw_and_cell_average.png fig14b_extrema.json)
fi
sha256sum "${CHECKSUM_FILES[@]}" > SHA256SUMS.txt

touch DGFS_JCP2019_FIG14B_SUCCESS
ZIP="$RUN_DIR/../normal_shock_M1p59_jcp2019_fig14b_${SLURM_JOB_ID:-manual}_COMPLETE.zip"
(cd "$RUN_DIR" && zip -9 -r "$ZIP" . -x '.cuda-cache/*' '*.zip' '*.sha256.txt')
sha256sum "$ZIP" > "$ZIP.sha256.txt"

echo "DGFS_JCP2019_FIG14B_COMPLETE"
echo "FINAL_PAPER_NORMALIZED_RESIDUAL=$FINAL_RESIDUAL"
echo "RUN_DIR=$RUN_DIR"
echo "DOWNLOAD_ZIP=$ZIP"
echo "DOWNLOAD_SHA256=$ZIP.sha256.txt"
