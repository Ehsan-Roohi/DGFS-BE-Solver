#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --job-name=dgfs-p6-m2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=vram24
#SBATCH --mem=32G
#SBATCH --time=03:50:00
set -Eeuo pipefail
trap 'rc=$?; rm -f P6_SUCCESS; echo "P6_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR
: "${DGFS_ENV:?}" "${DGFS_SOLVER_SRC:?}"
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load cuda/12.6 openmpi/5.0.3-cuda12.6 conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$DGFS_ENV"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
export PYTHONPATH="$DGFS_SOLVER_SRC${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi
python -m frfs import mesh_M2_32elem.msh mesh_M2.frfsm
python -m frfs run mesh_M2.frfsm dgfs_M2.ini -b cuda 2>&1 | tee solver_M2.log
test -s dist_dgfs_M2-0.25.frfss
test -s bulksol_dgfs_M2-0.25.frfss
test -s kinetic_residual_m2.csv
python -m frfs export mesh_M2.frfsm bulksol_dgfs_M2-0.25.frfss dgfs_M2_t0p25.vtu -d 48
touch P6_SUCCESS
ZIP="p6_m2_${SLURM_JOB_ID}.zip"
zip -q -9 "$ZIP" dgfs_M2.ini mesh_M2_32elem.msh mesh_M2.frfsm dist_dgfs_M2-0.25.frfss \
  bulksol_dgfs_M2-0.25.frfss kinetic_residual_m2.csv dgfs_M2_t0p25.vtu solver_M2.log \
  dvm_M2_reference.csv P6_CASE_CONTRACT.json P6_CASE_VERIFIED P6_SUCCESS "slurm-${SLURM_JOB_ID}.out"
zip -T "$ZIP"
sha256sum "$ZIP" > "$ZIP.sha256.txt"
echo P6_COMPLETE
echo "UPLOAD_ZIP=$PWD/$ZIP"
echo "UPLOAD_SHA=$PWD/$ZIP.sha256.txt"
