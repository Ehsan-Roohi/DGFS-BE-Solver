#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --job-name=dgfs-p4e
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=vram16
#SBATCH --mem=24G
#SBATCH --time=00:45:00
set -Eeuo pipefail
trap 'rc=$?; rm -f P4E_SUCCESS; echo "P4E_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR
DGFS_ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
DGFS_ENV=${DGFS_ENV:-$DGFS_ROOT/dgfs_py310}
BASELINE=${DGFS_P4A_ROOT:?DGFS_P4A_ROOT is required}
RUN_DIR=$(pwd)
SOLVER_SRC=${DGFS_SOLVER_SRC:-$RUN_DIR/src}
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load cuda/12.6
module load openmpi/5.0.3-cuda12.6
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$DGFS_ENV"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export CUDA_CACHE_PATH="$RUN_DIR/.cuda-cache" MPLCONFIGDIR="$RUN_DIR/.matplotlib"
export PYTHONPATH="$SOLVER_SRC:$RUN_DIR/p3${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" p4e_results

python solver_hook/apply_hook.py "$SOLVER_SRC"
python p3/gpu_layout_preflight.py --config dgfs_fig14b.ini --mode fplus-transverse 2>&1 | tee gpu_layout_preflight.log
grep -q P3B_GPU_LAYOUT_PREFLIGHT_PASS gpu_layout_preflight.log
python p3/make_restart_configs.py --base dgfs_fig14b.ini --residual-csv kinetic_residual.csv \
  --tstart 0 --tend 1.0 --dt-out 0.25 \
  --runs M16_transverse:16:16:fplus-transverse --out-dir configs

mkdir -p run_M16_transverse
cp mesh.frfsm run_M16_transverse/
ln -sf "$RUN_DIR/dist_dgfs_fig14b-0.0.frfss" run_M16_transverse/dist_dgfs_fig14b-0.0.frfss
cp configs/p3b_M16_transverse.ini run_M16_transverse/
echo "P4E_RUN_START $(date -Is)"
t0=$(date +%s.%N)
(cd run_M16_transverse && python -m frfs restart mesh.frfsm dist_dgfs_fig14b-0.0.frfss \
  p3b_M16_transverse.ini -b cuda 2>&1 | tee solver.log)
t1=$(date +%s.%N)
python - "$t0" "$t1" > run_M16_transverse/WALLTIME.txt <<'PY'
import sys
print(f"wall_seconds={float(sys.argv[2])-float(sys.argv[1]):.3f}")
PY
for time in 0.25 0.50 0.75 1.00; do
  test -s "run_M16_transverse/dist_p3b_M16_transverse-$time.frfss"
done
echo "P4E_RUN_DONE $(cat run_M16_transverse/WALLTIME.txt)"

python p3/p4e_time_history.py --baseline-root "$BASELINE" --transverse-root "$RUN_DIR" \
  --out-dir p4e_results 2>&1 | tee p4e_analysis.log
grep -q P4E_TIME_HISTORY_COMPLETE p4e_analysis.log
if python -c 'import matplotlib' >/dev/null 2>&1; then
  python p3/p4e_plot.py --profiles p4e_results/p4e_physical_profiles.csv \
    --history p4e_results/p4e_time_history.csv --out-dir p4e_results 2>&1 | tee p4e_plot.log
else
  echo P4E_PLOTS_SKIPPED_MATPLOTLIB_MISSING | tee p4e_plot.log
fi

touch P4E_SUCCESS
ZIP="$DGFS_ROOT/p4e_${SLURM_JOB_ID:-manual}.zip"
zip -q -9 -r "$ZIP" P4E_SUCCESS INPUT.txt p4e_results p4e_analysis.log p4e_plot.log \
  gpu_layout_preflight.log configs run_M16_transverse p3 solver_hook dgfs_fig14b.ini \
  mesh.frfsm "slurm-${SLURM_JOB_ID:-manual}.out" \
  -x '*/__pycache__/*' '*.pyc' 'run_M16_transverse/dist_dgfs_fig14b-0.0.frfss'
zip -T "$ZIP"
sha256sum "$ZIP" > "$ZIP.sha256.txt"
echo P4E_COMPLETE
echo "UPLOAD_ZIP=$ZIP"
echo "UPLOAD_SHA=$ZIP.sha256.txt"
