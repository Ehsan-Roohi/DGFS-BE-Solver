#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --job-name=dgfs-p4d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=vram16
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -Eeuo pipefail
trap 'rc=$?; rm -f P4D_SUCCESS; echo "P4D_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR
DGFS_ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
DGFS_ENV=${DGFS_ENV:-$DGFS_ROOT/dgfs_py310}
RUN_DIR=$(pwd)
SOLVER_SRC=${DGFS_SOLVER_SRC:-$RUN_DIR/src}
RUNS=M16_raw:16:16:none,M16_fplus:16:16:fplus,M16_transverse:16:16:fplus-transverse,M24_raw:16:24:none
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
mkdir -p "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"
python solver_hook/apply_hook.py "$SOLVER_SRC"
python p3/gpu_layout_preflight.py --config dgfs_fig14b.ini --mode fplus-transverse 2>&1 | tee gpu_layout_preflight.log
grep -q P3B_GPU_LAYOUT_PREFLIGHT_PASS gpu_layout_preflight.log
python p3/make_restart_configs.py --base dgfs_fig14b.ini --residual-csv kinetic_residual.csv \
  --tstart 0 --tend 0.25 --dt-out 0.05 --runs "$RUNS" --out-dir configs
IFS=',' read -r -a RUNLIST <<< "$RUNS"
for spec in "${RUNLIST[@]}"; do
  name=${spec%%:*}; d="run_$name"; mkdir -p "$d"
  cp mesh.frfsm "$d/"
  ln -sf "$RUN_DIR/dist_dgfs_fig14b-0.0.frfss" "$d/dist_dgfs_fig14b-0.0.frfss"
  cp "configs/p3b_$name.ini" "$d/"
  echo "P4D_RUN_START name=$name $(date -Is)"
  t0=$(date +%s.%N)
  (cd "$d" && python -m frfs restart mesh.frfsm dist_dgfs_fig14b-0.0.frfss "p3b_$name.ini" -b cuda 2>&1 | tee solver.log)
  t1=$(date +%s.%N)
  python - "$t0" "$t1" > "$d/WALLTIME.txt" <<'PY'
import sys
print(f"wall_seconds={float(sys.argv[2])-float(sys.argv[1]):.3f}")
PY
  test -s "$d/dist_p3b_$name-0.25.frfss"
  echo "P4D_RUN_DONE name=$name $(cat "$d/WALLTIME.txt")"
done
python p3/compare_restarts.py --config dgfs_fig14b.ini --mesh mesh.frfsm \
 --reference dist_dgfs_fig14b-0.0.frfss \
 --runs run_M16_raw run_M16_fplus run_M16_transverse run_M24_raw \
 --tstart 0 --tend 0.25 --output-json p4d_comparison.json --output-csv p4d_comparison.csv \
 --output-md p4d_comparison.md --output-figure p4d_profiles.png \
 --output-cell-figure p4d_cell_averages.png 2>&1 | tee p4d_compare.log
python p3/p4d_assess.py --csv p4d_comparison.csv --out p4d_assessment.md 2>&1 | tee p4d_assess.log
grep -q P4D_ASSESSMENT_COMPLETE p4d_assess.log
touch P4D_SUCCESS
ZIP="$DGFS_ROOT/p4d_${SLURM_JOB_ID:-manual}.zip"
zip -q -9 -r "$ZIP" p4d_* gpu_layout_preflight.log configs run_* p3 solver_hook INPUT.txt \
 dgfs_fig14b.ini mesh.frfsm "slurm-${SLURM_JOB_ID:-manual}.out" \
 -x '*/__pycache__/*' '*.pyc' 'run_*/dist_dgfs_fig14b-0.0.frfss'
zip -T "$ZIP"
sha256sum "$ZIP" > "$ZIP.sha256.txt"
echo P4D_COMPLETE
echo "UPLOAD_ZIP=$ZIP"
echo "UPLOAD_SHA=$ZIP.sha256.txt"
