#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --job-name=dgfs-p4a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=vram16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
set -Eeuo pipefail
trap 'rc=$?; rm -f P4A_SUCCESS; echo "P4A_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR
DGFS_ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
DGFS_ENV=${DGFS_ENV:-$DGFS_ROOT/dgfs_py310}
RUN_DIR=$(pwd)
SOLVER_SRC=${DGFS_SOLVER_SRC:-$RUN_DIR/src}
RUNS=M16_raw:16:16:none,M16_fplus:16:16:fplus,M24_raw:16:24:none
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load cuda/12.6
module load openmpi/5.0.3-cuda12.6
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$DGFS_ENV"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export CUDA_CACHE_PATH="$RUN_DIR/.cuda-cache"
export MPLCONFIGDIR="$RUN_DIR/.matplotlib"
export PYTHONPATH="$SOLVER_SRC:$RUN_DIR/p3${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"
test -s dgfs_fig14b.ini
test -s mesh.frfsm
test -s dist_dgfs_fig14b-0.0.frfss
python solver_hook/apply_hook.py "$SOLVER_SRC"
python p3/gpu_layout_preflight.py --config dgfs_fig14b.ini 2>&1 | tee gpu_layout_preflight.log
grep -q P3B_GPU_LAYOUT_PREFLIGHT_PASS gpu_layout_preflight.log
python p3/make_restart_configs.py --base dgfs_fig14b.ini --residual-csv kinetic_residual.csv \
  --tstart 0 --tend 1 --dt-out 0.25 --runs "$RUNS" --out-dir configs
IFS=',' read -r -a RUNLIST <<< "$RUNS"
for spec in "${RUNLIST[@]}"; do
  name=${spec%%:*}; d="run_$name"; mkdir -p "$d"
  cp mesh.frfsm "$d/"
  ln -sf "$RUN_DIR/dist_dgfs_fig14b-0.0.frfss" "$d/dist_dgfs_fig14b-0.0.frfss"
  cp "configs/p3b_$name.ini" "$d/"
  echo "P4A_RUN_START name=$name $(date -Is)"
  t0=$(date +%s.%N)
  (cd "$d" && python -m frfs restart mesh.frfsm dist_dgfs_fig14b-0.0.frfss "p3b_$name.ini" -b cuda 2>&1 | tee solver.log)
  t1=$(date +%s.%N)
  python - "$t0" "$t1" > "$d/WALLTIME.txt" <<'PY'
import sys
print(f"wall_seconds={float(sys.argv[2])-float(sys.argv[1]):.3f}")
PY
  test -s "$d/dist_p3b_$name-1.00.frfss"
  echo "P4A_RUN_DONE name=$name $(cat "$d/WALLTIME.txt")"
done
python p3/compare_restarts.py --config dgfs_fig14b.ini --mesh mesh.frfsm \
  --reference dist_dgfs_fig14b-0.0.frfss --runs run_M16_raw run_M16_fplus run_M24_raw \
  --tstart 0 --tend 1 --output-json p4a_comparison.json --output-csv p4a_comparison.csv \
  --output-md p4a_comparison.md --output-figure p4a_profiles_raw_dg.png \
  --output-cell-figure p4a_cell_averages.png 2>&1 | tee p4a_compare.log
python p3/p4a_assess.py --comparison p4a_comparison.json 2>&1 | tee p4a_assess.log
grep -q P4A_ASSESSMENT_PASS p4a_assess.log
# Plotting is post-processing: a missing optional plotting dependency must not
# turn a scientifically completed solver run into a failed batch job.
PLOT_PY=""
for py in "$DGFS_ENV/bin/python" "$(command -v python3 2>/dev/null || true)" "$(command -v python 2>/dev/null || true)"; do
  if [[ -n "$py" && -x "$py" ]] && "$py" -c 'import matplotlib, numpy' >/dev/null 2>&1; then
    PLOT_PY=$py
    break
  fi
done
if [[ -n "$PLOT_PY" ]]; then
  "$PLOT_PY" p3/p3c_physical_profiles.py --comparison p4a_comparison.json --mach 1.59 \
    --profiles-png p4a_physical_profiles_24points.png \
    --profiles-pdf p4a_physical_profiles_24points.pdf \
    --differences-png p4a_physical_differences.png \
    --differences-pdf p4a_physical_differences.pdf \
    --csv p4a_physical_profiles_24points.csv
  echo "P4A_FIGURES_COMPLETE python=$PLOT_PY"
else
  echo P4A_FIGURES_SKIPPED_NO_MATPLOTLIB
fi
touch P4A_SUCCESS
ZIP="$DGFS_ROOT/p4a_${SLURM_JOB_ID:-manual}.zip"
zip -q -9 -r "$ZIP" p4a_* gpu_layout_preflight.log configs run_* p3 solver_hook INPUT.txt \
  dgfs_fig14b.ini mesh.frfsm dist_dgfs_fig14b-0.0.frfss "slurm-${SLURM_JOB_ID:-manual}.out" \
  -x '*/__pycache__/*' '*.pyc' 'run_*/dist_dgfs_fig14b-0.0.frfss'
zip -T "$ZIP"
sha256sum "$ZIP" > "$ZIP.sha256.txt"
echo P4A_COMPLETE
echo "UPLOAD_ZIP=$ZIP"
echo "UPLOAD_SHA=$ZIP.sha256.txt"
