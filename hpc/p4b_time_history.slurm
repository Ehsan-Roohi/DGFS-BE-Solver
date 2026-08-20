#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --job-name=dgfs-p4b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -Eeuo pipefail
trap 'rc=$?; rm -f P4B_SUCCESS; echo "P4B_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

: "${DGFS_ROOT:?DGFS_ROOT is required}"
: "${DGFS_ENV:?DGFS_ENV is required}"
: "${DGFS_SOLVER_SRC:?DGFS_SOLVER_SRC is required}"
: "${P4A_DIR:?P4A_DIR is required}"
RUN_DIR=$(pwd)
PHASE3="$DGFS_SOLVER_SRC/cases/jcp2019_fig14b_normal_shock/phase3"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MPLCONFIGDIR="$RUN_DIR/.matplotlib"
mkdir -p results "$MPLCONFIGDIR"

"$DGFS_ENV/bin/python" "$PHASE3/p4b_time_history.py" \
  --root "$P4A_DIR" --out-dir "$RUN_DIR/results" 2>&1 | tee p4b_analysis.log
grep -q '^P4B_TIME_HISTORY_PASS$' p4b_analysis.log

PLOT_PY=""
CONDA_BASE=$(conda info --base 2>/dev/null || true)
CANDIDATES=("$DGFS_ENV/bin/python")
[[ -n "$CONDA_BASE" ]] && CANDIDATES+=("$CONDA_BASE/bin/python")
[[ -n "$(command -v python3 2>/dev/null || true)" ]] && CANDIDATES+=("$(command -v python3)")
for py in "${CANDIDATES[@]}"; do
  if [[ -n "$py" && -x "$py" ]] && "$py" -c 'import matplotlib' >/dev/null 2>&1; then
    PLOT_PY=$py
    break
  fi
done
if [[ -n "$PLOT_PY" ]]; then
  "$PLOT_PY" "$PHASE3/p4b_plot.py" --csv results/p4b_time_history.csv --out-dir results \
    2>&1 | tee p4b_plot.log
else
  echo P4B_FIGURES_SKIPPED_NO_MATPLOTLIB | tee p4b_plot.log
fi

printf 'p4a_dir=%s\nsource_ref=%s\n' "$P4A_DIR" "${DGFS_REF:-unknown}" > INPUT.txt
cp "$PHASE3/p4b_time_history.py" "$PHASE3/p4b_plot.py" .
touch P4B_SUCCESS
ZIP="$DGFS_ROOT/p4b_${SLURM_JOB_ID:-manual}.zip"
zip -q -9 -r "$ZIP" results p4b_analysis.log p4b_plot.log INPUT.txt P4B_SUCCESS \
  p4b_time_history.py p4b_plot.py "slurm-${SLURM_JOB_ID:-manual}.out"
zip -T "$ZIP"
sha256sum "$ZIP" > "$ZIP.sha256.txt"
echo P4B_COMPLETE
echo "UPLOAD_ZIP=$ZIP"
echo "UPLOAD_SHA=$ZIP.sha256.txt"
