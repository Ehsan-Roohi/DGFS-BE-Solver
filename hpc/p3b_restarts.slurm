#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --job-name=dgfs-p3b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=vram16
#SBATCH --mem=24G
#SBATCH --time=03:00:00

# Phase-3b: four short restarts t=30 -> 30.1 (M6 raw, M16 raw,
# M16 fplus, M24 raw) run sequentially in one job, then compared.
# Expects in $(pwd): dgfs_fig14b.ini, mesh.frfsm, dist_dgfs_fig14b-30.0.frfss,
# kinetic_residual.csv (production, for residual normalisation), p3/, solver_hook/.
set -Eeuo pipefail
trap 'rc=$?; rm -f P3B_SUCCESS; echo "P3B_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

DGFS_ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
DGFS_ENV=${DGFS_ENV:-$DGFS_ROOT/dgfs_py310}
RUN_DIR=$(pwd)
SOLVER_SRC=${DGFS_SOLVER_SRC:-$RUN_DIR/source/DGFS-BE-Solver}
RUNS=${DGFS_P3B_RUNS:-M6_raw:32:6:none,M16_raw:16:16:none,M16_fplus:16:16:fplus,M24_raw:16:24:none}
TEND=${DGFS_P3B_TEND:-30.1}
DTOUT=${DGFS_P3B_DTOUT:-0.05}

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
export PYTHONPATH="$SOLVER_SRC:$RUN_DIR/p3${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CUDA_CACHE_PATH"

test -s dgfs_fig14b.ini; test -s mesh.frfsm; test -s dist_dgfs_fig14b-30.0.frfss
test -s p3/make_restart_configs.py; test -s solver_hook/apply_hook.py
nvidia-smi

# install the projection hook into the solver checkout (idempotent)
python solver_hook/apply_hook.py "$SOLVER_SRC"
python -c "import frfs.solvers.dgfs.projection, frfs.solvers.dgfs.system; print('hook import ok')"
python p3/gpu_layout_preflight.py --config dgfs_fig14b.ini 2>&1 | tee gpu_layout_preflight.log
grep -q 'P3B_GPU_LAYOUT_PREFLIGHT_PASS' gpu_layout_preflight.log

python p3/make_restart_configs.py --base dgfs_fig14b.ini --residual-csv kinetic_residual.csv \
    --tstart 30 --tend "$TEND" --dt-out "$DTOUT" --runs "$RUNS" --out-dir configs

IFS=',' read -r -a RUNLIST <<< "$RUNS"
for spec in "${RUNLIST[@]}"; do
    name=${spec%%:*}
    d="run_$name"
    mkdir -p "$d"
    cp mesh.frfsm "$d/"
    ln -sf "$RUN_DIR/dist_dgfs_fig14b-30.0.frfss" "$d/dist_dgfs_fig14b-30.0.frfss"
    cp "configs/p3b_$name.ini" "$d/"
    echo "P3B_RUN_START name=$name $(date)"
    t0=$(date +%s.%N)
    ( cd "$d" && python -m frfs restart mesh.frfsm dist_dgfs_fig14b-30.0.frfss "p3b_$name.ini" -b cuda \
        2>&1 | tee solver.log )
    t1=$(date +%s.%N)
    python - "$t0" "$t1" > "$d/WALLTIME.txt" <<'PY'
import sys; print(f"wall_seconds={float(sys.argv[2])-float(sys.argv[1]):.3f}")
PY
    echo "P3B_RUN_DONE name=$name $(cat "$d/WALLTIME.txt")"
done

python p3/compare_restarts.py --config dgfs_fig14b.ini --mesh mesh.frfsm \
    --reference dist_dgfs_fig14b-30.0.frfss --runs run_* --tend "$TEND" \
    --output-json p3b_comparison.json --output-csv p3b_comparison.csv \
    --output-md p3b_comparison.md --output-figure p3b_comparison.png \
    --output-cell-figure p3b_cell_averages.png 2>&1 | tee p3b_compare.log
test -s p3b_comparison.json
touch P3B_SUCCESS
ZIP="$DGFS_ROOT/p3b_${SLURM_JOB_ID:-manual}.zip"
zip -q -9 -r "$ZIP" p3b_comparison.* p3b_cell_averages.png p3b_compare.log \
    gpu_layout_preflight.log configs run_* p3 solver_hook INPUT.txt \
    dgfs_fig14b.ini mesh.frfsm dist_dgfs_fig14b-30.0.frfss \
    "slurm-${SLURM_JOB_ID:-manual}.out" \
    -x '*/__pycache__/*' '*.pyc' 'run_*/dist_dgfs_fig14b-30.0.frfss'
zip -T "$ZIP"; sha256sum "$ZIP" > "$ZIP.sha256.txt"
echo "P3B_COMPLETE"; echo "UPLOAD_ZIP=$ZIP"; echo "UPLOAD_SHA=$ZIP.sha256.txt"
