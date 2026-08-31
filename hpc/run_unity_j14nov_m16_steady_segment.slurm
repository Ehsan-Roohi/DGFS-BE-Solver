#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="v100&x86_64"
#SBATCH --mem=24G
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%A_%a.out

set -Eeuo pipefail
trap 'rc=$?; if [[ -n "${STAGE:-}" ]]; then rm -f "$STAGE/SEGMENT_SUCCESS" 2>/dev/null || true; fi; echo "J14NOV_M16_STEADY_SEGMENT_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CLOSEOUT=${DGFS_CLOSEOUT:?DGFS_CLOSEOUT is required}
ORIGINAL=${DGFS_NOV_SOURCE:?DGFS_NOV_SOURCE is required}
SRC=${DGFS_SOLVER_SRC:?DGFS_SOLVER_SRC is required}
STEADY=${DGFS_STEADY_DIR:?DGFS_STEADY_DIR is required}
SEGMENT=${DGFS_SEGMENT:?DGFS_SEGMENT is required}
TSTART=${DGFS_TSTART:?DGFS_TSTART is required}
TEND=${DGFS_TEND:?DGFS_TEND is required}
DIST_DT_OUT=${DGFS_DIST_DT_OUT:-60.0}
MOM_DT_OUT=${DGFS_MOM_DT_OUT:-5.0}
MIN_FREE_GB=${DGFS_MIN_FREE_GB:-60}

CASES=(M16_raw M16_fplus)
IDX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
CASE=${CASES[$IDX]}
STAGE="$STEADY/stage_$SEGMENT/$CASE"
mkdir -p "$STAGE"
[[ ! -e "$STAGE/SEGMENT_SUCCESS" ]] || { echo "J14NOV_M16_STEADY_ALREADY_COMPLETE=$CASE:$SEGMENT"; exit 7; }

# Refuse to burn V100 time when the shared project filesystem is too full for
# even the reduced-cadence distribution checkpoints.
FREE_KB=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
NEED_KB=$((MIN_FREE_GB * 1024 * 1024))
if (( FREE_KB < NEED_KB )); then
    echo "J14NOV_M16_STEADY_LOW_SPACE free_kb=$FREE_KB required_kb=$NEED_KB min_free_gb=$MIN_FREE_GB"
    exit 9
fi

echo "J14NOV_M16_STEADY_STORAGE_POLICY case=$CASE dist_dt_out=$DIST_DT_OUT mom_dt_out=$MOM_DT_OUT min_free_gb=$MIN_FREE_GB free_kb=$FREE_KB"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load cuda/12.6
module load openmpi/5.0.3-cuda12.6
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export CUDA_CACHE_PATH="$STEADY/.cuda-cache-$IDX"
export PYTHONPATH="$SRC:$SRC/cases/jcp2019_fig14b_normal_shock/phase3${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CUDA_CACHE_PATH"

if (( SEGMENT == 1 )); then
    INPUT="$CLOSEOUT/final_runs/run_$CASE/dist_p3b_$CASE-180.25.frfss"
else
    PREV="$STEADY/stage_$((SEGMENT-1))/$CASE"
    test -e "$PREV/SEGMENT_SUCCESS"
    if [[ -e "$PREV/CONVERGED" ]]; then
        printf 'case=%s\nsource_segment=%s\nreason=already_converged\n' \
            "$CASE" "$((SEGMENT-1))" > "$STAGE/SKIPPED_CONVERGED.txt"
        touch "$STAGE/SEGMENT_SUCCESS" "$STAGE/CONVERGED"
        echo "J14NOV_M16_STEADY_SKIP case=$CASE segment=$SEGMENT reason=already_converged"
        exit 0
    fi
    INPUT=$(find "$PREV" -maxdepth 1 -type f -name 'dist_p3b_*.frfss' -printf '%f\n' |
        sort -V | tail -1)
    INPUT="$PREV/$INPUT"
fi
test -s "$INPUT"

case "$CASE" in
    M16_raw) SPEC=M16_raw:32:16:none ;;
    M16_fplus) SPEC=M16_fplus:32:16:fplus ;;
    *) echo "J14NOV_M16_STEADY_UNKNOWN_CASE=$CASE"; exit 8 ;;
esac

cp "$CLOSEOUT/final_runs/run_$CASE/mesh.frfsm" "$STAGE/mesh.frfsm"
ln -sf "$INPUT" "$STAGE/dist-source.frfss"
"$ENV_DIR/bin/python" "$SRC/cases/jcp2019_fig14b_normal_shock/phase3/make_restart_configs.py" \
    --base "$CLOSEOUT/final_runs/run_$CASE/p3b_$CASE.ini" \
    --residual-csv "$ORIGINAL/input/kinetic_residual.csv" \
    --tstart "$TSTART" --tend "$TEND" \
    --dist-dt-out "$DIST_DT_OUT" --mom-dt-out "$MOM_DT_OUT" \
    --residual-file kinetic_residual_p3b.csv --runs "$SPEC" --out-dir "$STAGE"

echo "J14NOV_M16_STEADY_RUN_START case=$CASE segment=$SEGMENT interval=$TSTART,$TEND input=$INPUT node=${SLURMD_NODENAME:-unknown}"
t0=$(date +%s.%N)
(
    cd "$STAGE"
    "$ENV_DIR/bin/python" -m frfs restart mesh.frfsm dist-source.frfss \
        "p3b_$CASE.ini" -b cuda 2>&1 | tee solver.log
)
t1=$(date +%s.%N)
"$ENV_DIR/bin/python" -c 'import sys; print(f"wall_seconds={float(sys.argv[2])-float(sys.argv[1]):.3f}")' \
    "$t0" "$t1" > "$STAGE/WALLTIME.txt"

END2=$("$ENV_DIR/bin/python" -c 'import sys; print(f"{float(sys.argv[1]):.2f}")' "$TEND")
FINAL_DIST="$STAGE/dist_p3b_$CASE-$END2.frfss"
FINAL_BULK="$STAGE/bulksol_p3b_$CASE-$END2.frfss"
test -s "$FINAL_DIST"
test -s "$FINAL_BULK"
test -s "$STAGE/kinetic_residual_p3b.csv"
read -r RF RN < <("$ENV_DIR/bin/python" - "$STAGE/kinetic_residual_p3b.csv" <<'PY'
import csv, math, sys
rows = []
with open(sys.argv[1], newline="") as stream:
    for row in csv.DictReader(stream):
        try:
            raw = float(row["f"])
            norm = float(row["f_normalized"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(raw) and math.isfinite(norm):
            rows.append((raw, norm))
if not rows:
    raise SystemExit("no finite residual records")
print(*rows[-1])
PY
)

{
    echo "case=$CASE"
    echo "segment=$SEGMENT"
    echo "tstart=$TSTART"
    echo "tend=$TEND"
    echo "input=$INPUT"
    echo "distribution_checkpoint_interval=$DIST_DT_OUT"
    echo "bulk_output_interval=$MOM_DT_OUT"
    echo "residual_raw=$RF"
    echo "residual_normalized=$RN"
    echo "node=${SLURMD_NODENAME:-unknown}"
    cat "$STAGE/WALLTIME.txt"
} > "$STAGE/SEGMENT_METADATA.txt"

rm -f "$STAGE/CONVERGED" "$STAGE/NOT_CONVERGED"
if "$ENV_DIR/bin/python" -c 'import sys; raise SystemExit(0 if float(sys.argv[1]) <= 1.0 else 1)' "$RN"; then
    touch "$STAGE/CONVERGED"
    echo "J14NOV_M16_STEADY_CONVERGED case=$CASE segment=$SEGMENT residual_normalized=$RN"
else
    touch "$STAGE/NOT_CONVERGED"
    echo "J14NOV_M16_STEADY_CONTINUE case=$CASE segment=$SEGMENT residual_normalized=$RN"
fi

# Keep intermediate checkpoints for recovery, but exclude them from the final package.
mkdir -p "$STAGE/checkpoints"
find "$STAGE" -maxdepth 1 -type f -name 'dist_p3b_*.frfss' ! -name "$(basename "$FINAL_DIST")" \
    -exec mv -t "$STAGE/checkpoints" -- {} +
find "$STAGE" -maxdepth 1 -type f -name 'bulksol_p3b_*.frfss' ! -name "$(basename "$FINAL_BULK")" \
    -exec mv -t "$STAGE/checkpoints" -- {} +

touch "$STAGE/SEGMENT_SUCCESS"
echo "J14NOV_M16_STEADY_RUN_DONE case=$CASE segment=$SEGMENT residual_normalized=$RN $(cat "$STAGE/WALLTIME.txt")"
