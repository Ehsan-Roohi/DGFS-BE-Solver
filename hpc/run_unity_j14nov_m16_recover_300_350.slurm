#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="v100&x86_64"
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=recover-%A_%a.out

set -Eeuo pipefail
trap 'rc=$?; echo "J14NOV_M16_RECOVERY_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CLOSEOUT=${DGFS_CLOSEOUT:?DGFS_CLOSEOUT is required}
SRC=${DGFS_SOLVER_SRC:?DGFS_SOLVER_SRC is required}
FAILED_STEADY=${DGFS_FAILED_STEADY:?DGFS_FAILED_STEADY is required}
RECOVERY=${DGFS_RECOVERY_DIR:?DGFS_RECOVERY_DIR is required}
TSTART=${DGFS_RECOVERY_TSTART:-300.25}
TEND=${DGFS_RECOVERY_TEND:-350.25}

CASES=(M16_raw M16_fplus)
IDX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
CASE=${CASES[$IDX]}
OUT="$RECOVERY/$CASE"
mkdir -p "$OUT"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load cuda/12.6
module load openmpi/5.0.3-cuda12.6
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export CUDA_CACHE_PATH="$RECOVERY/.cuda-cache-$IDX"
export PYTHONPATH="$SRC:$SRC/cases/jcp2019_fig14b_normal_shock/phase3${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CUDA_CACHE_PATH"

# Recovery is deliberately tiny, but refuse to start if the shared project is critically full.
FREE_KB=$(df -Pk "$ROOT" | awk 'NR==2{print $4}')
(( FREE_KB >= 20*1024*1024 )) || { echo "J14NOV_M16_RECOVERY_LOW_SPACE free_kb=$FREE_KB"; exit 20; }

INPUT="$FAILED_STEADY/stage_1/$CASE/dist_p3b_$CASE-${TSTART}.frfss"
test -s "$INPUT"
case "$CASE" in
    M16_raw) SPEC=M16_raw:32:16:none ;;
    M16_fplus) SPEC=M16_fplus:32:16:fplus ;;
    *) exit 8 ;;
esac

cp "$CLOSEOUT/final_runs/run_$CASE/mesh.frfsm" "$OUT/mesh.frfsm"
ln -sf "$INPUT" "$OUT/dist-source.frfss"
"$ENV_DIR/bin/python" "$SRC/cases/jcp2019_fig14b_normal_shock/phase3/make_restart_configs.py" \
    --base "$CLOSEOUT/final_runs/run_$CASE/p3b_$CASE.ini" \
    --residual-csv "$FAILED_STEADY/stage_1/$CASE/kinetic_residual_p3b.csv" \
    --tstart "$TSTART" --tend "$TEND" \
    --dist-dt-out 50.0 --mom-dt-out 5.0 \
    --residual-file kinetic_residual_recovery.csv --runs "$SPEC" --out-dir "$OUT"

echo "J14NOV_M16_RECOVERY_START case=$CASE interval=$TSTART,$TEND input=$INPUT node=${SLURMD_NODENAME:-unknown}"
t0=$(date +%s.%N)
(
  cd "$OUT"
  "$ENV_DIR/bin/python" -m frfs restart mesh.frfsm dist-source.frfss "p3b_$CASE.ini" -b cuda 2>&1 | tee solver.log
)
t1=$(date +%s.%N)
printf 'wall_seconds=%.3f\n' "$(python -c 'import sys; print(float(sys.argv[2])-float(sys.argv[1]))' "$t0" "$t1")" > "$OUT/WALLTIME.txt"

END2=$("$ENV_DIR/bin/python" -c 'import sys; print(f"{float(sys.argv[1]):.2f}")' "$TEND")
FINAL_DIST="$OUT/dist_p3b_$CASE-$END2.frfss"
FINAL_BULK="$OUT/bulksol_p3b_$CASE-$END2.frfss"
test -s "$FINAL_DIST"
test -s "$FINAL_BULK"
test -s "$OUT/kinetic_residual_recovery.csv"

read -r RF RN < <("$ENV_DIR/bin/python" - "$OUT/kinetic_residual_recovery.csv" <<'PY'
import csv, math, sys
rows=[]
with open(sys.argv[1], newline='') as f:
    for r in csv.DictReader(f):
        try: x=(float(r['f']), float(r['f_normalized']))
        except (KeyError,TypeError,ValueError): continue
        if all(math.isfinite(v) for v in x): rows.append(x)
if not rows: raise SystemExit('no finite residual rows')
print(*rows[-1])
PY
)

{
  echo "case=$CASE"
  echo "tstart=$TSTART"
  echo "tend=$TEND"
  echo "source_checkpoint=$INPUT"
  echo "residual_raw=$RF"
  echo "residual_normalized=$RN"
  cat "$OUT/WALLTIME.txt"
} > "$OUT/RECOVERY_METADATA.txt"

"$ENV_DIR/bin/python" -c 'import sys; raise SystemExit(0 if float(sys.argv[1]) <= 1.0 else 1)' "$RN"
touch "$OUT/CONVERGED" "$OUT/RECOVERY_SUCCESS"
echo "J14NOV_M16_RECOVERY_CONVERGED case=$CASE residual_normalized=$RN final_dist=$FINAL_DIST"
