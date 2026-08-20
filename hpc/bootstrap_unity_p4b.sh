#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "P4B_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

DGFS_ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
DGFS_ENV=${DGFS_ENV:-$DGFS_ROOT/dgfs_py310}
DGFS_REPO=${DGFS_REPO:-https://github.com/Ehsan-Roohi/DGFS-BE-Solver.git}
DGFS_REF=${DGFS_REF:-agent/phase3-angular-conservative-audit}
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$DGFS_ROOT/p4b_$STAMP"
SOLVER_SRC="$RUN_DIR/src"

test -x "$DGFS_ENV/bin/python"
if [[ -n "${P4A_DIR:-}" ]]; then
  SOURCE_P4A=$P4A_DIR
else
  SOURCE_P4A=$({ find "$DGFS_ROOT" -maxdepth 2 -type f -name P4A_SUCCESS -printf '%T@ %h\n' 2>/dev/null || true; } \
    | sort -nr | head -1 | cut -d' ' -f2-)
fi
[[ -d "$SOURCE_P4A" ]] || { echo P4B_P4A_SOURCE_NOT_FOUND; exit 2; }
test -s "$SOURCE_P4A/p4a_comparison.json"
for name in M16_raw M16_fplus M24_raw; do
  test -s "$SOURCE_P4A/run_$name/dist_p3b_$name-1.00.frfss"
done

mkdir -p "$RUN_DIR"
git clone --depth 1 --branch "$DGFS_REF" "$DGFS_REPO" "$SOLVER_SRC"
cp "$SOLVER_SRC/hpc/p4b_time_history.slurm" "$RUN_DIR/run.slurm"
JOB_ID=$(cd "$RUN_DIR" && sbatch --parsable \
  --export=ALL,DGFS_ROOT="$DGFS_ROOT",DGFS_ENV="$DGFS_ENV",DGFS_SOLVER_SRC="$SOLVER_SRC",P4A_DIR="$SOURCE_P4A",DGFS_REF="$DGFS_REF" \
  ./run.slurm)
echo "P4B_JOB_ID=${JOB_ID%%;*}"
echo "P4B_RUN_DIR=$RUN_DIR"
echo "P4B_SOURCE_P4A=$SOURCE_P4A"
echo P4B_BOOTSTRAP_COMPLETE
