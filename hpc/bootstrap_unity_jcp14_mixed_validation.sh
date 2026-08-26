#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "J14_VALIDATION_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
REPO=${DGFS_REPO:-https://github.com/Ehsan-Roohi/DGFS-BE-Solver.git}
REF=${DGFS_REF:-agent/jcp14-2gpu-checkpoint-continuation}
SOURCE=${DGFS_JCP14_SOURCE:-$ROOT/jcp14_20260822_164418}
STAMP=$(date +%Y%m%d_%H%M%S)
CAMPAIGN="$ROOT/j14val_$STAMP"
SRC="$CAMPAIGN/src"
STAGE="$CAMPAIGN/input"

command -v git >/dev/null
command -v sbatch >/dev/null
test -x "$ENV_DIR/bin/python"
test -e "$SOURCE/e4/CASE_SUCCESS"
test -s "$SOURCE/e4/dgfs.ini"
test -s "$SOURCE/e4/mesh.frfsm"
test -s "$SOURCE/e4/bulk-final.frfss"
test -s "$SOURCE/e4/kinetic_residual.csv"
test -s "$SOURCE/e8/dgfs.ini"
test -s "$SOURCE/e8/mesh.frfsm"
test -s "$SOURCE/e8/bulk-140.0.frfss"
test -s "$SOURCE/e8/kinetic_residual.csv"

mkdir -p "$STAGE/e4" "$STAGE/e8"
git init -q "$SRC"
git -C "$SRC" remote add origin "$REPO"
git -C "$SRC" fetch --depth 1 origin "$REF"
git -C "$SRC" checkout -q --detach FETCH_HEAD
cp "$SOURCE/e4/dgfs.ini" "$STAGE/e4/dgfs.ini"
cp "$SOURCE/e4/mesh.frfsm" "$STAGE/e4/mesh.frfsm"
cp "$SOURCE/e4/bulk-final.frfss" "$STAGE/e4/bulk-final.frfss"
cp "$SOURCE/e8/dgfs.ini" "$STAGE/e8/dgfs.ini"
cp "$SOURCE/e8/mesh.frfsm" "$STAGE/e8/mesh.frfsm"
cp "$SOURCE/e8/bulk-140.0.frfss" "$STAGE/e8/bulk-final.frfss"

JOB=$(cd "$CAMPAIGN" && sbatch --parsable \
    --export=ALL,DGFS_ROOT="$ROOT",DGFS_ENV="$ENV_DIR",DGFS_VALIDATION_CAMPAIGN="$CAMPAIGN",DGFS_JCP14_SOURCE="$SOURCE",DGFS_SOLVER_SRC="$SRC" \
    "$SRC/hpc/run_unity_jcp14_mixed_validation.slurm")
JOB=${JOB%%;*}
{
    echo "campaign=$CAMPAIGN"
    echo "job=$JOB"
    echo "source=$SOURCE"
    echo "grid4_source=bulk-final.frfss"
    echo "grid8_source=bulk-140.0.frfss"
    echo "repo_ref=$REF"
    echo "repo_commit=$(git -C "$SRC" rev-parse HEAD)"
} > "$CAMPAIGN/SUBMISSION.txt"
echo "J14_VALIDATION_JOB_ID=$JOB"
echo "J14_VALIDATION_RUN_DIR=$CAMPAIGN"
echo "J14_VALIDATION_BOOTSTRAP_COMPLETE"
