#!/usr/bin/env bash

set -Eeuo pipefail
trap 'rc=$?; echo "DGFS_KINETIC_BOOTSTRAP_FAILED rc=${rc} line=${LINENO}"; exit "$rc"' ERR

DGFS_ROOT="${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}"
DGFS_ENV="${DGFS_ENV:-$DGFS_ROOT/dgfs_py310}"
DGFS_REPO="${DGFS_REPO:-https://github.com/Ehsan-Roohi/DGFS-BE-Solver.git}"
DGFS_BRANCH="${DGFS_BRANCH:-agent/kinetic-diagnostics-v1}"
DGFS_SRC="${DGFS_SRC:-$DGFS_ROOT/DGFS-BE-Solver-kinetic-diagnostics-v1}"
BASELINE_RUN="${BASELINE_RUN:-$DGFS_ROOT/runs/normal_shock_M1p592_icfixed_20260817_041244}"

RESTART_NAME=dist_dgfs_2d_normalShock-6.0.frfss
MESH_NAME=mesh.frfsm
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$DGFS_ROOT/runs/normal_shock_M1p592_kinetic_smoke_$RUN_STAMP"

test -d "$DGFS_ROOT"
test -d "$DGFS_ENV"
test -s "$BASELINE_RUN/$MESH_NAME"
test -s "$BASELINE_RUN/$RESTART_NAME"

if [[ -d "$DGFS_SRC/.git" ]]; then
    if [[ -n "$(git -C "$DGFS_SRC" status --porcelain)" ]]; then
        echo "Existing kinetic checkout has local changes: $DGFS_SRC"
        exit 3
    fi

    git -C "$DGFS_SRC" fetch origin "$DGFS_BRANCH"
    git -C "$DGFS_SRC" checkout "$DGFS_BRANCH"
    git -C "$DGFS_SRC" merge --ff-only "origin/$DGFS_BRANCH"
elif [[ -e "$DGFS_SRC" ]]; then
    echo "DGFS_SRC exists but is not a git checkout: $DGFS_SRC"
    exit 4
else
    git clone --branch "$DGFS_BRANCH" --single-branch "$DGFS_REPO" "$DGFS_SRC"
fi

source /etc/profile.d/modules.sh 2>/dev/null || true
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$DGFS_ENV"
python -m pip install -e "$DGFS_SRC"

mkdir -p "$RUN_DIR"
cp "$BASELINE_RUN/$MESH_NAME" "$RUN_DIR/$MESH_NAME"
cp "$BASELINE_RUN/$RESTART_NAME" "$RUN_DIR/$RESTART_NAME"
cp "$DGFS_SRC/examples/normal_shock_M1p592_kinetic_diagnostics.ini" "$RUN_DIR/"
cp "$DGFS_SRC/examples/run_normal_shock_kinetic_smoke.slurm" "$RUN_DIR/"

cd "$RUN_DIR"
JOB_ID="$(sbatch --parsable \
    --export=ALL,DGFS_SRC="$DGFS_SRC",RESTART_FILE="$RESTART_NAME" \
    run_normal_shock_kinetic_smoke.slurm)"

printf 'DGFS_KINETIC_JOB_ID=%s\nDGFS_KINETIC_RUN_DIR=%s\n' \
    "$JOB_ID" "$RUN_DIR" | tee DGFS_KINETIC_JOB.env

echo "DGFS_KINETIC_BOOTSTRAP_COMPLETE"
echo "JOB_ID=$JOB_ID"
echo "RUN_DIR=$RUN_DIR"
