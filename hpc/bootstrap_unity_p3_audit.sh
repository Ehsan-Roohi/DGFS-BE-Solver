#!/usr/bin/env bash
# One-line Unity bootstrap for the phase-3a collision audit.
set -Eeuo pipefail
trap 'rc=$?; echo "P3A_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

DGFS_ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
DGFS_ENV=${DGFS_ENV:-$DGFS_ROOT/dgfs_py310}
DGFS_REPO=${DGFS_REPO:-https://github.com/Ehsan-Roohi/DGFS-BE-Solver.git}
DGFS_REF=${DGFS_REF:-agent/phase3-angular-conservative-audit}
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$DGFS_ROOT/p3a_$STAMP"
SOLVER_SRC="$RUN_DIR/src"

test -x "$DGFS_ENV/bin/python"
if [[ -n "${DGFS_SNAPSHOT:-}" ]]; then
    SNAPSHOT=$DGFS_SNAPSHOT
else
    SNAPSHOT=$({ find "$DGFS_ROOT/runs" -maxdepth 2 -type f \
        -name 'dist_dgfs_fig14b-30.0.frfss' -printf '%T@ %p\n' 2>/dev/null || true; } \
        | sort -nr | head -1 | cut -d' ' -f2-)
fi
[[ -s "$SNAPSHOT" ]] || { echo "P3A_SNAPSHOT_NOT_FOUND"; exit 2; }
SNAPSHOT_DIR=$(dirname "$SNAPSHOT")
test -s "$SNAPSHOT_DIR/mesh.frfsm"
test -s "$SNAPSHOT_DIR/dgfs_fig14b.ini"

mkdir -p "$RUN_DIR"
git clone --depth 1 --branch "$DGFS_REF" "$DGFS_REPO" "$SOLVER_SRC"
PKG="$SOLVER_SRC/cases/jcp2019_fig14b_normal_shock/phase3"
test -s "$PKG/audit_quadrature_projection.py"
cp "$SNAPSHOT" "$RUN_DIR/snapshot.frfss"
cp "$SNAPSHOT_DIR/mesh.frfsm" "$RUN_DIR/mesh.frfsm"
cp "$SNAPSHOT_DIR/dgfs_fig14b.ini" "$RUN_DIR/case.ini"
cp -r "$PKG" "$RUN_DIR/p3"
cp "$SOLVER_SRC/hpc/p3_quadrature_projection_audit.slurm" "$RUN_DIR/run.slurm"
printf 'source_snapshot=%s\nsource_ref=%s\n' "$SNAPSHOT" "$DGFS_REF" > "$RUN_DIR/INPUT.txt"

JOB_ID=$(cd "$RUN_DIR" && sbatch --parsable \
    --export=ALL,DGFS_ROOT="$DGFS_ROOT",DGFS_ENV="$DGFS_ENV",DGFS_SOLVER_SRC="$SOLVER_SRC" \
    ./run.slurm)
echo "P3A_JOB_ID=${JOB_ID%%;*}"
echo "P3A_RUN_DIR=$RUN_DIR"
echo "P3A_BOOTSTRAP_COMPLETE"
