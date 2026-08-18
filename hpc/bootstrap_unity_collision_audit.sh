#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "COLLISION_AUDIT_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

DGFS_ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
DGFS_ENV=${DGFS_ENV:-$DGFS_ROOT/dgfs_py310}
DGFS_REPO=${DGFS_REPO:-https://github.com/Ehsan-Roohi/DGFS-BE-Solver.git}
DGFS_REF=${DGFS_REF:-agent/phase2-collision-audit}
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$DGFS_ROOT/p2_$STAMP"
SOLVER_SRC="$RUN_DIR/source/DGFS-BE-Solver"
CASE_REL=cases/jcp2019_fig14b_normal_shock

command -v git >/dev/null
command -v sbatch >/dev/null
test -x "$DGFS_ENV/bin/python"

if [[ -n "${DGFS_SNAPSHOT:-}" ]]; then
    SNAPSHOT=$DGFS_SNAPSHOT
else
    SNAPSHOT=$({ find "$DGFS_ROOT/runs" -maxdepth 2 -type f \
        -name 'dist_dgfs_fig14b-30.0.frfss' -printf '%T@ %p\n' 2>/dev/null \
        || true; } | sort -nr | head -1 | cut -d' ' -f2-)
    if [[ -z "$SNAPSHOT" ]]; then
        SNAPSHOT=$({ find "$DGFS_ROOT/runs" -maxdepth 2 -type f \
            -name 'dist_dgfs_fig14b-*.frfss' -printf '%T@ %p\n' 2>/dev/null \
            || true; } | sort -nr | head -1 | cut -d' ' -f2-)
    fi
fi
[[ -s "$SNAPSHOT" ]] || { echo "DGFS_SNAPSHOT_NOT_FOUND"; exit 2; }
SNAPSHOT_DIR=$(dirname "$SNAPSHOT")
[[ -s "$SNAPSHOT_DIR/mesh.frfsm" ]] || { echo "DGFS_MESH_NOT_FOUND"; exit 3; }
[[ -s "$SNAPSHOT_DIR/dgfs_fig14b.ini" ]] || { echo "DGFS_CONFIG_NOT_FOUND"; exit 4; }

mkdir -p "$RUN_DIR/source"
git clone --depth 1 --branch "$DGFS_REF" "$DGFS_REPO" "$SOLVER_SRC"
cp "$SNAPSHOT" "$RUN_DIR/snapshot.frfss"
cp "$SNAPSHOT_DIR/mesh.frfsm" "$RUN_DIR/mesh.frfsm"
cp "$SNAPSHOT_DIR/dgfs_fig14b.ini" "$RUN_DIR/case.ini"
cp "$SOLVER_SRC/$CASE_REL/audit_collision.py" "$RUN_DIR/"
cp "$SOLVER_SRC/$CASE_REL/collision_audit.slurm" "$RUN_DIR/"
printf 'source_snapshot=%s\n' "$SNAPSHOT" > "$RUN_DIR/SNAPSHOT_SOURCE.txt"
chmod +x "$RUN_DIR/audit_collision.py" "$RUN_DIR/collision_audit.slurm"

JOB_ID=$(cd "$RUN_DIR" && sbatch --parsable \
    --export=ALL,DGFS_ROOT="$DGFS_ROOT",DGFS_ENV="$DGFS_ENV",DGFS_SOLVER_SRC="$SOLVER_SRC",DGFS_COLLISION_REPEATS="${DGFS_COLLISION_REPEATS:-3}",DGFS_COLLISION_MAX_POINTS="${DGFS_COLLISION_MAX_POINTS:-0}",DGFS_COLLISION_TOLERANCE="${DGFS_COLLISION_TOLERANCE:-1e-8}" \
    ./collision_audit.slurm)
JOB_ID=${JOB_ID%%;*}

echo "COLLISION_AUDIT_JOB_ID=$JOB_ID"
echo "COLLISION_AUDIT_RUN_DIR=$RUN_DIR"
echo "COLLISION_AUDIT_SOURCE_SNAPSHOT=$SNAPSHOT"
echo "COLLISION_AUDIT_BOOTSTRAP_COMPLETE"
