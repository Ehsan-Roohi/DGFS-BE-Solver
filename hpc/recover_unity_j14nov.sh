#!/usr/bin/env bash
# Resume the failed novelty campaign after the unsupported M_omega=12 setup.
# The successful M_omega=6 collision audit is reused; M_omega=16 is computed.

set -Eeuo pipefail
trap 'rc=$?; echo "J14NOV_RECOVERY_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
REPO=${DGFS_REPO:-https://github.com/Ehsan-Roohi/DGFS-BE-Solver.git}
REF=${DGFS_REF:-agent/jcp14-conservative-novelty}

CAMPAIGN=$(find "$ROOT" -maxdepth 1 -type d -name 'j14nov_*' -printf '%T@ %p\n' \
    | sort -nr | head -1 | cut -d' ' -f2-)
[[ -d "$CAMPAIGN" ]] || { echo J14NOV_CAMPAIGN_NOT_FOUND; exit 2; }
[[ -s "$CAMPAIGN/SUBMISSION.txt" ]] || { echo J14NOV_SUBMISSION_NOT_FOUND; exit 3; }

SOURCE=$(awk -F= '$1=="source_campaign"{print substr($0,index($0,"=")+1)}' \
    "$CAMPAIGN/SUBMISSION.txt" | tail -1)
[[ -d "$SOURCE/run" && -e "$SOURCE/run/CASE_SUCCESS" ]] || {
    echo "J14NOV_E8_SOURCE_NOT_VERIFIED=$SOURCE"
    exit 4
}
[[ -s "$CAMPAIGN/audit/M6.json" && -s "$CAMPAIGN/audit/M6.csv" ]] || {
    echo J14NOV_M6_AUDIT_NOT_REUSABLE
    exit 5
}
grep -q 'J14NOV_COLLISION_COMPLETE M=6' "$CAMPAIGN/audit/M6.log" || {
    echo J14NOV_M6_AUDIT_LOG_NOT_COMPLETE
    exit 6
}

ACTIVE=$(squeue -h -u "$USER" -n j14-nov -o '%A' 2>/dev/null | head -1 || true)
[[ -z "$ACTIVE" ]] || { echo "J14NOV_ACTIVE_JOB_ALREADY_EXISTS=$ACTIVE"; exit 7; }

STAMP=$(date +%Y%m%d_%H%M%S)
SRC="$CAMPAIGN/src_recovery_$STAMP"
mkdir -p "$SRC"
git -C "$SRC" init -q
git -C "$SRC" remote add origin "$REPO"
git -C "$SRC" fetch -q --depth=1 origin "$REF"
git -C "$SRC" checkout -q --detach FETCH_HEAD
COMMIT=$(git -C "$SRC" rev-parse HEAD)

JOB=$(cd "$CAMPAIGN" && sbatch --parsable \
    --export=ALL,DGFS_ROOT="$ROOT",DGFS_ENV="$ENV_DIR",DGFS_NOV_CAMPAIGN="$CAMPAIGN",DGFS_SOLVER_SRC="$SRC",DGFS_E8_SOURCE="$SOURCE" \
    "$SRC/hpc/run_unity_j14nov.slurm")
JOB=${JOB%%;*}

{
    echo "campaign=$CAMPAIGN"
    echo "job=$JOB"
    echo "source_campaign=$SOURCE"
    echo "reused_audit=$CAMPAIGN/audit/M6.json"
    echo "repo_ref=$REF"
    echo "repo_commit=$COMMIT"
    echo "submitted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$CAMPAIGN/RECOVERY_SUBMISSION_$JOB.txt"

echo "J14NOV_RECOVERY_JOB_ID=$JOB"
echo "J14NOV_RECOVERY_CAMPAIGN=$CAMPAIGN"
echo "J14NOV_REUSED_M6_AUDIT=$CAMPAIGN/audit/M6.json"
echo "J14NOV_RECOVERY_COMMIT=$COMMIT"
echo J14NOV_RECOVERY_SUBMITTED
