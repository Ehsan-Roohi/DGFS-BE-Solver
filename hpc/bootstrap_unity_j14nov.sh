#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "J14NOV_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
REPO=${DGFS_REPO:-https://github.com/Ehsan-Roohi/DGFS-BE-Solver.git}
REF=${DGFS_REF:-agent/jcp14-conservative-novelty}

if [[ -n "${DGFS_E8_SOURCE:-}" ]]; then
    E8=$DGFS_E8_SOURCE
else
    E8=$(find "$ROOT" -maxdepth 1 -type d -name 'j14e8_*' -printf '%T@ %p\n' |
        sort -nr | while read -r _ d; do
            if [[ -s "$d/SUBMISSION.txt" ]]; then echo "$d"; break; fi
        done)
fi
[[ -d "${E8:-}" ]] || { echo J14NOV_E8_CAMPAIGN_NOT_FOUND; exit 1; }
SOURCE_JOB=$(awk -F= '$1=="job"{print $2}' "$E8/SUBMISSION.txt")
[[ "$SOURCE_JOB" =~ ^[0-9]+$ ]] || { echo J14NOV_E8_JOB_NOT_FOUND; exit 2; }

DEP_ARGS=()
if [[ -e "$E8/run/CASE_SUCCESS" ]]; then
    echo "J14NOV_SOURCE_READY=$E8"
else
    STATE=$(sacct -j "$SOURCE_JOB" -X -n -o State 2>/dev/null | awk 'NF{print $1; exit}')
    case "$STATE" in
        FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL)
            echo "J14NOV_SOURCE_JOB_BAD_STATE=$STATE"
            exit 3
            ;;
        COMPLETED)
            echo J14NOV_SOURCE_COMPLETED_BUT_CASE_SUCCESS_MISSING
            exit 4
            ;;
    esac
    DEP_ARGS=(--dependency="afterok:$SOURCE_JOB")
    echo "J14NOV_WILL_WAIT_FOR_E8_JOB=$SOURCE_JOB state=${STATE:-UNKNOWN}"
fi

STAMP=$(date +%Y%m%d_%H%M%S)
CAMPAIGN="$ROOT/j14nov_$STAMP"
SRC="$CAMPAIGN/src"
mkdir -p "$CAMPAIGN"
git init -q "$SRC"
git -C "$SRC" remote add origin "$REPO"
git -C "$SRC" fetch --depth 1 origin "$REF"
git -C "$SRC" checkout -q --detach FETCH_HEAD
test -x "$ENV_DIR/bin/python"
test -s "$SRC/hpc/run_unity_j14nov.slurm"

JOB=$(cd "$CAMPAIGN" && sbatch --parsable "${DEP_ARGS[@]}" \
    --export=ALL,DGFS_ROOT="$ROOT",DGFS_ENV="$ENV_DIR",DGFS_NOV_CAMPAIGN="$CAMPAIGN",DGFS_SOLVER_SRC="$SRC",DGFS_E8_SOURCE="$E8" \
    "$SRC/hpc/run_unity_j14nov.slurm")
JOB=${JOB%%;*}
{
    echo "campaign=$CAMPAIGN"
    echo "job=$JOB"
    echo "source_campaign=$E8"
    echo "source_job=$SOURCE_JOB"
    echo "dependency=${DEP_ARGS[*]:-none}"
    echo "repo_ref=$REF"
    echo "repo_commit=$(git -C "$SRC" rev-parse HEAD)"
} > "$CAMPAIGN/SUBMISSION.txt"
echo "J14NOV_JOB_ID=$JOB"
echo "J14NOV_RUN_DIR=$CAMPAIGN"
echo "J14NOV_SOURCE_E8=$E8"
echo "J14NOV_DEPENDENCY=${DEP_ARGS[*]:-none}"
echo J14NOV_BOOTSTRAP_COMPLETE
