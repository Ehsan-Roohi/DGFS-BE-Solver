#!/usr/bin/env bash
# Submit the steady-state closeout for the JCP Figure 14 novelty campaign.

set -Eeuo pipefail
trap 'rc=$?; echo "J14NOV_CLOSEOUT_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
REPO=${DGFS_REPO:-https://github.com/Ehsan-Roohi/DGFS-BE-Solver.git}
REF=${DGFS_REF:-agent/jcp14-conservative-novelty}

if [[ -n "${DGFS_NOV_SOURCE:-}" ]]; then
    SOURCE=$DGFS_NOV_SOURCE
else
    SOURCE=$(find "$ROOT" -maxdepth 1 -type d -name 'j14nov_*' -printf '%T@ %p\n' |
        sort -nr | while read -r _ d; do
            if [[ -e "$d/NOVELTY_COMPLETE" && -s "$d/results/novelty_report.json" ]]; then
                echo "$d"
                break
            fi
        done)
fi
[[ -d "${SOURCE:-}" ]] || { echo J14NOV_CLOSEOUT_SOURCE_NOT_FOUND; exit 2; }
[[ -e "$SOURCE/NOVELTY_COMPLETE" ]] || { echo J14NOV_CLOSEOUT_SOURCE_INCOMPLETE; exit 3; }
for name in M6_raw M6_fplus M16_raw M16_fplus; do
    test -s "$SOURCE/run_$name/dist_p3b_$name-160.25.frfss"
    test -s "$SOURCE/run_$name/bulksol_p3b_$name-160.25.frfss"
    test -s "$SOURCE/configs/p3b_$name.ini"
done
test -s "$SOURCE/audit/M6.json"
test -s "$SOURCE/audit/M16.json"

ACTIVE=$(squeue -h -u "$USER" -o '%j' 2>/dev/null | grep -E '^j14c-' | head -1 || true)
[[ -z "$ACTIVE" ]] || { echo "J14NOV_CLOSEOUT_ACTIVE_JOB=$ACTIVE"; exit 4; }

STAMP=$(date +%Y%m%d_%H%M%S)
CLOSEOUT="$ROOT/j14novclose_$STAMP"
SRC="$CLOSEOUT/src"
mkdir -p "$CLOSEOUT" "$SRC"
git -C "$SRC" init -q
git -C "$SRC" remote add origin "$REPO"
git -C "$SRC" fetch -q --depth=1 origin "$REF"
git -C "$SRC" checkout -q --detach FETCH_HEAD
COMMIT=$(git -C "$SRC" rev-parse HEAD)
test -x "$ENV_DIR/bin/python"
test -s "$SRC/hpc/run_unity_j14nov_closeout_segment.slurm"
test -s "$SRC/hpc/run_unity_j14nov_closeout_pack.slurm"

COMMON="ALL,DGFS_ROOT=$ROOT,DGFS_ENV=$ENV_DIR,DGFS_CLOSEOUT=$CLOSEOUT,DGFS_NOV_SOURCE=$SOURCE,DGFS_SOLVER_SRC=$SRC"
J1=$(cd "$CLOSEOUT" && sbatch --parsable --array=0-2 --job-name=j14c-s1 \
    --export="$COMMON,DGFS_SEGMENT=1,DGFS_TSTART=160.25,DGFS_TEND=168.25" \
    "$SRC/hpc/run_unity_j14nov_closeout_segment.slurm")
J1=${J1%%;*}
J2=$(cd "$CLOSEOUT" && sbatch --parsable --array=0-2 --job-name=j14c-s2 \
    --dependency="afterok:$J1" \
    --export="$COMMON,DGFS_SEGMENT=2,DGFS_TSTART=168.25,DGFS_TEND=176.25" \
    "$SRC/hpc/run_unity_j14nov_closeout_segment.slurm")
J2=${J2%%;*}
J3=$(cd "$CLOSEOUT" && sbatch --parsable --array=0-2 --job-name=j14c-s3 \
    --dependency="afterok:$J2" \
    --export="$COMMON,DGFS_SEGMENT=3,DGFS_TSTART=176.25,DGFS_TEND=180.25" \
    "$SRC/hpc/run_unity_j14nov_closeout_segment.slurm")
J3=${J3%%;*}
JP=$(cd "$CLOSEOUT" && sbatch --parsable --job-name=j14c-pack \
    --dependency="afterok:$J3" --export="$COMMON" \
    "$SRC/hpc/run_unity_j14nov_closeout_pack.slurm")
JP=${JP%%;*}

{
    echo "closeout=$CLOSEOUT"
    echo "source_campaign=$SOURCE"
    echo "stage1_job=$J1"
    echo "stage2_job=$J2"
    echo "stage3_job=$J3"
    echo "pack_job=$JP"
    echo "cases=M6_fplus,M16_raw,M16_fplus"
    echo "segments=160.25:168.25,168.25:176.25,176.25:180.25"
    echo "steady_threshold=1.0"
    echo "repo_ref=$REF"
    echo "repo_commit=$COMMIT"
    echo "submitted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$CLOSEOUT/SUBMISSION.txt"

echo "J14NOV_CLOSEOUT_DIR=$CLOSEOUT"
echo "J14NOV_CLOSEOUT_STAGE1_JOB=$J1"
echo "J14NOV_CLOSEOUT_STAGE2_JOB=$J2"
echo "J14NOV_CLOSEOUT_STAGE3_JOB=$J3"
echo "J14NOV_CLOSEOUT_PACK_JOB=$JP"
echo "J14NOV_CLOSEOUT_COMMIT=$COMMIT"
echo J14NOV_CLOSEOUT_SUBMITTED
