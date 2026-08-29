#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=dispatch-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "J14NOV_M16_STEADY_DISPATCH_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CLOSEOUT=${DGFS_CLOSEOUT:?DGFS_CLOSEOUT is required}
ORIGINAL=${DGFS_NOV_SOURCE:?DGFS_NOV_SOURCE is required}
SRC=${DGFS_SOLVER_SRC:?DGFS_SOLVER_SRC is required}
STEADY=${DGFS_STEADY_DIR:?DGFS_STEADY_DIR is required}

for name in M16_raw M16_fplus; do
    test -e "$STEADY/stage_1/$name/SEGMENT_SUCCESS"
done

COMMON="ALL,DGFS_ROOT=$ROOT,DGFS_ENV=$ENV_DIR,DGFS_CLOSEOUT=$CLOSEOUT,DGFS_NOV_SOURCE=$ORIGINAL,DGFS_SOLVER_SRC=$SRC,DGFS_STEADY_DIR=$STEADY"
J2=$(cd "$STEADY" && sbatch --parsable --array=0-1 --job-name=j14s-s2 \
    --export="$COMMON,DGFS_SEGMENT=2,DGFS_TSTART=360.25,DGFS_TEND=540.25" \
    "$SRC/hpc/run_unity_j14nov_m16_steady_segment.slurm")
J2=${J2%%;*}
JP=$(cd "$STEADY" && sbatch --parsable --job-name=j14s-pack \
    --dependency="afterok:$J2" --export="$COMMON" \
    "$SRC/hpc/run_unity_j14nov_m16_steady_pack.slurm")
JP=${JP%%;*}

{
    echo "stage2_job=$J2"
    echo "pack_job=$JP"
    echo "dispatched_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$STEADY/SUBMISSION.txt"

echo "J14NOV_M16_STEADY_STAGE2_JOB=$J2"
echo "J14NOV_M16_STEADY_PACK_JOB=$JP"
echo J14NOV_M16_STEADY_DISPATCH_COMPLETE
