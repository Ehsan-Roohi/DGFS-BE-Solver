#!/usr/bin/env bash
# Recover only the two timed-out M16 closeout cases and then package all cases.

set -Eeuo pipefail
trap 'rc=$?; echo "J14NOV_M16_RECOVERY_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
REPO=${DGFS_REPO:-https://github.com/Ehsan-Roohi/DGFS-BE-Solver.git}
REF=${DGFS_REF:-agent/jcp14-conservative-novelty}

if [[ -n "${DGFS_CLOSEOUT:-}" ]]; then
    CLOSEOUT=$DGFS_CLOSEOUT
else
    CLOSEOUT=$(find "$ROOT" -maxdepth 1 -type d -name 'j14novclose_*' -printf '%T@ %p\n' |
        sort -nr | while read -r _ d; do
            if [[ -s "$d/SUBMISSION.txt" && ! -e "$d/CLOSEOUT_COMPLETE" ]]; then
                echo "$d"
                break
            fi
        done)
fi
[[ -d "${CLOSEOUT:-}" ]] || { echo J14NOV_M16_RECOVERY_CLOSEOUT_NOT_FOUND; exit 2; }
test -s "$CLOSEOUT/SUBMISSION.txt"

ORIGINAL=$(awk -F= '$1=="source_campaign"{print $2; exit}' "$CLOSEOUT/SUBMISSION.txt")
[[ -d "${ORIGINAL:-}" ]] || { echo J14NOV_M16_RECOVERY_SOURCE_NOT_FOUND; exit 3; }
test -x "$ENV_DIR/bin/python"

# The completed M6-fplus result is preserved and reused by the existing packer.
test -e "$CLOSEOUT/stage_1/M6_fplus/SEGMENT_SUCCESS"
test -e "$CLOSEOUT/stage_1/M6_fplus/CONVERGED"
test -s "$CLOSEOUT/stage_1/M6_fplus/dist_p3b_M6_fplus-168.25.frfss"
test -s "$CLOSEOUT/stage_1/M6_fplus/bulksol_p3b_M6_fplus-168.25.frfss"

for name in M16_raw M16_fplus; do
    test -s "$ORIGINAL/run_$name/dist_p3b_$name-160.25.frfss"
    test -s "$ORIGINAL/configs/p3b_$name.ini"
    if [[ -e "$CLOSEOUT/stage_3/$name/SEGMENT_SUCCESS" ]]; then
        echo "J14NOV_M16_RECOVERY_ALREADY_COMPLETE=$name"
        exit 4
    fi
done
test -s "$ORIGINAL/input/mesh.frfsm"
test -s "$ORIGINAL/input/kinetic_residual.csv"

ACTIVE=$(squeue -h -u "$USER" -o '%j' 2>/dev/null | grep -E '^(j14c-m16|j14c-pack)$' | head -1 || true)
[[ -z "$ACTIVE" ]] || { echo "J14NOV_M16_RECOVERY_ACTIVE_JOB=$ACTIVE"; exit 5; }

STAMP=$(date +%Y%m%d_%H%M%S)
SRC="$CLOSEOUT/src_m16_recovery_$STAMP"
mkdir -p "$SRC"
git -C "$SRC" init -q
git -C "$SRC" remote add origin "$REPO"
git -C "$SRC" fetch -q --depth=1 origin "$REF"
git -C "$SRC" checkout -q --detach FETCH_HEAD
COMMIT=$(git -C "$SRC" rev-parse HEAD)
test -s "$SRC/hpc/run_unity_j14nov_closeout_m16_recovery.slurm"
test -s "$SRC/hpc/run_unity_j14nov_closeout_pack.slurm"

# Apply the opt-in conservative collision hook once, before the array starts.
"$ENV_DIR/bin/python" \
    "$SRC/cases/jcp2019_fig14b_normal_shock/novelty/solver_hook/apply_hook.py" "$SRC"
grep -q 'DGFSCollisionInvariantProjector' "$SRC/frfs/solvers/dgfs/scattering.py"

COMMON="ALL,DGFS_ROOT=$ROOT,DGFS_ENV=$ENV_DIR,DGFS_CLOSEOUT=$CLOSEOUT,DGFS_NOV_SOURCE=$ORIGINAL,DGFS_SOLVER_SRC=$SRC"
JR=$(cd "$CLOSEOUT" && sbatch --parsable --array=0-1 --job-name=j14c-m16 \
    --export="$COMMON" "$SRC/hpc/run_unity_j14nov_closeout_m16_recovery.slurm")
JR=${JR%%;*}
JP=$(cd "$CLOSEOUT" && sbatch --parsable --job-name=j14c-pack \
    --dependency="afterok:$JR" --export="$COMMON" \
    "$SRC/hpc/run_unity_j14nov_closeout_pack.slurm")
JP=${JP%%;*}

{
    echo "m16_recovery_job=$JR"
    echo "final_pack_job=$JP"
    echo "m16_recovery_source=$SRC"
    echo "m16_recovery_ref=$REF"
    echo "m16_recovery_commit=$COMMIT"
    echo "m16_recovery_submitted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$CLOSEOUT/SUBMISSION.txt"

echo "J14NOV_M16_RECOVERY_CLOSEOUT=$CLOSEOUT"
echo "J14NOV_M16_RECOVERY_JOB=$JR"
echo "J14NOV_M16_RECOVERY_PACK_JOB=$JP"
echo "J14NOV_M16_RECOVERY_COMMIT=$COMMIT"
echo J14NOV_M16_RECOVERY_SUBMITTED
