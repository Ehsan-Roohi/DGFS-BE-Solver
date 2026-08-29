#!/usr/bin/env bash
# Continue only M16 raw/fplus from the completed J14 closeout package.

set -Eeuo pipefail
trap 'rc=$?; echo "J14NOV_M16_STEADY_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
REPO=${DGFS_REPO:-https://github.com/Ehsan-Roohi/DGFS-BE-Solver.git}
REF=${DGFS_REF:-agent/jcp14-conservative-novelty}

if [[ -n "${DGFS_CLOSEOUT:-}" ]]; then
    CLOSEOUT=$DGFS_CLOSEOUT
else
    CLOSEOUT=$(find "$ROOT" -maxdepth 1 -type d -name 'j14novclose_*' -printf '%T@ %p\n' |
        sort -nr | while read -r _ d; do
            if [[ -e "$d/CLOSEOUT_COMPLETE" && -s "$d/results/CLOSEOUT_STATUS.json" ]]; then
                echo "$d"
                break
            fi
        done)
fi
[[ -d "${CLOSEOUT:-}" ]] || { echo J14NOV_M16_STEADY_CLOSEOUT_NOT_FOUND; exit 2; }
test -e "$CLOSEOUT/CLOSEOUT_COMPLETE"
test -s "$CLOSEOUT/SUBMISSION.txt"
test -x "$ENV_DIR/bin/python"

ORIGINAL=$(awk -F= '$1=="source_campaign"{print $2; exit}' "$CLOSEOUT/SUBMISSION.txt")
[[ -d "${ORIGINAL:-}" ]] || { echo J14NOV_M16_STEADY_SOURCE_NOT_FOUND; exit 3; }
test -s "$ORIGINAL/input/dgfs.ini"
test -s "$ORIGINAL/input/mesh.frfsm"
test -s "$ORIGINAL/input/dist-source.frfss"
test -s "$ORIGINAL/input/kinetic_residual.csv"

# M6 is already steady and is never submitted again.
for name in M6_raw M6_fplus; do
    test -e "$CLOSEOUT/final_runs/run_$name/CONVERGED"
    test -s "$CLOSEOUT/final_runs/run_$name/kinetic_residual_p3b.csv"
done
for M in M6 M16; do
    test -s "$CLOSEOUT/audit/$M.json"
    test -s "$CLOSEOUT/audit/$M.csv"
    test -s "$CLOSEOUT/audit/$M.log"
done

for name in M16_raw M16_fplus; do
    d="$CLOSEOUT/final_runs/run_$name"
    test -s "$d/dist_p3b_$name-180.25.frfss"
    test -s "$d/bulksol_p3b_$name-180.25.frfss"
    test -s "$d/p3b_$name.ini"
    test -s "$d/mesh.frfsm"
    test -s "$d/kinetic_residual_p3b.csv"
done

ACTIVE=$(squeue -h -u "$USER" -o '%j' 2>/dev/null | grep -E '^j14s-' | head -1 || true)
[[ -z "$ACTIVE" ]] || { echo "J14NOV_M16_STEADY_ACTIVE_JOB=$ACTIVE"; exit 4; }

STAMP=$(date +%Y%m%d_%H%M%S)
STEADY="$CLOSEOUT/m16_steady_$STAMP"
SRC="$STEADY/src"
mkdir -p "$STEADY" "$SRC"
git -C "$SRC" init -q
git -C "$SRC" remote add origin "$REPO"
git -C "$SRC" fetch -q --depth=1 origin "$REF"
git -C "$SRC" checkout -q --detach FETCH_HEAD
COMMIT=$(git -C "$SRC" rev-parse HEAD)
test -s "$SRC/hpc/run_unity_j14nov_m16_steady_segment.slurm"
test -s "$SRC/hpc/run_unity_j14nov_m16_steady_dispatch.slurm"
test -s "$SRC/hpc/run_unity_j14nov_m16_steady_pack.slurm"

# Install the opt-in conservative collision hook in this detached checkout.
HOOK="$SRC/cases/jcp2019_fig14b_normal_shock/solver_hook/apply_hook.py"
test -s "$HOOK"
"$ENV_DIR/bin/python" "$HOOK" "$SRC"
grep -q 'self.projector' "$SRC/frfs/solvers/dgfs/system.py"

COMMON="ALL,DGFS_ROOT=$ROOT,DGFS_ENV=$ENV_DIR,DGFS_CLOSEOUT=$CLOSEOUT,DGFS_NOV_SOURCE=$ORIGINAL,DGFS_SOLVER_SRC=$SRC,DGFS_STEADY_DIR=$STEADY"
J1=$(cd "$STEADY" && sbatch --parsable --array=0-1 --job-name=j14s-s1 \
    --export="$COMMON,DGFS_SEGMENT=1,DGFS_TSTART=180.25,DGFS_TEND=360.25" \
    "$SRC/hpc/run_unity_j14nov_m16_steady_segment.slurm")
J1=${J1%%;*}

{
    echo "steady_dir=$STEADY"
    echo "closeout=$CLOSEOUT"
    echo "source_campaign=$ORIGINAL"
    echo "stage1_job=$J1"
    echo "cases=M16_raw,M16_fplus"
    echo "segments=180.25:360.25,360.25:540.25"
    echo "steady_threshold=1.0"
    echo "repo_ref=$REF"
    echo "repo_commit=$COMMIT"
    echo "submitted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$STEADY/SUBMISSION.txt"

# Delay stage 2 submission until stage 1 releases its two QOS job slots.
JD=$(cd "$STEADY" && sbatch --parsable --job-name=j14s-next \
    --dependency="afterok:$J1" --export="$COMMON" \
    "$SRC/hpc/run_unity_j14nov_m16_steady_dispatch.slurm")
JD=${JD%%;*}
echo "dispatcher_job=$JD" >> "$STEADY/SUBMISSION.txt"

echo "J14NOV_M16_STEADY_DIR=$STEADY"
echo "J14NOV_M16_STEADY_STAGE1_JOB=$J1"
echo "J14NOV_M16_STEADY_DISPATCHER_JOB=$JD"
echo "J14NOV_M16_STEADY_COMMIT=$COMMIT"
echo J14NOV_M16_STEADY_SUBMITTED
