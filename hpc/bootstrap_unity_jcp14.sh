#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "JCP14_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

DGFS_ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
DGFS_ENV=${DGFS_ENV:-$DGFS_ROOT/dgfs_py310}
DGFS_REPO=${DGFS_REPO:-https://github.com/Ehsan-Roohi/DGFS-BE-Solver.git}
DGFS_REF=${DGFS_REF:-agent/jcp2019-fig14b-ohwada-validation}
DGFS_MAX_SEGMENTS=${DGFS_MAX_SEGMENTS:-24}
STAMP=$(date +%Y%m%d_%H%M%S)
CAMPAIGN="$DGFS_ROOT/jcp14_$STAMP"
SOLVER_SRC="$CAMPAIGN/src"
CASE="$SOLVER_SRC/cases/jcp2019_fig14b_validation"

command -v git >/dev/null
command -v sbatch >/dev/null
test -x "$DGFS_ENV/bin/python"
mkdir -p "$CAMPAIGN"
git clone --depth 1 --branch "$DGFS_REF" "$DGFS_REPO" "$SOLVER_SRC"

PYTHONPATH="$SOLVER_SRC" "$DGFS_ENV/bin/python" "$CASE/verify_case.py"
PYTHONPATH="$SOLVER_SRC" "$DGFS_ENV/bin/python" -c \
  'from frfs.backends import BaseBackend; from frfs.readers.native import NativeReader; print("DGFS_RUNTIME_PASS")'

for N in 4 8; do
    mkdir -p "$CAMPAIGN/e${N}"
    cp "$CASE/dgfs.ini" "$CAMPAIGN/e${N}/"
    cp "$CASE/mesh_${N}e.msh" "$CAMPAIGN/e${N}/mesh.msh"
done

ARRAY_JOB=$(cd "$CAMPAIGN" && sbatch --parsable --array=4,8%2 \
    --export=ALL,DGFS_ROOT="$DGFS_ROOT",DGFS_ENV="$DGFS_ENV",DGFS_JCP14_CAMPAIGN="$CAMPAIGN",DGFS_SOLVER_SRC="$SOLVER_SRC",DGFS_MAX_SEGMENTS="$DGFS_MAX_SEGMENTS" \
    "$SOLVER_SRC/hpc/run_unity_jcp14_case.slurm")
ARRAY_JOB=${ARRAY_JOB%%;*}
PACK_JOB=$(cd "$CAMPAIGN" && sbatch --parsable --dependency="afterok:${ARRAY_JOB}" \
    --export=ALL,DGFS_ROOT="$DGFS_ROOT",DGFS_ENV="$DGFS_ENV",DGFS_JCP14_CAMPAIGN="$CAMPAIGN",DGFS_SOLVER_SRC="$SOLVER_SRC" \
    "$SOLVER_SRC/hpc/pack_unity_jcp14.slurm")
PACK_JOB=${PACK_JOB%%;*}

{
    echo "campaign=$CAMPAIGN"
    echo "array_job=$ARRAY_JOB"
    echo "pack_job=$PACK_JOB"
    echo "repo_ref=$DGFS_REF"
    echo "repo_commit=$(git -C "$SOLVER_SRC" rev-parse HEAD)"
    echo "case=Mach_1.59_JCP_Figure_14_only"
} > "$CAMPAIGN/SUBMISSION.txt"
echo "JCP14_ARRAY_JOB_ID=$ARRAY_JOB"
echo "JCP14_PACK_JOB_ID=$PACK_JOB"
echo "JCP14_CAMPAIGN=$CAMPAIGN"
echo "JCP14_BOOTSTRAP_COMPLETE"
