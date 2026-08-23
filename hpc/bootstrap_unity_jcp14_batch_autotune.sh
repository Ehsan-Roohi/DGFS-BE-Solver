#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "JCP14_AUTOTUNE_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
REPO=${DGFS_REPO:-https://github.com/Ehsan-Roohi/DGFS-BE-Solver.git}
REF=${DGFS_REF:-agent/jcp14-batch-autotune}
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$ROOT/jcp14_tune_$STAMP"
SRC="$RUN_DIR/src"

command -v git >/dev/null
command -v sbatch >/dev/null
test -x "$ENV_DIR/bin/python"
mkdir -p "$RUN_DIR"
git clone --depth 1 --branch "$REF" "$REPO" "$SRC"
PYTHONPATH="$SRC" "$ENV_DIR/bin/python" "$SRC/cases/jcp2019_fig14b_validation/verify_case.py"
PYTHONPATH="$SRC" "$ENV_DIR/bin/python" -m py_compile "$SRC/hpc/verify_batch_equivalence.py"

JOB=$(cd "$RUN_DIR" && sbatch --parsable \
    --export=ALL,DGFS_ROOT="$ROOT",DGFS_ENV="$ENV_DIR",DGFS_AUTOTUNE_DIR="$RUN_DIR",DGFS_SOLVER_SRC="$SRC" \
    "$SRC/hpc/run_unity_jcp14_batch_autotune.slurm")
JOB=${JOB%%;*}
{
    echo "run_dir=$RUN_DIR"
    echo "job=$JOB"
    echo "repo_ref=$REF"
    echo "repo_commit=$(git -C "$SRC" rev-parse HEAD)"
    echo "batches=64,256,1024"
    echo "benchmark=JCP_Figure14_4element_t0_to_t0.1"
} > "$RUN_DIR/SUBMISSION.txt"
echo "JCP14_AUTOTUNE_JOB_ID=$JOB"
echo "JCP14_AUTOTUNE_RUN_DIR=$RUN_DIR"
echo JCP14_AUTOTUNE_BOOTSTRAP_COMPLETE
