#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "JCP14_MPI_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
REPO=${DGFS_REPO:-https://github.com/Ehsan-Roohi/DGFS-BE-Solver.git}
REF=${DGFS_REF:-agent/jcp14-2gpu-checkpoint-continuation}
SOURCE_CAMPAIGN=${DGFS_JCP14_SOURCE:-$ROOT/jcp14_20260822_164418}
MAX_SEGMENTS=${DGFS_MAX_SEGMENTS:-24}
STAMP=$(date +%Y%m%d_%H%M%S)
CAMPAIGN="$ROOT/jcp14_mpi_$STAMP"
SRC="$CAMPAIGN/src"

command -v git >/dev/null
command -v sbatch >/dev/null
test -x "$ENV_DIR/bin/python"
test -d "$SOURCE_CAMPAIGN/e4"
test -d "$SOURCE_CAMPAIGN/e8"
mkdir -p "$CAMPAIGN"
git clone --depth 1 --branch "$REF" "$REPO" "$SRC"

PYTHONPATH="$SRC" "$ENV_DIR/bin/python" \
    "$SRC/cases/jcp2019_fig14b_validation/verify_case.py"
PYTHONPATH="$SRC" "$ENV_DIR/bin/python" -m py_compile \
    "$SRC/hpc/snapshot_jcp14_checkpoint.py" \
    "$SRC/hpc/partition_line_checkpoint.py" \
    "$SRC/hpc/combine_native_moments.py" \
    "$SRC/hpc/make_jcp14_continuation_config.py" \
    "$SRC/hpc/verify_jcp14_mpi_equivalence.py"

for N in 4 8; do
    mkdir -p "$CAMPAIGN/e$N"
    PYTHONPATH="$SRC" "$ENV_DIR/bin/python" \
        "$SRC/hpc/snapshot_jcp14_checkpoint.py" \
        --source "$SOURCE_CAMPAIGN/e$N" --output "$CAMPAIGN/e$N"
done

ARRAY_JOB=$(cd "$CAMPAIGN" && sbatch --parsable --array=4,8%2 \
    --export=ALL,DGFS_ROOT="$ROOT",DGFS_ENV="$ENV_DIR",DGFS_MPI_CAMPAIGN="$CAMPAIGN",DGFS_SOLVER_SRC="$SRC",DGFS_MAX_SEGMENTS="$MAX_SEGMENTS" \
    "$SRC/hpc/run_unity_jcp14_2gpu.slurm")
ARRAY_JOB=${ARRAY_JOB%%;*}
PACK_JOB=$(cd "$CAMPAIGN" && sbatch --parsable --dependency="afterok:$ARRAY_JOB" \
    --export=ALL,DGFS_ROOT="$ROOT",DGFS_ENV="$ENV_DIR",DGFS_MPI_CAMPAIGN="$CAMPAIGN",DGFS_SOLVER_SRC="$SRC" \
    "$SRC/hpc/pack_unity_jcp14_2gpu.slurm")
PACK_JOB=${PACK_JOB%%;*}

{
    echo "campaign=$CAMPAIGN"
    echo "source_campaign=$SOURCE_CAMPAIGN"
    echo "array_job=$ARRAY_JOB"
    echo "pack_job=$PACK_JOB"
    echo "repo_ref=$REF"
    echo "repo_commit=$(git -C "$SRC" rev-parse HEAD)"
    echo "velocity_batch=256"
    echo "mpi_ranks=2"
    echo "benchmark_span=0.5"
} > "$CAMPAIGN/SUBMISSION.txt"

echo "JCP14_MPI_ARRAY_JOB_ID=$ARRAY_JOB"
echo "JCP14_MPI_PACK_JOB_ID=$PACK_JOB"
echo "JCP14_MPI_CAMPAIGN=$CAMPAIGN"
echo JCP14_MPI_BOOTSTRAP_COMPLETE
