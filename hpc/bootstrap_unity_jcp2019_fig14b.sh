#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "DGFS_JCP2019_FIG14B_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

DGFS_ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
DGFS_ENV=${DGFS_ENV:-$DGFS_ROOT/dgfs_py310}
DGFS_REPO=${DGFS_REPO:-https://github.com/Ehsan-Roohi/DGFS-BE-Solver.git}
DGFS_REF=${DGFS_REF:-agent/jcp2019-fig14b-exact}
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$DGFS_ROOT/runs/normal_shock_M1p59_jcp2019_fig14b_$STAMP"
SOLVER_SRC="$RUN_DIR/source/DGFS-BE-Solver"
CASE_REL=cases/jcp2019_fig14b_normal_shock

command -v git >/dev/null
command -v sbatch >/dev/null
test -x "$DGFS_ENV/bin/python"
mkdir -p "$RUN_DIR/source"
git clone --depth 1 --branch "$DGFS_REF" "$DGFS_REPO" "$SOLVER_SRC"

cp "$SOLVER_SRC/$CASE_REL/dgfs_fig14b.ini" "$RUN_DIR/"
cp "$SOLVER_SRC/$CASE_REL/mesh_fig14b_8elem.msh" "$RUN_DIR/"
cp "$SOLVER_SRC/$CASE_REL/verify_case.py" "$RUN_DIR/"
cp "$SOLVER_SRC/$CASE_REL/plot_fig14b.py" "$RUN_DIR/"
cp "$SOLVER_SRC/$CASE_REL/run.slurm" "$RUN_DIR/"
chmod +x "$RUN_DIR/verify_case.py" "$RUN_DIR/plot_fig14b.py" "$RUN_DIR/run.slurm"

(cd "$SOLVER_SRC" && PYTHONPATH="$SOLVER_SRC" \
    "$DGFS_ENV/bin/python" "$CASE_REL/verify_case.py")
PYTHONPATH="$SOLVER_SRC" "$DGFS_ENV/bin/python" -c \
    'from frfs.backends import BaseBackend; from frfs.readers.native import NativeReader; print("DGFS_PYTHON_COMPATIBILITY_VERIFIED")'
JOB_ID=$(cd "$RUN_DIR" && sbatch --parsable \
    --export=ALL,DGFS_ROOT="$DGFS_ROOT",DGFS_ENV="$DGFS_ENV",DGFS_SOLVER_SRC="$SOLVER_SRC" \
    ./run.slurm)
JOB_ID=${JOB_ID%%;*}

echo "DGFS_JCP2019_FIG14B_JOB_ID=$JOB_ID"
echo "DGFS_JCP2019_FIG14B_RUN_DIR=$RUN_DIR"
echo "DGFS_JCP2019_FIG14B_BOOTSTRAP_COMPLETE"
