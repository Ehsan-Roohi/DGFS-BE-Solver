#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "J14_E8_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
REPO=${DGFS_REPO:-https://github.com/Ehsan-Roohi/DGFS-BE-Solver.git}
REF=${DGFS_REF:-agent/jcp14-2gpu-checkpoint-continuation}
SOURCE=${DGFS_E8_SOURCE:-$ROOT/jcp14_20260822_164418/e8}
SOURCE_TIME=${DGFS_E8_SOURCE_TIME:-140.0}
MAX_SEGMENTS=${DGFS_MAX_SEGMENTS:-6}
STAMP=$(date +%Y%m%d_%H%M%S)
CAMPAIGN="$ROOT/j14e8_$STAMP"
SRC="$CAMPAIGN/src"
RUN="$CAMPAIGN/run"

command -v git >/dev/null
command -v sbatch >/dev/null
test -x "$ENV_DIR/bin/python"
test -s "$SOURCE/mesh.frfsm"
test -s "$SOURCE/dist-${SOURCE_TIME}.frfss"
test -s "$SOURCE/bulk-${SOURCE_TIME}.frfss"
test -s "$SOURCE/dgfs.ini"
test -s "$SOURCE/kinetic_residual.csv"
[[ "$SOURCE_TIME" == "140.0" ]] || { echo "SUPPORTED_SOURCE_TIME_IS_140.0"; exit 2; }

mkdir -p "$RUN"
git init -q "$SRC"
git -C "$SRC" remote add origin "$REPO"
git -C "$SRC" fetch --depth 1 origin "$REF"
git -C "$SRC" checkout -q --detach FETCH_HEAD
cp "$SOURCE/mesh.frfsm" "$RUN/mesh.frfsm"
cp "$SOURCE/dist-${SOURCE_TIME}.frfss" "$RUN/source-dist-140.0.frfss"
cp "$SOURCE/bulk-${SOURCE_TIME}.frfss" "$RUN/source-bulk-140.0.frfss"
cp "$SOURCE/dgfs.ini" "$RUN/source-dgfs.ini"
cp "$SOURCE/kinetic_residual.csv" "$RUN/source-kinetic-residual.csv"

"$ENV_DIR/bin/python" - "$RUN" "$SOURCE" "$SOURCE_TIME" "$SRC" <<'PY'
import hashlib, json, pathlib, subprocess, sys
run = pathlib.Path(sys.argv[1])
source = pathlib.Path(sys.argv[2])
source_time = float(sys.argv[3])
src = pathlib.Path(sys.argv[4])
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
files = {
    'mesh': run / 'mesh.frfsm',
    'dist': run / 'source-dist-140.0.frfss',
    'bulk': run / 'source-bulk-140.0.frfss',
    'config': run / 'source-dgfs.ini',
    'residual': run / 'source-kinetic-residual.csv',
}
record = {
    'source_directory': str(source),
    'source_time': source_time,
    'spatial_elements': 8,
    'velocity_batch': 256,
    'solver_commit': subprocess.check_output(
        ['git', '-C', str(src), 'rev-parse', 'HEAD'], text=True).strip(),
    'sha256': {name: digest(path) for name, path in files.items()},
}
(run / 'SOURCE.json').write_text(json.dumps(record, indent=2) + '\n')
PY

PYTHONPATH="$SRC" "$ENV_DIR/bin/python" "$SRC/cases/jcp2019_fig14b_validation/verify_case.py"
JOB=$(cd "$RUN" && sbatch --parsable \
    --export=ALL,DGFS_ROOT="$ROOT",DGFS_ENV="$ENV_DIR",DGFS_E8_CAMPAIGN="$CAMPAIGN",DGFS_SOLVER_SRC="$SRC",DGFS_MAX_SEGMENTS="$MAX_SEGMENTS" \
    "$SRC/hpc/run_unity_jcp14_e8_resume.slurm")
JOB=${JOB%%;*}
{
    echo "campaign=$CAMPAIGN"
    echo "job=$JOB"
    echo "source=$SOURCE"
    echo "source_time=$SOURCE_TIME"
    echo "repo_ref=$REF"
    echo "repo_commit=$(git -C "$SRC" rev-parse HEAD)"
} > "$CAMPAIGN/SUBMISSION.txt"
echo "J14_E8_JOB_ID=$JOB"
echo "J14_E8_RUN_DIR=$CAMPAIGN"
echo "J14_E8_BOOTSTRAP_COMPLETE"
