#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --job-name=jcp14-tune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=vram16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "JCP14_AUTOTUNE_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
RUN_DIR=${DGFS_AUTOTUNE_DIR:?DGFS_AUTOTUNE_DIR is required}
SRC=${DGFS_SOLVER_SRC:?DGFS_SOLVER_SRC is required}
CASE="$SRC/cases/jcp2019_fig14b_validation"
cd "$RUN_DIR"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load cuda/12.6
module load openmpi/5.0.3-cuda12.6
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_CACHE_PATH="$RUN_DIR/.cuda-cache"
mkdir -p "$CUDA_CACHE_PATH"

python "$CASE/verify_case.py"
nvidia-smi

for BATCH in 64 256 1024; do
    R="$RUN_DIR/batch${BATCH}"
    mkdir -p "$R"
    cp "$CASE/mesh_4e.msh" "$R/mesh.msh"
    python - "$CASE/dgfs.ini" "$R/dgfs.ini" "$BATCH" <<'PY'
import configparser, sys
source, target, batch = sys.argv[1:]
cfg = configparser.ConfigParser(); cfg.optionxform = str
with open(source) as stream: cfg.read_file(stream)
cfg.set('constants', 'NvBatchSize', batch)
cfg.set('solver-time-integrator', 'tstart', '0')
cfg.set('solver-time-integrator', 'tend', '0.1')
cfg.set('soln-plugin-dgfsdistwriterstd', 'dt-out', '0.1')
cfg.set('soln-plugin-dgfsmomwriterstd', 'dt-out', '0.1')
if not cfg.has_section('backend-cuda'): cfg.add_section('backend-cuda')
cfg.set('backend-cuda', 'device-id', '0')
cfg.set('backend-cuda', 'mpi-type', 'standard')
with open(target, 'w') as stream: cfg.write(stream)
PY
    (
        cd "$R"
        python -m frfs import mesh.msh mesh.frfsm
        set +e
        /usr/bin/time -f '%e' -o WALLTIME_SECONDS.txt \
            python -m frfs run mesh.frfsm dgfs.ini -b cuda > solver.log 2>&1
        RC=$?
        set -e
        if (( RC == 0 )) && [[ -s dist-0.1.frfss && -s bulk-0.1.frfss ]]; then
            echo complete > RUN_STATUS.txt
        else
            echo "failed_rc=$RC" > RUN_STATUS.txt
        fi
    )
    echo "JCP14_AUTOTUNE_VARIANT batch=$BATCH status=$(cat "$R/RUN_STATUS.txt")"
done

python "$SRC/hpc/verify_batch_equivalence.py" --root "$RUN_DIR" \
    --batches 64 256 1024 --output "$RUN_DIR/batch_equivalence.json"
test -s RECOMMENDATION.env

STAGE="$RUN_DIR/package"
mkdir -p "$STAGE"
cp batch_equivalence.json batch_equivalence.csv RECOMMENDATION.env "$STAGE/"
for BATCH in 64 256 1024; do
    mkdir -p "$STAGE/batch${BATCH}"
    for F in dgfs.ini solver.log WALLTIME_SECONDS.txt RUN_STATUS.txt kinetic_residual.csv \
             dist-0.1.frfss bulk-0.1.frfss; do
        [[ -f "batch${BATCH}/$F" ]] && cp "batch${BATCH}/$F" "$STAGE/batch${BATCH}/"
    done
done
git -C "$SRC" rev-parse HEAD > "$STAGE/GIT_COMMIT.txt"
(cd "$STAGE" && find . -type f ! -name FILES.sha256 -print0 | sort -z | xargs -0 sha256sum > FILES.sha256)
ZIP="$ROOT/jcp14_tune_${SLURM_JOB_ID}.zip"
(cd "$STAGE" && zip -9 -r "$ZIP" .)
zip -T "$ZIP"
sha256sum "$ZIP" > "$ZIP.sha256.txt"
touch JCP14_AUTOTUNE_SUCCESS
echo JCP14_AUTOTUNE_COMPLETE
cat RECOMMENDATION.env
echo "UPLOAD_ZIP=$ZIP"
echo "UPLOAD_SHA=$ZIP.sha256.txt"
ls -lh "$ZIP" "$ZIP.sha256.txt"
