#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --job-name=jcp14-mp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=pack-%j.out

set -Eeuo pipefail
trap 'rc=$?; rm -f "${CAMPAIGN:-.}/JCP14_MPI_SUCCESS"; echo "JCP14_MPI_PACK_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR
ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CAMPAIGN=${DGFS_MPI_CAMPAIGN:?DGFS_MPI_CAMPAIGN is required}
SRC=${DGFS_SOLVER_SRC:?DGFS_SOLVER_SRC is required}
cd "$CAMPAIGN"
test -s e4/CASE_SUCCESS
test -s e8/CASE_SUCCESS

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"

python "$SRC/cases/jcp2019_fig14b_validation/compare_fig14.py" \
    --campaign "$CAMPAIGN" \
    --reference "$SRC/cases/jcp2019_fig14b_validation/fig14_digitized.csv" \
    --output-dir validation
test -s validation/fig14_validation.svg
test -s validation/fig14_metrics.csv

STAGE="$CAMPAIGN/package"
mkdir -p "$STAGE/e4" "$STAGE/e8" "$STAGE/validation" "$STAGE/code"
cp SUBMISSION.txt "$STAGE/code/"
for N in 4 8; do
    for F in SOURCE.json dgfs.ini mesh.frfsm dist-final.frfss bulk-final.frfss \
             physical-final.vtu kinetic_residual.csv mpi_benchmark.json METADATA.txt \
             SHA256SUMS.txt CASE_SUCCESS; do
        cp "e$N/$F" "$STAGE/e$N/"
    done
    cp e$N/serial_bench/WALLTIME_SECONDS.txt "$STAGE/e$N/serial_benchmark_wall.txt"
    cp e$N/mpi_bench/WALLTIME_SECONDS.txt "$STAGE/e$N/mpi_benchmark_wall.txt"
done
cp validation/* "$STAGE/validation/"
find "$SRC/cases/jcp2019_fig14b_validation" -maxdepth 1 -type f -exec cp {} "$STAGE/code/" \;
for F in bootstrap_unity_jcp14_2gpu.sh run_unity_jcp14_2gpu.slurm \
         pack_unity_jcp14_2gpu.slurm snapshot_jcp14_checkpoint.py \
         partition_line_checkpoint.py combine_native_moments.py \
         make_jcp14_continuation_config.py verify_jcp14_mpi_equivalence.py; do
    cp "$SRC/hpc/$F" "$STAGE/code/"
done
git -C "$SRC" rev-parse HEAD > "$STAGE/code/GIT_COMMIT.txt"
(cd "$STAGE" && find . -type f ! -name FILES.sha256 -print0 | sort -z | xargs -0 sha256sum > FILES.sha256)
touch JCP14_MPI_SUCCESS
ZIP="$ROOT/jcp14_mpi_${SLURM_JOB_ID}.zip"
(cd "$STAGE" && zip -9 -r "$ZIP" .)
zip -T "$ZIP"
sha256sum "$ZIP" > "$ZIP.sha256.txt"
echo JCP14_MPI_COMPLETE
echo "UPLOAD_ZIP=$ZIP"
echo "UPLOAD_SHA=$ZIP.sha256.txt"
ls -lh "$ZIP" "$ZIP.sha256.txt"
