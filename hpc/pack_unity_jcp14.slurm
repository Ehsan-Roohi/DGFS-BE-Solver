#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --job-name=jcp14-pack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=pack-%j.out

set -Eeuo pipefail
trap 'rc=$?; rm -f "${CAMPAIGN:-.}/JCP14_SUCCESS"; echo "JCP14_PACK_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR
DGFS_ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
DGFS_ENV=${DGFS_ENV:-$DGFS_ROOT/dgfs_py310}
CAMPAIGN=${DGFS_JCP14_CAMPAIGN:?DGFS_JCP14_CAMPAIGN is required}
SOLVER_SRC=${DGFS_SOLVER_SRC:?DGFS_SOLVER_SRC is required}
cd "$CAMPAIGN"
test -s e4/CASE_SUCCESS
test -s e8/CASE_SUCCESS

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$DGFS_ENV"
export PYTHONPATH="$SOLVER_SRC${PYTHONPATH:+:$PYTHONPATH}"

python src/cases/jcp2019_fig14b_validation/compare_fig14.py \
    --campaign "$CAMPAIGN" \
    --reference src/cases/jcp2019_fig14b_validation/fig14_digitized.csv \
    --output-dir validation
test -s validation/fig14_validation.svg
test -s validation/fig14_metrics.csv

STAGE="$CAMPAIGN/package"
mkdir -p "$STAGE/e4" "$STAGE/e8" "$STAGE/validation" "$STAGE/code"
cp -p "$CAMPAIGN/SUBMISSION.txt" "$STAGE/code/"
for N in 4 8; do
    for F in dgfs.ini mesh.msh mesh.frfsm dist-final.frfss bulk-final.frfss \
             physical-final.vtu kinetic_residual.csv METADATA.txt SHA256SUMS.txt CASE_SUCCESS; do
        cp "e${N}/$F" "$STAGE/e${N}/"
    done
    cp e${N}/solver_*.log e${N}/convergence_*.json "$STAGE/e${N}/"
done
cp validation/* "$STAGE/validation/"
find src/cases/jcp2019_fig14b_validation -maxdepth 1 -type f \
    -exec cp {} "$STAGE/code/" \;
git -C src rev-parse HEAD > "$STAGE/code/GIT_COMMIT.txt"
git -C src status --short > "$STAGE/code/GIT_STATUS.txt"
(cd "$STAGE" && find . -type f ! -name FILES.sha256 -print0 | sort -z | xargs -0 sha256sum > FILES.sha256)
touch JCP14_SUCCESS

ZIP="$DGFS_ROOT/jcp14_${SLURM_JOB_ID}.zip"
(cd "$STAGE" && zip -9 -r "$ZIP" .)
zip -T "$ZIP"
sha256sum "$ZIP" > "$ZIP.sha256.txt"
echo "JCP14_COMPLETE"
echo "UPLOAD_ZIP=$ZIP"
echo "UPLOAD_SHA=$ZIP.sha256.txt"
ls -lh "$ZIP" "$ZIP.sha256.txt"
