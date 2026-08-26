#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --job-name=j14-e8-res
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=vram16
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out

set -Eeuo pipefail
trap 'rc=$?; rm -f CASE_SUCCESS; echo "J14_E8_RESUME_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CAMPAIGN=${DGFS_E8_CAMPAIGN:?DGFS_E8_CAMPAIGN is required}
SRC=${DGFS_SOLVER_SRC:?DGFS_SOLVER_SRC is required}
MAX_SEGMENTS=${DGFS_MAX_SEGMENTS:-6}
SEGMENT_SPAN=${DGFS_SEGMENT_SPAN:-10}
NOMINAL_DT=0.001
THRESHOLD=2e-5
RUN_DIR="$CAMPAIGN/run"
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
mkdir -p "$CUDA_CACHE_PATH" segments

test -s mesh.frfsm
test -s source-dist-140.0.frfss
test -s source-bulk-140.0.frfss
test -s source-dgfs.ini
test -s source-kinetic-residual.csv
python "$SRC/cases/jcp2019_fig14b_validation/verify_case.py"
nvidia-smi

BASELINE=$(python - source-kinetic-residual.csv <<'PY'
import csv, math, sys
with open(sys.argv[1], newline='') as stream:
    for row in csv.DictReader(stream):
        try:
            raw = float(row['f'])
            normalized = float(row['f_normalized'])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(raw) and math.isfinite(normalized) and normalized > 0:
            print(f'{raw / normalized:.17g}')
            break
    else:
        raise SystemExit('NORMALISATION_REFERENCE_NOT_FOUND')
PY
)

rm -f kinetic_residual.csv CASE_SUCCESS CASE_NOT_CONVERGED
START=140
RESTART=source-dist-140.0.frfss
FINAL_END=""
FINAL_RESIDUAL=""

for ((SEGMENT=0; SEGMENT<MAX_SEGMENTS; SEGMENT++)); do
    END=$((START + SEGMENT_SPAN))
    TAG=$(printf '%02d' "$SEGMENT")
    END_TAG=$(printf '%.1f' "$END")
    CFG="segments/segment_${TAG}.ini"
    LOG="segments/solver_${TAG}.log"
    CONV="segments/convergence_${TAG}.json"

    python "$SRC/hpc/make_jcp14_continuation_config.py" \
        --source source-dgfs.ini --output "$CFG" --start "$START" --end "$END" \
        --output-span "$SEGMENT_SPAN" --normalisation "$BASELINE" --batch 256

    echo "J14_E8_SEGMENT_START segment=$SEGMENT interval=$START,$END restart=$RESTART"
    /usr/bin/time -f '%e' -o "segments/wall_${TAG}.txt" \
        "$ENV_DIR/bin/python" -m frfs restart mesh.frfsm "$RESTART" "$CFG" -b cuda \
        2>&1 | tee "$LOG"

    test -s "dist-${END_TAG}.frfss"
    test -s "bulk-${END_TAG}.frfss"
    if python "$SRC/cases/jcp2019_fig14b_validation/check_convergence.py" \
            --csv kinetic_residual.csv --segment-start "$START" --segment-end "$END" \
            --dt "$NOMINAL_DT" --threshold "$THRESHOLD" --output "$CONV"; then
        FINAL_END=$END_TAG
        FINAL_RESIDUAL=$(python -c "import json; print(json.load(open('$CONV'))['paper_normalized_residual'])")
        break
    else
        RC=$?
        (( RC == 2 )) || exit "$RC"
    fi
    RESTART="dist-${END_TAG}.frfss"
    START=$END
done

if [[ -z "$FINAL_END" ]]; then
    touch CASE_NOT_CONVERGED
    echo "J14_E8_MAX_SEGMENTS_REACHED max=$MAX_SEGMENTS"
    exit 5
fi

cp "dist-${FINAL_END}.frfss" dist-final.frfss
cp "bulk-${FINAL_END}.frfss" bulk-final.frfss
cp "segments/segment_$(printf '%02d' "$SEGMENT").ini" dgfs-final.ini
"$ENV_DIR/bin/python" -m frfs export mesh.frfsm bulk-final.frfss physical-final.vtu -d 24
test -s physical-final.vtu
{
    echo "benchmark=Jaiswal_Alexeenko_Hu_JCP378_2019_Figure14"
    echo "mach=1.59"
    echo "spatial_elements=8"
    echo "source_checkpoint_time=140.0"
    echo "velocity_batch=256"
    echo "final_time=$FINAL_END"
    echo "paper_normalized_residual=$FINAL_RESIDUAL"
    echo "paper_threshold=$THRESHOLD"
    echo "solver_commit=$(git -C "$SRC" rev-parse HEAD)"
} > METADATA.txt
touch CASE_SUCCESS
sha256sum SOURCE.json mesh.frfsm source-dist-140.0.frfss source-bulk-140.0.frfss \
    dgfs-final.ini dist-final.frfss bulk-final.frfss physical-final.vtu \
    kinetic_residual.csv METADATA.txt CASE_SUCCESS > SHA256SUMS.txt

ZIP="$ROOT/j14e8_${SLURM_JOB_ID}.zip"
zip -9 -r "$ZIP" . -x '.cuda-cache/*'
zip -T "$ZIP"
sha256sum "$ZIP" > "$ZIP.sha256.txt"
echo "J14_E8_RESUME_COMPLETE final_time=$FINAL_END residual=$FINAL_RESIDUAL"
echo "UPLOAD_ZIP=$ZIP"
echo "UPLOAD_SHA=$ZIP.sha256.txt"
ls -lh "$ZIP" "$ZIP.sha256.txt"

