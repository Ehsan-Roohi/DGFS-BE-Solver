#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --job-name=dgfs-jcp-f14b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=vram16
#SBATCH --mem=16G
#SBATCH --time=03:50:00

set -Eeuo pipefail
trap 'rc=$?; rm -f DGFS_JCP2019_FIG14B_SUCCESS; echo "DGFS_JCP2019_FIG14B_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

DGFS_ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
DGFS_ENV=${DGFS_ENV:-$DGFS_ROOT/dgfs_py310}
RUN_DIR=$(pwd)
SOLVER_SRC=${DGFS_SOLVER_SRC:-$RUN_DIR/source/DGFS-BE-Solver}
SEGMENT=${DGFS_SEGMENT:-0}
MAX_SEGMENTS=${DGFS_MAX_SEGMENTS:-30}
SEGMENT_SPAN=${DGFS_SEGMENT_SPAN:-10}
NOMINAL_DT=${DGFS_NOMINAL_DT:-0.001}
THRESHOLD=${DGFS_CONVERGENCE_THRESHOLD:-2e-5}

[[ "$SEGMENT" =~ ^[0-9]+$ ]]
[[ "$MAX_SEGMENTS" =~ ^[1-9][0-9]*$ ]]
[[ "$SEGMENT_SPAN" =~ ^[1-9][0-9]*$ ]]
(( SEGMENT < MAX_SEGMENTS ))
START=$((SEGMENT * SEGMENT_SPAN))
END=$(((SEGMENT + 1) * SEGMENT_SPAN))
TAG=$(printf '%02d' "$SEGMENT")
END_TAG=$(printf '%.1f' "$END")
SEG_CFG="dgfs_fig14b_segment_${TAG}.ini"
SEG_LOG="solver_segment_${TAG}.log"
CONV_JSON="convergence_segment_${TAG}.json"

if [[ -s DGFS_JCP2019_FIG14B_SUCCESS ]]; then
    echo "DGFS_JCP2019_FIG14B_ALREADY_COMPLETE"
    exit 0
fi

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load cuda/12.6
module load openmpi/5.0.3-cuda12.6
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$DGFS_ENV"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_CACHE_PATH="$RUN_DIR/.cuda-cache"
export PYTHONPATH="$SOLVER_SRC${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CUDA_CACHE_PATH"

python verify_case.py
nvidia-smi

BASELINE=""
if (( SEGMENT > 0 )); then
    test -s kinetic_residual.csv
    BASELINE=$(python - kinetic_residual.csv <<'PY'
import csv
import math
import sys

with open(sys.argv[1], newline='') as stream:
    for row in csv.DictReader(stream):
        try:
            raw = float(row['f'])
            normalized = float(row['f_normalized'])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(raw) and math.isfinite(normalized):
            print(f'{raw:.17g}')
            break
    else:
        raise SystemExit('NORMALISATION_REFERENCE_NOT_FOUND')
PY
    )
fi

python - dgfs_fig14b.ini "$SEG_CFG" "$START" "$END" "$SEGMENT_SPAN" "$BASELINE" <<'PY'
import configparser
import sys

source, target, start, end, span, baseline = sys.argv[1:]
cfg = configparser.ConfigParser()
cfg.optionxform = str
with open(source) as stream:
    cfg.read_file(stream)
cfg.set('solver-time-integrator', 'tstart', start)
cfg.set('solver-time-integrator', 'tend', end)
cfg.set('soln-plugin-dgfsdistwriterstd', 'dt-out', span)
cfg.set('soln-plugin-dgfsmomwriterstd', 'dt-out', span)
option = 'normalisation-resid'
if baseline:
    cfg.set('soln-plugin-dgfsresidualstd', option, baseline)
else:
    cfg.remove_option('soln-plugin-dgfsresidualstd', option)
with open(target, 'w') as stream:
    cfg.write(stream)
PY

echo "DGFS_SEGMENT=$SEGMENT"
echo "DGFS_TIME_INTERVAL=$START,$END"
echo "DGFS_MAX_SEGMENTS=$MAX_SEGMENTS"

if (( SEGMENT == 0 )); then
    test ! -e kinetic_residual.csv
    python -m frfs import mesh_fig14b_8elem.msh mesh.frfsm
    python -m frfs run mesh.frfsm "$SEG_CFG" -b cuda 2>&1 | tee "$SEG_LOG"
else
    PREV_TAG=$(printf '%.1f' "$START")
    PREV_DIST="dist_dgfs_fig14b-${PREV_TAG}.frfss"
    test -s mesh.frfsm
    test -s "$PREV_DIST"
    python -m frfs restart mesh.frfsm "$PREV_DIST" "$SEG_CFG" -b cuda \
        2>&1 | tee "$SEG_LOG"
fi

FINAL_DIST="dist_dgfs_fig14b-${END_TAG}.frfss"
FINAL_BULK="bulksol_dgfs_fig14b-${END_TAG}.frfss"
test -s "$FINAL_DIST"
test -s "$FINAL_BULK"
test -s kinetic_residual.csv

CONVERGED=0
if python check_convergence.py --csv kinetic_residual.csv \
        --segment-start "$START" --segment-end "$END" --dt "$NOMINAL_DT" \
        --threshold "$THRESHOLD" --output "$CONV_JSON" \
        2>&1 | tee "convergence_segment_${TAG}.log"; then
    CONVERGED=1
else
    CHECK_RC=$?
    if (( CHECK_RC != 2 )); then
        exit "$CHECK_RC"
    fi
fi

FINAL_RESIDUAL=$(python - "$CONV_JSON" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))['paper_normalized_residual'])
PY
)

if (( ! CONVERGED )); then
    touch "DGFS_JCP2019_FIG14B_SEGMENT_${TAG}_CONTINUE"
    NEXT=$((SEGMENT + 1))
    if (( NEXT >= MAX_SEGMENTS )); then
        touch DGFS_JCP2019_FIG14B_NOT_CONVERGED
        echo "DGFS_JCP2019_FIG14B_MAX_SEGMENTS_REACHED"
        echo "FINAL_PAPER_NORMALIZED_RESIDUAL=$FINAL_RESIDUAL"
        exit 5
    fi

    NEXT_JOB=$(sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" \
        --export=ALL,DGFS_ROOT="$DGFS_ROOT",DGFS_ENV="$DGFS_ENV",DGFS_SOLVER_SRC="$SOLVER_SRC",DGFS_SEGMENT="$NEXT",DGFS_MAX_SEGMENTS="$MAX_SEGMENTS",DGFS_SEGMENT_SPAN="$SEGMENT_SPAN",DGFS_NOMINAL_DT="$NOMINAL_DT",DGFS_CONVERGENCE_THRESHOLD="$THRESHOLD" \
        ./run.slurm)
    NEXT_JOB=${NEXT_JOB%%;*}
    echo "DGFS_JCP2019_FIG14B_CONTINUATION_JOB_ID=$NEXT_JOB"
    echo "FINAL_PAPER_NORMALIZED_RESIDUAL=$FINAL_RESIDUAL"
    echo "RUN_DIR=$RUN_DIR"
    exit 0
fi

cp "$FINAL_DIST" dist_dgfs_fig14b-final.frfss
cp "$FINAL_BULK" bulksol_dgfs_fig14b-final.frfss
python -m frfs export mesh.frfsm bulksol_dgfs_fig14b-final.frfss \
    fig14b_final.vtu -d 24
test -s fig14b_final.vtu

if python -c 'import matplotlib' >/dev/null 2>&1; then
    python plot_fig14b.py --run-dir . \
        --bulk-soln bulksol_dgfs_fig14b-final.frfss \
        --output fig14b_raw_and_cell_average.png
else
    echo "PLOT_SKIPPED: matplotlib is not installed in $DGFS_ENV"
fi

{
    echo "paper=Jaiswal_Alexeenko_Hu_JCP378_2019_Figure14b"
    echo "slurm_job_id=${SLURM_JOB_ID:-not-slurm}"
    echo "solver_commit=$(git -C "$SOLVER_SRC" rev-parse HEAD)"
    echo "physical_domain_m=-0.015,0.015"
    echo "spatial_elements=8"
    echo "polynomial_degree=2"
    echo "time_integrator=SSP-RK2"
    echo "velocity_grid=32^3"
    echo "velocity_domain=-7,7"
    echo "final_time=$END"
    echo "final_segment=$SEGMENT"
    echo "final_paper_normalized_residual=$FINAL_RESIDUAL"
    echo "paper_convergence_threshold=$THRESHOLD"
} > RUN_METADATA.txt

touch DGFS_JCP2019_FIG14B_SUCCESS
CHECKSUM_FILES=(dgfs_fig14b.ini mesh_fig14b_8elem.msh mesh.frfsm
    dist_dgfs_fig14b-final.frfss bulksol_dgfs_fig14b-final.frfss
    fig14b_final.vtu kinetic_residual.csv RUN_METADATA.txt run.slurm
    check_convergence.py DGFS_JCP2019_FIG14B_SUCCESS "$CONV_JSON")
if [[ -s fig14b_raw_and_cell_average.png ]]; then
    CHECKSUM_FILES+=(fig14b_raw_and_cell_average.png fig14b_extrema.json)
fi
sha256sum "${CHECKSUM_FILES[@]}" > SHA256SUMS.txt

ZIP="$RUN_DIR/../normal_shock_M1p59_jcp2019_fig14b_${SLURM_JOB_ID:-manual}_COMPLETE.zip"
(cd "$RUN_DIR" && zip -9 -r "$ZIP" . \
    -x '.cuda-cache/*' 'source/DGFS-BE-Solver/.git/*' '*.zip' '*.sha256.txt')
zip -T "$ZIP"
sha256sum "$ZIP" > "$ZIP.sha256.txt"

echo "DGFS_JCP2019_FIG14B_COMPLETE"
echo "FINAL_PAPER_NORMALIZED_RESIDUAL=$FINAL_RESIDUAL"
echo "RUN_DIR=$RUN_DIR"
echo "DOWNLOAD_ZIP=$ZIP"
echo "DOWNLOAD_SHA256=$ZIP.sha256.txt"
