#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --job-name=jcp14-exact
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=vram16
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%A_%a.out

set -Eeuo pipefail
trap 'rc=$?; rm -f CASE_SUCCESS; echo "JCP14_CASE_FAILED elements=${ELEMENTS:-unknown} rc=$rc line=$LINENO"; exit "$rc"' ERR

DGFS_ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
DGFS_ENV=${DGFS_ENV:-$DGFS_ROOT/dgfs_py310}
CAMPAIGN=${DGFS_JCP14_CAMPAIGN:?DGFS_JCP14_CAMPAIGN is required}
SOLVER_SRC=${DGFS_SOLVER_SRC:?DGFS_SOLVER_SRC is required}
ELEMENTS=${SLURM_ARRAY_TASK_ID:?run as array 4,8}
MAX_SEGMENTS=${DGFS_MAX_SEGMENTS:-24}
SEGMENT_SPAN=${DGFS_SEGMENT_SPAN:-10}
NOMINAL_DT=0.001
THRESHOLD=2e-5
RUN_DIR="$CAMPAIGN/e${ELEMENTS}"
cd "$RUN_DIR"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load cuda/12.6
module load openmpi/5.0.3-cuda12.6
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$DGFS_ENV"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONPATH="$SOLVER_SRC${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_CACHE_PATH="$RUN_DIR/.cuda-cache"
mkdir -p "$CUDA_CACHE_PATH"

python "$CAMPAIGN/src/cases/jcp2019_fig14b_validation/verify_case.py"
nvidia-smi
python -m frfs import mesh.msh mesh.frfsm

BASELINE=""
FINAL_END=""
FINAL_RESIDUAL=""
for ((SEGMENT=0; SEGMENT<MAX_SEGMENTS; SEGMENT++)); do
    START=$((SEGMENT * SEGMENT_SPAN))
    END=$(((SEGMENT + 1) * SEGMENT_SPAN))
    TAG=$(printf '%02d' "$SEGMENT")
    CFG="segment_${TAG}.ini"
    LOG="solver_${TAG}.log"
    CONV="convergence_${TAG}.json"

    if (( SEGMENT > 0 )) && [[ -z "$BASELINE" ]]; then
        BASELINE=$(python - kinetic_residual.csv <<'PY'
import csv, math, sys
with open(sys.argv[1], newline='') as stream:
    for row in csv.DictReader(stream):
        try:
            raw, normalized = float(row['f']), float(row['f_normalized'])
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

    python - dgfs.ini "$CFG" "$START" "$END" "$SEGMENT_SPAN" "$BASELINE" <<'PY'
import configparser, sys
source, target, start, end, span, baseline = sys.argv[1:]
cfg = configparser.ConfigParser(); cfg.optionxform = str
with open(source) as stream: cfg.read_file(stream)
cfg.set('solver-time-integrator', 'tstart', start)
cfg.set('solver-time-integrator', 'tend', end)
cfg.set('soln-plugin-dgfsdistwriterstd', 'dt-out', span)
cfg.set('soln-plugin-dgfsmomwriterstd', 'dt-out', span)
if baseline:
    cfg.set('soln-plugin-dgfsresidualstd', 'normalisation-resid', baseline)
else:
    cfg.remove_option('soln-plugin-dgfsresidualstd', 'normalisation-resid')
with open(target, 'w') as stream: cfg.write(stream)
PY

    echo "JCP14_SEGMENT_START elements=$ELEMENTS segment=$SEGMENT interval=$START,$END"
    if (( SEGMENT == 0 )); then
        test ! -e kinetic_residual.csv
        python -m frfs run mesh.frfsm "$CFG" -b cuda 2>&1 | tee "$LOG"
    else
        PREV=$(printf '%.1f' "$START")
        test -s "dist-${PREV}.frfss"
        python -m frfs restart mesh.frfsm "dist-${PREV}.frfss" "$CFG" -b cuda \
            2>&1 | tee "$LOG"
    fi

    END_TAG=$(printf '%.1f' "$END")
    test -s "dist-${END_TAG}.frfss"
    test -s "bulk-${END_TAG}.frfss"
    if python "$CAMPAIGN/src/cases/jcp2019_fig14b_validation/check_convergence.py" \
            --csv kinetic_residual.csv --segment-start "$START" --segment-end "$END" \
            --dt "$NOMINAL_DT" --threshold "$THRESHOLD" --output "$CONV"; then
        FINAL_END=$END_TAG
        FINAL_RESIDUAL=$(python -c "import json; print(json.load(open('$CONV'))['paper_normalized_residual'])")
        break
    else
        RC=$?
        (( RC == 2 )) || exit "$RC"
    fi
done

if [[ -z "$FINAL_END" ]]; then
    touch CASE_NOT_CONVERGED
    echo "JCP14_MAX_SEGMENTS_REACHED elements=$ELEMENTS max=$MAX_SEGMENTS"
    exit 5
fi

cp "dist-${FINAL_END}.frfss" dist-final.frfss
cp "bulk-${FINAL_END}.frfss" bulk-final.frfss
python -m frfs export mesh.frfsm bulk-final.frfss physical-final.vtu -d 24
test -s physical-final.vtu
{
    echo "benchmark=Jaiswal_Alexeenko_Hu_JCP378_2019_Figure14"
    echo "reference=Ohwada_PhysFluidsA5_1993_217_234"
    echo "mach=1.59"
    echo "kn_global_approx=0.055"
    echo "physical_domain_mm=-15,15"
    echo "spatial_elements=$ELEMENTS"
    echo "DG_order=3"
    echo "velocity_grid=32^3"
    echo "velocity_domain=-7,7"
    echo "angular_M=6"
    echo "limiter=none"
    echo "final_time=$FINAL_END"
    echo "paper_normalized_residual=$FINAL_RESIDUAL"
    echo "paper_threshold=$THRESHOLD"
    echo "solver_commit=$(git -C "$SOLVER_SRC" rev-parse HEAD)"
} > METADATA.txt
touch CASE_SUCCESS
sha256sum dgfs.ini mesh.msh mesh.frfsm dist-final.frfss bulk-final.frfss \
    physical-final.vtu kinetic_residual.csv METADATA.txt CASE_SUCCESS > SHA256SUMS.txt
echo "JCP14_CASE_COMPLETE elements=$ELEMENTS final_time=$FINAL_END residual=$FINAL_RESIDUAL"
