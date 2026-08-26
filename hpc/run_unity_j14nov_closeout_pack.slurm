#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -Eeuo pipefail
trap 'rc=$?; rm -f CLOSEOUT_COMPLETE; echo "J14NOV_CLOSEOUT_PACK_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CLOSEOUT=${DGFS_CLOSEOUT:?DGFS_CLOSEOUT is required}
ORIGINAL=${DGFS_NOV_SOURCE:?DGFS_NOV_SOURCE is required}
SRC=${DGFS_SOLVER_SRC:?DGFS_SOLVER_SRC is required}
cd "$CLOSEOUT"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="$CLOSEOUT/.matplotlib"
export PYTHONPATH="$SRC:$SRC/cases/jcp2019_fig14b_normal_shock/phase3:$SRC/cases/jcp2019_fig14b_validation${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$MPLCONFIGDIR" final_runs paper_cases results audit

cp "$ORIGINAL/audit/M6.json" "$ORIGINAL/audit/M6.csv" "$ORIGINAL/audit/M6.log" audit/
cp "$ORIGINAL/audit/M16.json" "$ORIGINAL/audit/M16.csv" "$ORIGINAL/audit/M16.log" audit/

# Reuse the already-converged legacy M6 raw result without another GPU solve.
mkdir -p final_runs/run_M6_raw
cp "$ORIGINAL/input/mesh.frfsm" final_runs/run_M6_raw/mesh.frfsm
cp "$ORIGINAL/configs/p3b_M6_raw.ini" final_runs/run_M6_raw/p3b_M6_raw.ini
cp "$ORIGINAL/run_M6_raw/dist_p3b_M6_raw-160.25.frfss" final_runs/run_M6_raw/
cp "$ORIGINAL/run_M6_raw/bulksol_p3b_M6_raw-160.25.frfss" final_runs/run_M6_raw/
cp "$ORIGINAL/run_M6_raw/kinetic_residual_p3b.csv" final_runs/run_M6_raw/
cp "$ORIGINAL/run_M6_raw/WALLTIME.txt" final_runs/run_M6_raw/
touch final_runs/run_M6_raw/CONVERGED

for name in M6_fplus M16_raw M16_fplus; do
    FINAL=""
    for seg in 3 2 1; do
        d="$CLOSEOUT/stage_$seg/$name"
        if [[ -e "$d/SEGMENT_SUCCESS" && ! -e "$d/SKIPPED_CONVERGED.txt" ]]; then
            FINAL=$d
            break
        fi
    done
    [[ -n "$FINAL" ]] || { echo "J14NOV_CLOSEOUT_NO_FINAL_STAGE=$name"; exit 6; }
    out="final_runs/run_$name"
    mkdir -p "$out"
    cp "$FINAL/mesh.frfsm" "$out/mesh.frfsm"
    cp "$FINAL/p3b_$name.ini" "$out/p3b_$name.ini"
    cp "$FINAL"/dist_p3b_*.frfss "$out/"
    cp "$FINAL"/bulksol_p3b_*.frfss "$out/"
    cp "$FINAL/kinetic_residual_p3b.csv" "$out/kinetic_residual_p3b.csv"
    $ENV_DIR/bin/python - "$CLOSEOUT" "$name" > "$out/WALLTIME.txt" <<'PY'
import pathlib, sys
root, name = pathlib.Path(sys.argv[1]), sys.argv[2]
total = 0.0
for path in root.glob(f"stage_*/{name}/WALLTIME.txt"):
    total += float(path.read_text().split("=", 1)[1])
print(f"wall_seconds={total:.3f}")
PY
    if [[ -e "$FINAL/CONVERGED" ]]; then touch "$out/CONVERGED"; fi
    echo "J14NOV_CLOSEOUT_FINAL_STAGE case=$name stage=$FINAL converged=$(test -e "$out/CONVERGED" && echo yes || echo no)"
done

$ENV_DIR/bin/python "$SRC/cases/jcp2019_fig14b_normal_shock/phase3/compare_restarts.py" \
    --config "$ORIGINAL/input/dgfs.ini" --mesh "$ORIGINAL/input/mesh.frfsm" \
    --reference "$ORIGINAL/input/dist-source.frfss" \
    --runs final_runs/run_M6_raw final_runs/run_M6_fplus final_runs/run_M16_raw final_runs/run_M16_fplus \
    --tstart 160.0 --tend 180.25 \
    --output-json results/restart_comparison.json \
    --output-csv results/restart_comparison.csv \
    --output-md results/restart_comparison.md \
    --output-figure results/restart_profiles.png \
    --output-cell-figure results/restart_cell_averages.png \
    2>&1 | tee results/restart_compare.log

for name in M6_raw M6_fplus M16_raw M16_fplus; do
    d="paper_cases/$name"
    mkdir -p "$d"
    cp "final_runs/run_$name/p3b_$name.ini" "$d/dgfs.ini"
    cp "final_runs/run_$name/mesh.frfsm" "$d/mesh.frfsm"
    BULK=$(find "final_runs/run_$name" -maxdepth 1 -type f -name 'bulksol_p3b_*.frfss' -printf '%f\n' | sort -V | tail -1)
    cp "final_runs/run_$name/$BULK" "$d/bulk-final.frfss"
done

$ENV_DIR/bin/python "$SRC/cases/jcp2019_fig14b_normal_shock/novelty/j14_novelty_assess.py" \
    --comparison results/restart_comparison.json \
    --audit-m6 audit/M6.json --audit-m16 audit/M16.json \
    --paper-cases paper_cases \
    --paper-reference "$SRC/cases/jcp2019_fig14b_validation/fig14_digitized.csv" \
    --output-dir results 2>&1 | tee results/assessment.log

$ENV_DIR/bin/python "$SRC/cases/jcp2019_fig14b_normal_shock/novelty/j14_novelty_closeout_gate.py" \
    --comparison results/restart_comparison.json \
    --report results/novelty_report.json --summary results/SUMMARY.md \
    --output-dir results --steady-threshold 1.0 2>&1 | tee results/closeout_gate.log

{
    echo "benchmark=Jaiswal_Alexeenko_Hu_JCP378_2019_Figure14"
    echo "mach=1.59"
    echo "kn_global_approx=0.055"
    echo "spatial_elements=8"
    echo "source_campaign=$ORIGINAL"
    echo "steady_threshold=1.0"
    echo "solver_commit=$(git -C "$SRC" rev-parse HEAD)"
    echo "claim_gate=$(if [[ -e results/CLAIM_GATE_PASS ]]; then echo PASS; else echo FAIL; fi)"
} > CLOSEOUT_METADATA.txt
cp "$SRC/cases/jcp2019_fig14b_normal_shock/novelty/CLOSEOUT.md" .
touch CLOSEOUT_COMPLETE
find audit results final_runs paper_cases -type f -print0 | sort -z | xargs -0 sha256sum > FILES.sha256

ZIP="$ROOT/j14novclose_${SLURM_JOB_ID:-manual}.zip"
zip -q -9 -r "$ZIP" audit results final_runs paper_cases CLOSEOUT.md \
    SUBMISSION.txt CLOSEOUT_METADATA.txt CLOSEOUT_COMPLETE FILES.sha256 "slurm-${SLURM_JOB_ID:-manual}.out" \
    -x '.matplotlib/*' '*/__pycache__/*' '*.pyc'
zip -T "$ZIP"
sha256sum "$ZIP" > "$ZIP.sha256.txt"
echo J14NOV_CLOSEOUT_COMPLETE
echo "CLAIM_GATE=$(if [[ -e results/CLAIM_GATE_PASS ]]; then echo PASS; else echo FAIL; fi)"
echo "UPLOAD_ZIP=$ZIP"
echo "UPLOAD_SHA=$ZIP.sha256.txt"
ls -lh "$ZIP" "$ZIP.sha256.txt"
