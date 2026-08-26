#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --job-name=j14-nov
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=vram16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -Eeuo pipefail
trap 'rc=$?; rm -f NOVELTY_COMPLETE; echo "J14NOV_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CAMPAIGN=${DGFS_NOV_CAMPAIGN:?DGFS_NOV_CAMPAIGN is required}
SRC=${DGFS_SOLVER_SRC:?DGFS_SOLVER_SRC is required}
E8=${DGFS_E8_SOURCE:?DGFS_E8_SOURCE is required}
SOURCE="$E8/run"
RUNS=M6_raw:32:6:none,M6_fplus:32:6:fplus,M12_fplus:32:12:fplus,M12_raw:32:12:none
cd "$CAMPAIGN"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load cuda/12.6
module load openmpi/5.0.3-cuda12.6
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export CUDA_CACHE_PATH="$CAMPAIGN/.cuda-cache"
export MPLCONFIGDIR="$CAMPAIGN/.matplotlib"
export PYTHONPATH="$SRC:$SRC/cases/jcp2019_fig14b_normal_shock/phase3:$SRC/cases/jcp2019_fig14b_validation${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" input configs audit results paper_cases
cp "$SRC/cases/jcp2019_fig14b_normal_shock/novelty/NOVELTY.md" NOVELTY.md

test -e "$SOURCE/CASE_SUCCESS"
test -s "$SOURCE/dist-final.frfss"
test -s "$SOURCE/bulk-final.frfss"
test -s "$SOURCE/dgfs-final.ini"
test -s "$SOURCE/mesh.frfsm"
test -s "$SOURCE/kinetic_residual.csv"
test -s "$SOURCE/METADATA.txt"

cp "$SOURCE/dist-final.frfss" input/dist-source.frfss
cp "$SOURCE/bulk-final.frfss" input/bulk-source.frfss
cp "$SOURCE/dgfs-final.ini" input/dgfs.ini
cp "$SOURCE/mesh.frfsm" input/mesh.frfsm
cp "$SOURCE/kinetic_residual.csv" input/kinetic_residual.csv
cp "$SOURCE/METADATA.txt" input/E8_METADATA.txt
cp "$SOURCE/SHA256SUMS.txt" input/E8_SHA256SUMS.txt
(cd "$SOURCE" && sha256sum -c SHA256SUMS.txt)

python "$SRC/cases/jcp2019_fig14b_normal_shock/solver_hook/apply_hook.py" "$SRC"
grep -q "self.projector" "$SRC/frfs/solvers/dgfs/system.py"

START=$(awk -F= '$1=="final_time"{print $2}' input/E8_METADATA.txt)
[[ "$START" =~ ^[0-9]+([.][0-9]+)?$ ]]
read -r END END2 < <("$ENV_DIR/bin/python" - "$START" <<'PY'
import sys
end = float(sys.argv[1]) + 0.25
print(f"{end:.12g} {end:.2f}")
PY
)
echo "J14NOV_INTERVAL=$START,$END"

python "$SRC/cases/jcp2019_fig14b_normal_shock/phase3/make_restart_configs.py"     --base input/dgfs.ini --residual-csv input/kinetic_residual.csv     --tstart "$START" --tend "$END" --dt-out 0.25 --runs "$RUNS" --out-dir configs

echo J14NOV_COLLISION_AUDIT_START_M6
"$ENV_DIR/bin/python" "$SRC/cases/jcp2019_fig14b_normal_shock/novelty/j14_novelty_collision.py"     --config configs/p3b_M6_raw.ini --snapshot input/dist-source.frfss     --mesh input/mesh.frfsm --output-json audit/M6.json --output-csv audit/M6.csv     --repeats 3 2>&1 | tee audit/M6.log

echo J14NOV_COLLISION_AUDIT_START_M12
"$ENV_DIR/bin/python" "$SRC/cases/jcp2019_fig14b_normal_shock/novelty/j14_novelty_collision.py"     --config configs/p3b_M12_raw.ini --snapshot input/dist-source.frfss     --mesh input/mesh.frfsm --output-json audit/M12.json --output-csv audit/M12.csv     --repeats 3 2>&1 | tee audit/M12.log

IFS=',' read -r -a SPECS <<< "$RUNS"
RUNDIRS=()
for spec in "${SPECS[@]}"; do
    name=${spec%%:*}
    d="run_$name"
    RUNDIRS+=("$d")
    mkdir -p "$d"
    cp input/mesh.frfsm "$d/mesh.frfsm"
    ln -sf "$CAMPAIGN/input/dist-source.frfss" "$d/dist-source.frfss"
    cp "configs/p3b_$name.ini" "$d/p3b_$name.ini"
    echo "J14NOV_RUN_START name=$name M_omega=$(cut -d: -f3 <<< "$spec") projection=$(cut -d: -f4 <<< "$spec")"
    t0=$(date +%s.%N)
    (
        cd "$d"
        "$ENV_DIR/bin/python" -m frfs restart mesh.frfsm dist-source.frfss             "p3b_$name.ini" -b cuda 2>&1 | tee solver.log
    )
    t1=$(date +%s.%N)
    "$ENV_DIR/bin/python" - "$t0" "$t1" > "$d/WALLTIME.txt" <<'PY'
import sys
print(f"wall_seconds={float(sys.argv[2])-float(sys.argv[1]):.3f}")
PY
    test -s "$d/dist_p3b_$name-$END2.frfss"
    test -s "$d/bulksol_p3b_$name-$END2.frfss"
    echo "J14NOV_RUN_DONE name=$name $(cat "$d/WALLTIME.txt")"
done

"$ENV_DIR/bin/python" "$SRC/cases/jcp2019_fig14b_normal_shock/phase3/compare_restarts.py"     --config input/dgfs.ini --mesh input/mesh.frfsm --reference input/dist-source.frfss     --runs "${RUNDIRS[@]}" --tstart "$START" --tend "$END"     --output-json results/restart_comparison.json     --output-csv results/restart_comparison.csv     --output-md results/restart_comparison.md     --output-figure results/restart_profiles.png     --output-cell-figure results/restart_cell_averages.png     2>&1 | tee results/restart_compare.log

for spec in "${SPECS[@]}"; do
    name=${spec%%:*}
    d="paper_cases/$name"
    mkdir -p "$d"
    cp "configs/p3b_$name.ini" "$d/dgfs.ini"
    cp input/mesh.frfsm "$d/mesh.frfsm"
    cp "run_$name/bulksol_p3b_$name-$END2.frfss" "$d/bulk-final.frfss"
done

"$ENV_DIR/bin/python" "$SRC/cases/jcp2019_fig14b_normal_shock/novelty/j14_novelty_assess.py"     --comparison results/restart_comparison.json     --audit-m6 audit/M6.json --audit-m12 audit/M12.json     --paper-cases paper_cases     --paper-reference "$SRC/cases/jcp2019_fig14b_validation/fig14_digitized.csv"     --output-dir results 2>&1 | tee results/assessment.log

{
    echo "benchmark=Jaiswal_Alexeenko_Hu_JCP378_2019_Figure14"
    echo "mach=1.59"
    echo "kn_global_approx=0.055"
    echo "spatial_elements=8"
    echo "source_campaign=$E8"
    echo "source_time=$START"
    echo "restart_end=$END"
    echo "runs=$RUNS"
    echo "solver_commit=$(git -C "$SRC" rev-parse HEAD)"
    echo "claim_gate=$(if [[ -e results/CLAIM_GATE_PASS ]]; then echo PASS; else echo FAIL; fi)"
} > NOVELTY_METADATA.txt
touch NOVELTY_COMPLETE
find input configs audit results paper_cases -type f -print0 | sort -z | xargs -0 sha256sum > FILES.sha256

ZIP="$ROOT/j14nov_${SLURM_JOB_ID:-manual}.zip"
zip -q -9 -r "$ZIP" input configs audit results paper_cases run_* NOVELTY.md \
    NOVELTY_METADATA.txt NOVELTY_COMPLETE FILES.sha256 "slurm-${SLURM_JOB_ID:-manual}.out" \
    -x '.cuda-cache/*' '.matplotlib/*' '*/__pycache__/*' '*.pyc' 'run_*/dist-source.frfss'
zip -T "$ZIP"
sha256sum "$ZIP" > "$ZIP.sha256.txt"
echo J14NOV_COMPLETE
echo "CLAIM_GATE=$(if [[ -e results/CLAIM_GATE_PASS ]]; then echo PASS; else echo FAIL; fi)"
echo "UPLOAD_ZIP=$ZIP"
echo "UPLOAD_SHA=$ZIP.sha256.txt"
ls -lh "$ZIP" "$ZIP.sha256.txt"
