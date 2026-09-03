#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "FOURWAY_MAXWELLIAN_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CLOSE=${DGFS_CLOSEOUT:-$ROOT/j14novclose_20260826_211439}
STEADY=${DGFS_STEADY_DIR:-$CLOSE/m16_steady_20260831_125816}
SRC=${DGFS_SOLVER_SRC:-$STEADY/src}
OUT=${DGFS_OUTPUT_DIR:-$PWD}
REF=${DGFS_REF:-2300ebc90a775f8c341842c383d47b5090719ab0}
M6=$CLOSE/final_runs/run_M6_raw
STAMP=$(date +%Y%m%d_%H%M%S)
WORK=$CLOSE/fourway_maxwellian_$STAMP
mkdir -p "$WORK" "$OUT"

CFG=$M6/p3b_M6_raw.ini
MESH=$M6/mesh.frfsm
SNAP=$(find "$M6" -maxdepth 2 -type f -name 'dist_p3b_M6_raw-*.frfss' -printf '%p\n' 2>/dev/null | sort -V | tail -1)
for p in "$ENV_DIR/bin/python" "$CFG" "$MESH" "$SNAP"; do [[ -e "$p" ]] || { echo "MISSING=$p"; exit 2; }; done

curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/DGFS-BE-Solver/$REF/cases/jcp2019_fig14b_normal_shock/novelty/j14_fourway_maxwellian_audit.py" -o "$WORK/j14_fourway_maxwellian_audit.py"
"$ENV_DIR/bin/python" -m py_compile "$WORK/j14_fourway_maxwellian_audit.py"

cat > "$WORK/run.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="v100&x86_64"
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=dgfs-4wayM
#SBATCH --output=SLURM_LOG_PLACEHOLDER
set -Eeuo pipefail
trap 'rc=$?; echo "FOURWAY_MAXWELLIAN_JOB_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ENV_DIR=${DGFS_ENV:?}; SRC=${DGFS_SOLVER_SRC:?}; WORK=${DGFS_WORK:?}; OUT=${DGFS_OUTPUT_DIR:?}
CFG=${DGFS_CONFIG:?}; MESH=${DGFS_MESH:?}; SNAP=${DGFS_SNAPSHOT:?}
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
export CUDA_CACHE_PATH="$WORK/.cuda-cache"
mkdir -p "$CUDA_CACHE_PATH"

echo "FOURWAY_MAXWELLIAN_START node=${SLURMD_NODENAME:-unknown} snapshot=$SNAP"
nvidia-smi --query-gpu=name --format=csv,noheader

"$ENV_DIR/bin/python" "$WORK/j14_fourway_maxwellian_audit.py" \
  --label M6_raw \
  --config "$CFG" --mesh "$MESH" --snapshot "$SNAP" \
  --output-json "$OUT/FOURWAY_MAXWELLIAN_M6.json" --repeats 2

"$ENV_DIR/bin/python" - "$OUT/FOURWAY_MAXWELLIAN_M6.json" "$OUT/FOURWAY_MAXWELLIAN_M6_SUMMARY.md" <<'PY'
import json,sys
j=json.load(open(sys.argv[1])); s=j['summary']; modes=['euclidean','fplus','maxwellian']
lines=['# M6 four-way Maxwellian operator audit','',
       f"- raw max invariant defect: {s['raw_max_invariant_defect']:.6e}",'',
       '| mode | max invariant defect | median correction L2 | scaled Gram cond | tail corr frac | low-support corr frac | negative-node corr frac |',
       '|---|---:|---:|---:|---:|---:|---:|']
for m in modes:
    lines.append(f"| {m} | {s['max_invariant_defect'][m]:.4e} | {s['median_relative_correction_l2'][m]:.4e} | {s['median_scaled_gram_condition'][m]:.3f} | {s['median_tail_correction_fraction'][m]:.4e} | {s['median_low_support_correction_fraction'][m]:.4e} | {s['median_negative_node_correction_fraction'][m]:.4e} |")
lines += ['','## Median high-order disturbance relative to raw cancellation scale','',
          '| mode | heatflux_x | heatflux_z | Pdev_xx | Pxz | c4 |','|---|---:|---:|---:|---:|---:|']
for m in modes:
    h=s['median_high_order_relative_disturbance'][m]
    lines.append(f"| {m} | {h['heatflux_x']:.4e} | {h['heatflux_z']:.4e} | {h['deviatoric_stress_xx']:.4e} | {h['stress_xz']:.4e} | {h['fourth_scalar']:.4e} |")
open(sys.argv[2],'w').write('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY

rm -rf "$CUDA_CACHE_PATH" || true
echo "FOURWAY_MAXWELLIAN_JOB_COMPLETE"
ls -lh "$OUT/FOURWAY_MAXWELLIAN_M6.json" "$OUT/FOURWAY_MAXWELLIAN_M6_SUMMARY.md"
SLURM
sed -i "s|SLURM_LOG_PLACEHOLDER|$OUT/dgfs-fourwayM-%j.out|" "$WORK/run.slurm"

JOB=$(sbatch --parsable \
  --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_SOLVER_SRC="$SRC",DGFS_WORK="$WORK",DGFS_OUTPUT_DIR="$OUT",DGFS_CONFIG="$CFG",DGFS_MESH="$MESH",DGFS_SNAPSHOT="$SNAP" \
  "$WORK/run.slurm")
JOB=${JOB%%;*}
echo "FOURWAY_MAXWELLIAN_JOB=$JOB"
echo "FOURWAY_MAXWELLIAN_SNAPSHOT=$SNAP"
echo "FOURWAY_MAXWELLIAN_JSON=$OUT/FOURWAY_MAXWELLIAN_M6.json"
echo "FOURWAY_MAXWELLIAN_BOOTSTRAP_COMPLETE"
