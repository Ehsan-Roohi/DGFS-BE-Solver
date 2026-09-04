#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "M3_OPERATOR_DIAG_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CLOSE=${DGFS_CLOSEOUT:-$ROOT/j14novclose_20260826_211439}
STEADY=${DGFS_STEADY_DIR:-$CLOSE/m16_steady_20260831_125816}
SRC=${DGFS_SOLVER_SRC:-$STEADY/src}
OUT=${DGFS_OUTPUT_DIR:-$PWD}
REF=${DGFS_REF:-4fa0da1b4edad44447defa1de837fb1e9539ff25}
STAGE1=${DGFS_M3_STAGE1_WORK:-$(find "$CLOSE" -maxdepth 1 -type d -name 'm3_stage1_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)}
STAMP=$(date +%Y%m%d_%H%M%S)
WORK=$CLOSE/m3_operator_diag_$STAMP
mkdir -p "$WORK" "$OUT"

[[ -d "$STAGE1" ]] || { echo "M3_STAGE1_WORK_NOT_FOUND=$STAGE1"; exit 2; }
for p in "$ENV_DIR/bin/python" "$SRC"; do [[ -e "$p" ]] || { echo "MISSING=$p"; exit 2; }; done

AUDIT="$WORK/j14_fourway_maxwellian_audit.py"
curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/DGFS-BE-Solver/$REF/cases/jcp2019_fig14b_normal_shock/novelty/j14_fourway_maxwellian_audit.py" -o "$AUDIT"
"$ENV_DIR/bin/python" -m py_compile "$AUDIT"

NAMES=(M3_raw M3_fplus)
for NAME in "${NAMES[@]}"; do
  CASE="$STAGE1/$NAME"
  CFG="$CASE/$NAME.ini"
  MESH="$CASE/mesh.frfsm"
  SNAP=$(find "$CASE" -maxdepth 1 -type f -name "dist_${NAME}-*.frfss" -printf '%p\n' 2>/dev/null | sort -V | tail -1)
  for p in "$CFG" "$MESH" "$SNAP"; do [[ -s "$p" ]] || { echo "MISSING_STAGE1_INPUT=$p"; exit 3; }; done
  printf '%s|%s|%s|%s\n' "$NAME" "$CFG" "$MESH" "$SNAP" >> "$WORK/inputs.txt"
done

echo "===== M3 OPERATOR DIAG INPUTS ====="
cat "$WORK/inputs.txt"

cat > "$WORK/run.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="v100&x86_64"
#SBATCH --mem=12G
#SBATCH --time=00:25:00
#SBATCH --array=0-1
#SBATCH --job-name=dgfs-m3-opdiag
#SBATCH --output=ARRAY_LOG_PLACEHOLDER
set -Eeuo pipefail
trap 'rc=$?; echo "M3_OPERATOR_DIAG_RUN_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR
ENV_DIR=${DGFS_ENV:?}; SRC=${DGFS_SOLVER_SRC:?}; WORK=${DGFS_WORK:?}; OUT=${DGFS_OUTPUT_DIR:?}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$WORK/inputs.txt")
IFS='|' read -r NAME CFG MESH SNAP <<< "$LINE"
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
export CUDA_CACHE_PATH="$WORK/.cuda-cache-$SLURM_ARRAY_TASK_ID"
mkdir -p "$CUDA_CACHE_PATH"
echo "M3_OPERATOR_DIAG_START name=$NAME node=${SLURMD_NODENAME:-unknown} snapshot=$SNAP"
nvidia-smi --query-gpu=name --format=csv,noheader
"$ENV_DIR/bin/python" "$WORK/j14_fourway_maxwellian_audit.py" \
  --label "$NAME" --config "$CFG" --mesh "$MESH" --snapshot "$SNAP" \
  --output-json "$OUT/M3_OPERATOR_DIAG_${NAME}.json" --repeats 2
rm -rf "$CUDA_CACHE_PATH" || true
echo "M3_OPERATOR_DIAG_DONE name=$NAME"
SLURM
sed -i "s|ARRAY_LOG_PLACEHOLDER|$OUT/dgfs-m3-opdiag-%A_%a.out|" "$WORK/run.slurm"

cat > "$WORK/aggregate.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --job-name=dgfs-m3-opagg
#SBATCH --output=AGG_LOG_PLACEHOLDER
set -Eeuo pipefail
OUT=${DGFS_OUTPUT_DIR:?}; ENV_DIR=${DGFS_ENV:?}; STAGE1=${DGFS_M3_STAGE1_WORK:?}
"$ENV_DIR/bin/python" - "$OUT" "$STAGE1" <<'PY'
import json,pathlib,sys,zipfile
out=pathlib.Path(sys.argv[1]); stage1=pathlib.Path(sys.argv[2]); states=['M3_raw','M3_fplus']; modes=['euclidean','fplus','maxwellian']
js={s:json.load(open(out/f'M3_OPERATOR_DIAG_{s}.json')) for s in states}
lines=['# Mach-3 stage-1 operator diagnosis','',
       'Same-state raw collision operator audited with Euclidean, fplus, and Maxwellian conservative projections. No time integration is performed in this diagnostic.','']
for state in states:
    sm=js[state]['summary']
    lines += [f'## Input state: {state}','',f"- raw max invariant defect: {sm['raw_max_invariant_defect']:.6e}",'',
      '| mode | max inv defect | median correction L2 | scaled Gram cond | tail corr frac | low-support corr frac | negative-node corr frac |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for m in modes:
        lines.append(f"| {m} | {sm['max_invariant_defect'][m]:.4e} | {sm['median_relative_correction_l2'][m]:.4e} | {sm['median_scaled_gram_condition'][m]:.3f} | {sm['median_tail_correction_fraction'][m]:.4e} | {sm['median_low_support_correction_fraction'][m]:.4e} | {sm['median_negative_node_correction_fraction'][m]:.4e} |")
    lines += ['','### Median high-order disturbance relative to raw cancellation scale','',
      '| mode | heatflux_x | heatflux_z | Pdev_xx | Pxz | c4 |','|---|---:|---:|---:|---:|---:|']
    for m in modes:
        h=sm['median_high_order_relative_disturbance'][m]
        lines.append(f"| {m} | {h['heatflux_x']:.4e} | {h['heatflux_z']:.4e} | {h['deviatoric_stress_xx']:.4e} | {h['stress_xz']:.4e} | {h['fourth_scalar']:.4e} |")
    lines += ['']
lines += [f'M3_OPERATOR_DIAG_STAGE1_WORK={stage1}']
(out/'M3_OPERATOR_DIAG_SUMMARY.md').write_text('\n'.join(lines)+'\n')
with zipfile.ZipFile(out/'DGFS_M3_OPERATOR_DIAG.zip','w',zipfile.ZIP_DEFLATED) as z:
    for s in states: z.write(out/f'M3_OPERATOR_DIAG_{s}.json',f'M3_OPERATOR_DIAG_{s}.json')
    z.write(out/'M3_OPERATOR_DIAG_SUMMARY.md','M3_OPERATOR_DIAG_SUMMARY.md')
print('\n'.join(lines))
PY
echo "M3_OPERATOR_DIAG_AGG_COMPLETE"
SLURM
sed -i "s|AGG_LOG_PLACEHOLDER|$OUT/dgfs-m3-opagg-%j.out|" "$WORK/aggregate.slurm"

ARRAY=$(sbatch --parsable --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_SOLVER_SRC="$SRC",DGFS_WORK="$WORK",DGFS_OUTPUT_DIR="$OUT" "$WORK/run.slurm")
ARRAY=${ARRAY%%;*}
AGG=$(sbatch --parsable --dependency=afterok:$ARRAY --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_OUTPUT_DIR="$OUT",DGFS_M3_STAGE1_WORK="$STAGE1" "$WORK/aggregate.slurm")
AGG=${AGG%%;*}
echo "M3_OPERATOR_DIAG_ARRAY_JOB=$ARRAY"
echo "M3_OPERATOR_DIAG_AGG_JOB=$AGG"
echo "M3_OPERATOR_DIAG_SUMMARY=$OUT/M3_OPERATOR_DIAG_SUMMARY.md"
echo "M3_OPERATOR_DIAG_BOOTSTRAP_COMPLETE"
