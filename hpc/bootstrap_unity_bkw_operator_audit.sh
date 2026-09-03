#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "BKW_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CLOSE=${DGFS_CLOSEOUT:-$ROOT/j14novclose_20260826_211439}
STEADY=${DGFS_STEADY_DIR:-$CLOSE/m16_steady_20260831_125816}
SRC=${DGFS_SOLVER_SRC:-$STEADY/src}
OUT=${DGFS_OUTPUT_DIR:-$PWD}
REF=${DGFS_REF:-062d675f09ebb87b78c12d78b02b6ab7b8f4660a}
BASECFG=$CLOSE/final_runs/run_M6_raw/p3b_M6_raw.ini
STAMP=$(date +%Y%m%d_%H%M%S)
WORK=$CLOSE/bkw_operator_$STAMP
mkdir -p "$WORK" "$OUT"
for p in "$ENV_DIR/bin/python" "$BASECFG" "$SRC"; do [[ -e "$p" ]] || { echo "MISSING=$p"; exit 2; }; done

curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/DGFS-BE-Solver/$REF/cases/jcp2019_fig14b_normal_shock/novelty/j14_bkw_operator_audit.py" -o "$WORK/j14_bkw_operator_audit.py"
"$ENV_DIR/bin/python" -m py_compile "$WORK/j14_bkw_operator_audit.py"

cat > "$WORK/run.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="v100&x86_64"
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --job-name=dgfs-bkw-op
#SBATCH --output=ARRAY_LOG_PLACEHOLDER
set -Eeuo pipefail
trap 'rc=$?; echo "BKW_OPERATOR_JOB_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR
ENV_DIR=${DGFS_ENV:?}; SRC=${DGFS_SOLVER_SRC:?}; WORK=${DGFS_WORK:?}; OUT=${DGFS_OUTPUT_DIR:?}; CFG=${DGFS_CONFIG:?}
MVALS=(6 16); M=${MVALS[$SLURM_ARRAY_TASK_ID]}
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load cuda/12.6
module load openmpi/5.0.3-cuda12.6
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_CACHE_PATH="$WORK/.cuda-cache-$M"
mkdir -p "$CUDA_CACHE_PATH"
echo "BKW_OPERATOR_START M=$M node=${SLURMD_NODENAME:-unknown}"
nvidia-smi --query-gpu=name --format=csv,noheader
"$ENV_DIR/bin/python" "$WORK/j14_bkw_operator_audit.py" \
  --config "$CFG" --M "$M" --Nrho 32 --time 0.0 --fd-dt 1e-6 \
  --output-json "$OUT/BKW_OPERATOR_M${M}.json"
rm -rf "$CUDA_CACHE_PATH" || true
echo "BKW_OPERATOR_DONE M=$M"
SLURM
sed -i "s|ARRAY_LOG_PLACEHOLDER|$OUT/dgfs-bkw-op-%A_%a.out|" "$WORK/run.slurm"

cat > "$WORK/aggregate.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --job-name=dgfs-bkw-agg
#SBATCH --output=AGG_LOG_PLACEHOLDER
set -Eeuo pipefail
OUT=${DGFS_OUTPUT_DIR:?}; ENV_DIR=${DGFS_ENV:?}
"$ENV_DIR/bin/python" - "$OUT" <<'PY'
import json,pathlib,sys,zipfile
out=pathlib.Path(sys.argv[1]); Ms=[6,16]
js={M:json.load(open(out/f'BKW_OPERATOR_M{M}.json')) for M in Ms}
lines=['# Exact-BKW four-way operator audit','',
       'One raw-to-exact scalar collision-frequency calibration is fitted per angular order and then held fixed for raw, Euclidean, fplus and Maxwellian modes.','',
       '| M_omega | mode | invariant defect | full operator L2 error | core operator L2 error | vx4 rate error | c4 rate error | vx6 rate error |',
       '|---:|---|---:|---:|---:|---:|---:|---:|']
for M in Ms:
    for mode in ['raw','euclidean','fplus','maxwellian']:
        r=js[M]['results'][mode]; e=r['moment_rate_relative_error']
        lines.append(f"| {M} | {mode} | {r['invariant_defect']:.4e} | {r['relative_operator_l2_full']:.4e} | {r['relative_operator_l2_core']:.4e} | {e['vx4']:.4e} | {e['radial_c4']:.4e} | {e['vx6']:.4e} |")
lines += ['','## BKW state / calibration']
for M in Ms:
    j=js[M]
    lines.append(f"- M={M}: K={j['BKW']['K']:.6f}, discrete mass={j['discrete_BKW']['mass']:.12f}, min(f)={j['discrete_BKW']['min_f']:.4e}, alpha(raw->exact)={j['collision']['alpha_raw_to_exact']:.6e}")
(out/'BKW_OPERATOR_SUMMARY.md').write_text('\n'.join(lines)+'\n')
with zipfile.ZipFile(out/'DGFS_BKW_OPERATOR_AUDIT.zip','w',zipfile.ZIP_DEFLATED) as z:
    for M in Ms: z.write(out/f'BKW_OPERATOR_M{M}.json',f'BKW_OPERATOR_M{M}.json')
    z.write(out/'BKW_OPERATOR_SUMMARY.md','BKW_OPERATOR_SUMMARY.md')
print('\n'.join(lines))
PY
echo "BKW_OPERATOR_AGG_COMPLETE"
SLURM
sed -i "s|AGG_LOG_PLACEHOLDER|$OUT/dgfs-bkw-agg-%j.out|" "$WORK/aggregate.slurm"

ARRAY=$(sbatch --parsable --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_SOLVER_SRC="$SRC",DGFS_WORK="$WORK",DGFS_OUTPUT_DIR="$OUT",DGFS_CONFIG="$BASECFG" "$WORK/run.slurm")
ARRAY=${ARRAY%%;*}
AGG=$(sbatch --parsable --dependency=afterok:$ARRAY --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_OUTPUT_DIR="$OUT" "$WORK/aggregate.slurm")
AGG=${AGG%%;*}
echo "BKW_ARRAY_JOB=$ARRAY"
echo "BKW_AGG_JOB=$AGG"
echo "BKW_OUTPUT=$OUT/BKW_OPERATOR_SUMMARY.md"
echo "BKW_BOOTSTRAP_COMPLETE"
