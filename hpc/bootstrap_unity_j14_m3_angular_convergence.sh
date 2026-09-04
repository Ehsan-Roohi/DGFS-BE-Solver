#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "M3_ANGULAR_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CLOSE=${DGFS_CLOSEOUT:-$ROOT/j14novclose_20260826_211439}
STEADY=${DGFS_STEADY_DIR:-$CLOSE/m16_steady_20260831_125816}
SRC=${DGFS_SOLVER_SRC:-$STEADY/src}
OUT=${DGFS_OUTPUT_DIR:-$PWD}
REF=${DGFS_REF:-8f2cf7496acb7e64f640ce9aa454152406429b51}
STAGE1=${DGFS_M3_STAGE1_WORK:-$CLOSE/m3_stage1_20260904_014555}
CFG=$STAGE1/M3_raw/M3_raw.ini
SNAP=$STAGE1/M3_raw/dist_M3_raw-10.25.frfss
SCRATCH_BASE=${DGFS_M3_ANGULAR_SCRATCH:-/scratch4/workspace/roohie_umass_edu-mfc-a40-cv}
STAMP=$(date +%Y%m%d_%H%M%S)
WORK=$SCRATCH_BASE/dgfs_m3_angular_$STAMP
mkdir -p "$WORK" "$OUT"

for p in "$ENV_DIR/bin/python" "$SRC" "$CFG" "$SNAP"; do
  [[ -e "$p" ]] || { echo "MISSING=$p"; exit 2; }
done

curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/DGFS-BE-Solver/$REF/cases/jcp2019_fig14b_normal_shock/novelty/j14_m3_resolution_audit.py" \
  -o "$WORK/j14_m3_resolution_audit.py"
"$ENV_DIR/bin/python" -m py_compile "$WORK/j14_m3_resolution_audit.py"

cat > "$WORK/run.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="v100&x86_64"
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --job-name=dgfs-m3-ang
#SBATCH --output=ARRAY_LOG_PLACEHOLDER
set -Eeuo pipefail
trap 'rc=$?; echo "M3_ANGULAR_JOB_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ENV_DIR=${DGFS_ENV:?}; SRC=${DGFS_SOLVER_SRC:?}; WORK=${DGFS_WORK:?}; CFG=${DGFS_CONFIG:?}; SNAP=${DGFS_SNAPSHOT:?}
MS=(6 16 24 35 47)
i=$SLURM_ARRAY_TASK_ID
M=${MS[$i]}
LABEL=M${M}

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
export CUDA_CACHE_PATH="$WORK/.cuda-cache-$M"
mkdir -p "$CUDA_CACHE_PATH"

echo "M3_ANGULAR_START M=$M node=${SLURMD_NODENAME:-unknown}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$ENV_DIR/bin/python" "$WORK/j14_m3_resolution_audit.py" \
  --config "$CFG" --snapshot "$SNAP" --label "$LABEL" \
  --Nv 40 --Nrho 40 --M "$M" --L 8.75 \
  --output "$WORK/$LABEL.json"

rm -rf "$CUDA_CACHE_PATH" || true
echo "M3_ANGULAR_DONE M=$M"
SLURM
sed -i "s|ARRAY_LOG_PLACEHOLDER|$WORK/dgfs-m3-ang-%A_%a.out|" "$WORK/run.slurm"

cat > "$WORK/aggregate.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --job-name=dgfs-m3-angagg
#SBATCH --output=AGG_LOG_PLACEHOLDER
set -Eeuo pipefail
WORK=${DGFS_WORK:?}; OUT=${DGFS_OUTPUT_DIR:?}; ENV_DIR=${DGFS_ENV:?}

"$ENV_DIR/bin/python" - "$WORK" "$OUT" <<'PY'
import json, math, pathlib, sys, zipfile
work=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
Ms=[6,16,24,35,47]
J={M:json.load(open(work/f'M{M}.json')) for M in Ms}
ref=J[47]
fields=['qx','qz','Pdev_xx','Pxz','c4']

def rel(a,b):
    num=math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
    den=math.sqrt(sum(y*y for y in b)) or 1e-300
    return num/den

lines=['# Mach-3 angular convergence audit','',
       'Same M3_raw t=10.25 state and same velocity grid for every case: L=8.75, Nv=40, Nrho=40. Only angular order M_omega changes.','',
       'The Euclidean correction norm is reported as a weighting-independent distance from raw Q to the conservative subspace. The cancellation-normalized invariant defect is retained only as a secondary diagnostic.','',
       '| M_omega | raw max inv defect | raw median inv defect | median Euclidean distance L2 | median fplus corr L2 | median Maxwellian corr L2 |',
       '|---:|---:|---:|---:|---:|---:|']
for M in Ms:
    s=J[M]['summary']
    lines.append(f"| {M} | {s['raw_max_inv_defect']:.4e} | {s['raw_median_inv_defect']:.4e} | {s['median_rel_corr_l2']['euclidean']:.4e} | {s['median_rel_corr_l2']['fplus']:.4e} | {s['median_rel_corr_l2']['maxwellian']:.4e} |")

lines += ['','## Raw high-order collision production versus M_omega=47 on the same velocity grid','',
          '| M_omega | qx relL2 | qz relL2 | Pdev_xx relL2 | Pxz relL2 | c4 relL2 |',
          '|---:|---:|---:|---:|---:|---:|']
for M in Ms:
    vals={}
    for k in fields:
        a=[r['raw_moments'][k] for r in J[M]['records']]
        b=[r['raw_moments'][k] for r in ref['records']]
        vals[k]=rel(a,b)
    lines.append(f"| {M} | {vals['qx']:.4e} | {vals['qz']:.4e} | {vals['Pdev_xx']:.4e} | {vals['Pxz']:.4e} | {vals['c4']:.4e} |")

# Direct M35-vs-M47 convergence line for a simple production-choice gate.
vals35={}
for k in fields:
    a=[r['raw_moments'][k] for r in J[35]['records']]
    b=[r['raw_moments'][k] for r in J[47]['records']]
    vals35[k]=rel(a,b)
max35=max(vals35.values())
D35=J[35]['summary']['median_rel_corr_l2']['euclidean']
D47=J[47]['summary']['median_rel_corr_l2']['euclidean']
drel=abs(D35-D47)/max(abs(D47),1e-300)
lines += ['', '## Production-choice gate', '',
          f'- max raw high-order relL2 difference M35 vs M47: {max35:.6e}',
          f'- Euclidean conservation-distance relative difference M35 vs M47: {drel:.6e}',
          f'- M35_APPROX_CONVERGED={"PASS" if max35 < 1e-2 and drel < 0.10 else "FAIL"}',
          '- Do not select a production angular order from raw max invariant defect alone; it is a cancellation-normalized maximum and need not be monotone.']

summary=out/'M3_ANGULAR_CONVERGENCE_SUMMARY.md'
summary.write_text('\n'.join(lines)+'\n')
zip_path=out/'DGFS_M3_ANGULAR_CONVERGENCE.zip'
zip_path.unlink(missing_ok=True)
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for M in Ms: z.write(work/f'M{M}.json',f'M{M}.json')
    z.write(summary,summary.name)
print('\n'.join(lines))
PY

echo "M3_ANGULAR_AGG_COMPLETE"
SLURM
sed -i "s|AGG_LOG_PLACEHOLDER|$OUT/dgfs-m3-angagg-%j.out|" "$WORK/aggregate.slurm"

ARRAY=$(sbatch --parsable \
  --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_SOLVER_SRC="$SRC",DGFS_WORK="$WORK",DGFS_CONFIG="$CFG",DGFS_SNAPSHOT="$SNAP" \
  "$WORK/run.slurm")
ARRAY=${ARRAY%%;*}
AGG=$(sbatch --parsable --dependency=afterok:$ARRAY \
  --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_WORK="$WORK",DGFS_OUTPUT_DIR="$OUT" \
  "$WORK/aggregate.slurm")
AGG=${AGG%%;*}

echo "M3_ANGULAR_ARRAY_JOB=$ARRAY"
echo "M3_ANGULAR_AGG_JOB=$AGG"
echo "M3_ANGULAR_WORK=$WORK"
echo "M3_ANGULAR_SUMMARY=$OUT/M3_ANGULAR_CONVERGENCE_SUMMARY.md"
echo "M3_ANGULAR_BOOTSTRAP_COMPLETE"
