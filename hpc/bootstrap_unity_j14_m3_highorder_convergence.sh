#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "M3_HIGHORDER_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CLOSE=${DGFS_CLOSEOUT:-$ROOT/j14novclose_20260826_211439}
STEADY=${DGFS_STEADY_DIR:-$CLOSE/m16_steady_20260831_125816}
SRC=${DGFS_SOLVER_SRC:-$STEADY/src}
OUT=${DGFS_OUTPUT_DIR:-$PWD}
STAGE1=${DGFS_M3_STAGE1_WORK:-$CLOSE/m3_stage1_20260904_014555}
CFG=$STAGE1/M3_raw/M3_raw.ini
SNAP=$STAGE1/M3_raw/dist_M3_raw-10.25.frfss
SCRATCH_BASE=${DGFS_M3_HIGHORDER_SCRATCH:-/scratch4/workspace/roohie_umass_edu-mfc-a40-cv}
STAMP=$(date +%Y%m%d_%H%M%S)
WORK=$SCRATCH_BASE/dgfs_m3_highorder_$STAMP
mkdir -p "$WORK" "$OUT"

for p in "$ENV_DIR/bin/python" "$SRC" "$CFG" "$SNAP"; do
    [[ -e "$p" ]] || { echo "MISSING=$p"; exit 2; }
done

# Reuse the already-audited fixed-state operator evaluator from the production branch.
EVAL_REF=${DGFS_M3_EVAL_REF:-8f2cf7496acb7e64f640ce9aa454152406429b51}
curl -fsSL \
  "https://raw.githubusercontent.com/Ehsan-Roohi/DGFS-BE-Solver/$EVAL_REF/cases/jcp2019_fig14b_normal_shock/novelty/j14_m3_resolution_audit.py" \
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
#SBATCH --mem=28G
#SBATCH --time=01:30:00
#SBATCH --array=0-4
#SBATCH --job-name=dgfs-m3-hiang
#SBATCH --output=ARRAY_LOG_PLACEHOLDER
set -Eeuo pipefail
trap 'rc=$?; echo "M3_HIGHORDER_JOB_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ENV_DIR=${DGFS_ENV:?}
SRC=${DGFS_SOLVER_SRC:?}
WORK=${DGFS_WORK:?}
CFG=${DGFS_CONFIG:?}
SNAP=${DGFS_SNAPSHOT:?}

MS=(35 47 60 78 96)
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
export CUDA_CACHE_PATH="$WORK/.cuda-cache-$i"
mkdir -p "$CUDA_CACHE_PATH"

echo "M3_HIGHORDER_START M=$M node=${SLURMD_NODENAME:-unknown}"
nvidia-smi --query-gpu=name --format=csv,noheader
"$ENV_DIR/bin/python" "$WORK/j14_m3_resolution_audit.py" \
  --config "$CFG" --snapshot "$SNAP" --label "$LABEL" \
  --Nv 40 --Nrho 40 --M "$M" --L 8.75 \
  --output "$WORK/$LABEL.json"
rm -rf "$CUDA_CACHE_PATH" || true
echo "M3_HIGHORDER_DONE M=$M"
SLURM
sed -i "s|ARRAY_LOG_PLACEHOLDER|$WORK/dgfs-m3-hiang-%A_%a.out|" "$WORK/run.slurm"

cat > "$WORK/aggregate.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:10:00
#SBATCH --job-name=dgfs-m3-hiagg
#SBATCH --output=AGG_LOG_PLACEHOLDER
set -Eeuo pipefail
WORK=${DGFS_WORK:?}
OUT=${DGFS_OUTPUT_DIR:?}
ENV_DIR=${DGFS_ENV:?}

"$ENV_DIR/bin/python" - "$WORK" "$OUT" <<'PY'
import json, pathlib, sys, zipfile
import numpy as np

work=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
orders=[35,47,60,78,96]
J={m:json.load(open(work/f'M{m}.json')) for m in orders}
ref=J[96]
fields=['qx','qz','Pdev_xx','Pxz','c4']

def vec(j,k):
    return np.array([r['raw_moments'][k] for r in j['records']],float)

def rel(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))

lines=[
'# Mach-3 high-order angular convergence audit','',
'Same M3_raw t=10.25 state, L=8.75, Nv=40, Nrho=40 for every case. Only M_omega changes. M_omega=96 is used only as the highest-order comparison in this audit, not assumed a priori to be exact.','',
'| M_omega | raw max inv defect | raw median inv defect | Euclidean distance L2 | fplus corr L2 | Maxwellian corr L2 |',
'|---:|---:|---:|---:|---:|---:|']
for m in orders:
    s=J[m]['summary']
    lines.append(f"| {m} | {s['raw_max_inv_defect']:.4e} | {s['raw_median_inv_defect']:.4e} | {s['median_rel_corr_l2']['euclidean']:.4e} | {s['median_rel_corr_l2']['fplus']:.4e} | {s['median_rel_corr_l2']['maxwellian']:.4e} |")

lines += ['', '## Raw high-order production relative to M_omega=96','',
'| M_omega | qx relL2 | qz relL2 | Pdev_xx relL2 | Pxz relL2 | c4 relL2 | max |',
'|---:|---:|---:|---:|---:|---:|---:|']
for m in orders:
    vals=[rel(vec(J[m],k),vec(ref,k)) for k in fields]
    lines.append(f"| {m} | "+' | '.join(f'{v:.4e}' for v in vals)+f" | {max(vals):.4e} |")

lines += ['', '## Adjacent-order changes in raw high-order production','',
'| pair | qx | qz | Pdev_xx | Pxz | c4 | max | Euclidean-distance relative change |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for a,b in zip(orders[:-1],orders[1:]):
    vals=[rel(vec(J[a],k),vec(J[b],k)) for k in fields]
    da=J[a]['summary']['median_rel_corr_l2']['euclidean']; db=J[b]['summary']['median_rel_corr_l2']['euclidean']
    drel=abs(da-db)/max(abs(db),1e-300)
    lines.append(f"| {a}->{b} | "+' | '.join(f'{v:.4e}' for v in vals)+f" | {max(vals):.4e} | {drel:.4e} |")

lines += ['', 'Interpretation rule: choose production M_omega from convergence of physical raw collision moments and the absolute conservation-distance scale together; do not reject an otherwise converged order solely because the relative change of an already tiny Euclidean distance is large.']

summary=out/'M3_HIGHORDER_CONVERGENCE_SUMMARY.md'
summary.write_text('\n'.join(lines)+'\n')
zip_path=out/'DGFS_M3_HIGHORDER_CONVERGENCE.zip'
zip_path.unlink(missing_ok=True)
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for m in orders: z.write(work/f'M{m}.json',f'M{m}.json')
    z.write(summary,summary.name)
print('\n'.join(lines))
print('BUNDLE='+str(zip_path))
PY

echo "M3_HIGHORDER_AGG_COMPLETE"
SLURM
sed -i "s|AGG_LOG_PLACEHOLDER|$OUT/dgfs-m3-hiagg-%j.out|" "$WORK/aggregate.slurm"

ARRAY=$(sbatch --parsable \
  --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_SOLVER_SRC="$SRC",DGFS_WORK="$WORK",DGFS_CONFIG="$CFG",DGFS_SNAPSHOT="$SNAP" \
  "$WORK/run.slurm")
ARRAY=${ARRAY%%;*}
AGG=$(sbatch --parsable --dependency=afterok:$ARRAY \
  --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_WORK="$WORK",DGFS_OUTPUT_DIR="$OUT" \
  "$WORK/aggregate.slurm")
AGG=${AGG%%;*}

echo "M3_HIGHORDER_ARRAY_JOB=$ARRAY"
echo "M3_HIGHORDER_AGG_JOB=$AGG"
echo "M3_HIGHORDER_WORK=$WORK"
echo "M3_HIGHORDER_SUMMARY=$OUT/M3_HIGHORDER_CONVERGENCE_SUMMARY.md"
echo "M3_HIGHORDER_BOOTSTRAP_COMPLETE"
