#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "M3_RESOLUTION_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR
ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CLOSE=${DGFS_CLOSEOUT:-$ROOT/j14novclose_20260826_211439}
STEADY=${DGFS_STEADY_DIR:-$CLOSE/m16_steady_20260831_125816}
SRC=${DGFS_SOLVER_SRC:-$STEADY/src}
OUT=${DGFS_OUTPUT_DIR:-$PWD}
REF=${DGFS_REF:-9c57960eb97719d0129c89ec6221e4a30207d67a}
STAGE1=${DGFS_M3_STAGE1_WORK:-$CLOSE/m3_stage1_20260904_014555}
CFG=$STAGE1/M3_raw/M3_raw.ini
SNAP=$STAGE1/M3_raw/dist_M3_raw-10.25.frfss
SCRATCH_BASE=${DGFS_M3_RES_SCRATCH:-/scratch4/workspace/roohie_umass_edu-mfc-a40-cv}
STAMP=$(date +%Y%m%d_%H%M%S)
WORK=$SCRATCH_BASE/dgfs_m3_resolution_$STAMP
mkdir -p "$WORK" "$OUT"
for p in "$ENV_DIR/bin/python" "$SRC" "$CFG" "$SNAP"; do [[ -e "$p" ]] || { echo "MISSING=$p"; exit 2; }; done
curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/DGFS-BE-Solver/$REF/cases/jcp2019_fig14b_normal_shock/novelty/j14_m3_resolution_audit.py" -o "$WORK/j14_m3_resolution_audit.py"
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
#SBATCH --time=00:45:00
#SBATCH --array=0-5
#SBATCH --job-name=dgfs-m3-res
#SBATCH --output=ARRAY_LOG_PLACEHOLDER
set -Eeuo pipefail
trap 'rc=$?; echo "M3_RESOLUTION_JOB_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR
ENV_DIR=${DGFS_ENV:?}; SRC=${DGFS_SOLVER_SRC:?}; WORK=${DGFS_WORK:?}; CFG=${DGFS_CONFIG:?}; SNAP=${DGFS_SNAPSHOT:?}
LABELS=(base_M6 ang_M16 ang_M24 radial_M16_N64 box10_M16 box10_M24)
NVS=(40 40 40 40 48 48)
NRS=(40 40 40 64 48 48)
MS=(6 16 24 16 16 24)
LS=(8.75 8.75 8.75 8.75 10.50 10.50)
i=$SLURM_ARRAY_TASK_ID; LABEL=${LABELS[$i]}; NV=${NVS[$i]}; NR=${NRS[$i]}; M=${MS[$i]}; L=${LS[$i]}
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge; module load cuda/12.6; module load openmpi/5.0.3-cuda12.6; module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$ENV_DIR"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}" CUDA_CACHE_PATH="$WORK/.cuda-cache-$i"; mkdir -p "$CUDA_CACHE_PATH"
echo "M3_RESOLUTION_START label=$LABEL Nv=$NV Nrho=$NR M=$M L=$L node=${SLURMD_NODENAME:-unknown}"
nvidia-smi --query-gpu=name --format=csv,noheader
"$ENV_DIR/bin/python" "$WORK/j14_m3_resolution_audit.py" --config "$CFG" --snapshot "$SNAP" --label "$LABEL" --Nv "$NV" --Nrho "$NR" --M "$M" --L "$L" --output "$WORK/$LABEL.json"
rm -rf "$CUDA_CACHE_PATH" || true
echo "M3_RESOLUTION_DONE label=$LABEL"
SLURM
sed -i "s|ARRAY_LOG_PLACEHOLDER|$WORK/dgfs-m3-res-%A_%a.out|" "$WORK/run.slurm"
cat > "$WORK/aggregate.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --job-name=dgfs-m3-resagg
#SBATCH --output=AGG_LOG_PLACEHOLDER
set -Eeuo pipefail
WORK=${DGFS_WORK:?}; OUT=${DGFS_OUTPUT_DIR:?}; ENV_DIR=${DGFS_ENV:?}
"$ENV_DIR/bin/python" - "$WORK" "$OUT" <<'PY'
import json,math,pathlib,sys,zipfile,numpy as np
work=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); labels=['base_M6','ang_M16','ang_M24','radial_M16_N64','box10_M16','box10_M24']
J={x:json.load(open(work/f'{x}.json')) for x in labels}; ref=J['box10_M24']
lines=['# Mach-3 raw collision-operator resolution audit','',
'All cases evaluate the same M3_raw t=10.25 state. Larger velocity boxes use exact zero-padding at fixed dv=0.4375; no interpolation is used.','',
'| label | L | Nv | Nrho | M_omega | raw max inv defect | median fplus corr L2 | median Maxwellian corr L2 | median theta=.25 corr L2 |',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for x in labels:
 j=J[x]; g=j['grid']; s=j['summary']; lines.append(f"| {x} | {g['L']:.2f} | {g['Nv']} | {g['Nrho']} | {g['M_omega']} | {s['raw_max_inv_defect']:.4e} | {s['median_rel_corr_l2']['fplus']:.4e} | {s['median_rel_corr_l2']['maxwellian']:.4e} | {s['median_rel_corr_l2']['theta025']:.4e} |")
lines += ['','## Raw high-order production versus high-resolution reference (box10_M24)','', '| label | qx relL2 | qz relL2 | Pdev_xx relL2 | Pxz relL2 | c4 relL2 |','|---|---:|---:|---:|---:|---:|']
for x in labels:
 vals={}
 for k in ['qx','qz','Pdev_xx','Pxz','c4']:
  a=np.array([r['raw_moments'][k] for r in J[x]['records']],float); b=np.array([r['raw_moments'][k] for r in ref['records']],float); vals[k]=np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)
 lines.append(f"| {x} | {vals['qx']:.4e} | {vals['qz']:.4e} | {vals['Pdev_xx']:.4e} | {vals['Pxz']:.4e} | {vals['c4']:.4e} |")
(out/'M3_RESOLUTION_AUDIT_SUMMARY.md').write_text('\n'.join(lines)+'\n')
with zipfile.ZipFile(out/'DGFS_M3_RESOLUTION_AUDIT.zip','w',zipfile.ZIP_DEFLATED) as z:
 for x in labels: z.write(work/f'{x}.json',f'{x}.json')
 z.write(out/'M3_RESOLUTION_AUDIT_SUMMARY.md','M3_RESOLUTION_AUDIT_SUMMARY.md')
print('\n'.join(lines))
PY
echo "M3_RESOLUTION_AGG_COMPLETE"
SLURM
sed -i "s|AGG_LOG_PLACEHOLDER|$OUT/dgfs-m3-resagg-%j.out|" "$WORK/aggregate.slurm"
ARRAY=$(sbatch --parsable --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_SOLVER_SRC="$SRC",DGFS_WORK="$WORK",DGFS_CONFIG="$CFG",DGFS_SNAPSHOT="$SNAP" "$WORK/run.slurm"); ARRAY=${ARRAY%%;*}
AGG=$(sbatch --parsable --dependency=afterok:$ARRAY --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_WORK="$WORK",DGFS_OUTPUT_DIR="$OUT" "$WORK/aggregate.slurm"); AGG=${AGG%%;*}
echo "M3_RESOLUTION_ARRAY_JOB=$ARRAY"
echo "M3_RESOLUTION_AGG_JOB=$AGG"
echo "M3_RESOLUTION_WORK=$WORK"
echo "M3_RESOLUTION_SUMMARY=$OUT/M3_RESOLUTION_AUDIT_SUMMARY.md"
echo "M3_RESOLUTION_BOOTSTRAP_COMPLETE"
