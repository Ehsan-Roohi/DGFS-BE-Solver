#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "BKW_BOXSCALE_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CLOSE=${DGFS_CLOSEOUT:-$ROOT/j14novclose_20260826_211439}
STEADY=${DGFS_STEADY_DIR:-$CLOSE/m16_steady_20260831_125816}
SRC=${DGFS_SOLVER_SRC:-$STEADY/src}
OUT=${DGFS_OUTPUT_DIR:-$PWD}
REF=${DGFS_REF:-d46406479360e57cbcacda16d21ad555b0dca33e}
BASECFG=$CLOSE/final_runs/run_M6_raw/p3b_M6_raw.ini
STAMP=$(date +%Y%m%d_%H%M%S)
WORK=$CLOSE/bkw_boxscale_$STAMP
mkdir -p "$WORK" "$OUT"
for p in "$ENV_DIR/bin/python" "$BASECFG" "$SRC"; do [[ -e "$p" ]] || { echo "MISSING=$p"; exit 2; }; done

curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/DGFS-BE-Solver/$REF/cases/jcp2019_fig14b_normal_shock/novelty/j14_bkw_box_scaling.py" -o "$WORK/j14_bkw_box_scaling.py"
"$ENV_DIR/bin/python" -m py_compile "$WORK/j14_bkw_box_scaling.py"

cat > "$WORK/run.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="v100&x86_64"
#SBATCH --mem=14G
#SBATCH --time=00:30:00
#SBATCH --array=0-3
#SBATCH --job-name=dgfs-bkw-box
#SBATCH --output=ARRAY_LOG_PLACEHOLDER
set -Eeuo pipefail
trap 'rc=$?; echo "BKW_BOXSCALE_JOB_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR
ENV_DIR=${DGFS_ENV:?}; SRC=${DGFS_SOLVER_SRC:?}; WORK=${DGFS_WORK:?}; OUT=${DGFS_OUTPUT_DIR:?}; CFG=${DGFS_CONFIG:?}
Ls=(5.25 7.00 8.75 10.50)
Nvs=(24 32 40 48)
Nrs=(24 32 40 48)
i=${SLURM_ARRAY_TASK_ID}; L=${Ls[$i]}; NV=${Nvs[$i]}; NR=${Nrs[$i]}
TAG=$(printf 'L%04.2f_Nv%d' "$L" "$NV" | tr '.' 'p')
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
export CUDA_CACHE_PATH="$WORK/.cuda-cache-$i"
mkdir -p "$CUDA_CACHE_PATH"
echo "BKW_BOXSCALE_START L=$L Nv=$NV Nrho=$NR M=6 node=${SLURMD_NODENAME:-unknown}"
nvidia-smi --query-gpu=name --format=csv,noheader
"$ENV_DIR/bin/python" "$WORK/j14_bkw_box_scaling.py" \
  --config "$CFG" --L "$L" --Nv "$NV" --Nrho "$NR" --M 6 --time 0.0 --fd-dt 1e-6 \
  --output-json "$OUT/BKW_BOXSCALE_${TAG}.json"
rm -rf "$CUDA_CACHE_PATH" || true
echo "BKW_BOXSCALE_DONE L=$L Nv=$NV"
SLURM
sed -i "s|ARRAY_LOG_PLACEHOLDER|$OUT/dgfs-bkw-box-%A_%a.out|" "$WORK/run.slurm"

cat > "$WORK/aggregate.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --job-name=dgfs-bkw-box-agg
#SBATCH --output=AGG_LOG_PLACEHOLDER
set -Eeuo pipefail
OUT=${DGFS_OUTPUT_DIR:?}; ENV_DIR=${DGFS_ENV:?}
"$ENV_DIR/bin/python" - "$OUT" <<'PY'
import json,pathlib,sys,zipfile
out=pathlib.Path(sys.argv[1])
spec=[(5.25,24),(7.00,32),(8.75,40),(10.50,48)]
def tag(L,Nv): return f'L{L:04.2f}_Nv{Nv}'.replace('.','p')
js={(L,Nv):json.load(open(out/f'BKW_BOXSCALE_{tag(L,Nv)}.json')) for L,Nv in spec}
lines=['# BKW velocity-box scaling audit','',
       'BKW t=0 (K=0.6), M_omega=6.  Velocity spacing is held approximately fixed while the box is enlarged.  Projection metrics are interpreted relative to raw Q on the same grid.','',
       '| L | Nv | dv | mode | inv defect | corr L2 | tail corr frac | low-support corr frac | outer-box corr frac | c4 error/raw | vx6 error/raw |',
       '|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
for L,Nv in spec:
    j=js[(L,Nv)]; g=j['grid']
    for m in ['euclidean','fplus','maxwellian']:
        r=j['results'][m]; rat=r['moment_error_ratio_to_raw']
        lines.append(f"| {L:.2f} | {Nv} | {g['dv']:.5f} | {m} | {r['invariant_defect']:.3e} | {r['relative_correction_l2']:.3e} | {r['tail_correction_fraction']:.3e} | {r['low_support_correction_fraction']:.3e} | {r['outer_box_correction_fraction']:.3e} | {rat['radial_c4']:.4f} | {rat['vx6']:.4f} |")
lines += ['','## Raw-grid diagnostics','',
          '| L | Nv | discrete mass | raw inv defect | raw exact-op L2 | low-support node frac |',
          '|---:|---:|---:|---:|---:|---:|']
for L,Nv in spec:
    j=js[(L,Nv)]
    lines.append(f"| {L:.2f} | {Nv} | {j['discrete_state']['mass']:.12f} | {j['results']['raw']['invariant_defect']:.3e} | {j['collision']['raw_operator_l2_error']:.3e} | {j['discrete_state']['low_support_node_fraction']:.3e} |")
(out/'BKW_BOXSCALE_SUMMARY.md').write_text('\n'.join(lines)+'\n')
with zipfile.ZipFile(out/'DGFS_BKW_BOXSCALE_AUDIT.zip','w',zipfile.ZIP_DEFLATED) as z:
    for L,Nv in spec: z.write(out/f'BKW_BOXSCALE_{tag(L,Nv)}.json',f'BKW_BOXSCALE_{tag(L,Nv)}.json')
    z.write(out/'BKW_BOXSCALE_SUMMARY.md','BKW_BOXSCALE_SUMMARY.md')
print('\n'.join(lines))
PY
echo "BKW_BOXSCALE_AGG_COMPLETE"
SLURM
sed -i "s|AGG_LOG_PLACEHOLDER|$OUT/dgfs-bkw-box-agg-%j.out|" "$WORK/aggregate.slurm"

ARRAY=$(sbatch --parsable --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_SOLVER_SRC="$SRC",DGFS_WORK="$WORK",DGFS_OUTPUT_DIR="$OUT",DGFS_CONFIG="$BASECFG" "$WORK/run.slurm")
ARRAY=${ARRAY%%;*}
AGG=$(sbatch --parsable --dependency=afterok:$ARRAY --export=ALL,DGFS_ENV="$ENV_DIR",DGFS_OUTPUT_DIR="$OUT" "$WORK/aggregate.slurm")
AGG=${AGG%%;*}
echo "BKW_BOXSCALE_ARRAY_JOB=$ARRAY"
echo "BKW_BOXSCALE_AGG_JOB=$AGG"
echo "BKW_BOXSCALE_OUTPUT=$OUT/BKW_BOXSCALE_SUMMARY.md"
echo "BKW_BOXSCALE_BOOTSTRAP_COMPLETE"
