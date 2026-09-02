#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "M6_EUCLIDEAN_STEADY_BOOTSTRAP_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CLOSE=${DGFS_CLOSEOUT:-$ROOT/j14novclose_20260826_211439}
STEADY=${DGFS_STEADY_DIR:-$CLOSE/m16_steady_20260831_125816}
SRC=${DGFS_SOLVER_SRC:-$STEADY/src}
REF=${DGFS_REF:-2cab177b247e451fc875a3ad784b7d951591837f}
OUT=${DGFS_OUTPUT_DIR:-$PWD}
SEG_DUR=${DGFS_EUCLIDEAN_SEGMENT_DURATION:-20.0}
MAX_SEG=${DGFS_EUCLIDEAN_MAX_SEGMENTS:-4}
DIST_DT=${DGFS_EUCLIDEAN_DIST_DT:-10.0}
MOM_DT=${DGFS_EUCLIDEAN_MOM_DT:-2.5}
STATION_TOL=${DGFS_EUCLIDEAN_STATION_TOL:-5e-4}
STAMP=$(date +%Y%m%d_%H%M%S)
WORK=$CLOSE/m6_euclidean_steady_$STAMP
M6=$CLOSE/final_runs/run_M6_raw

mkdir -p "$WORK" "$OUT"
test -x "$ENV_DIR/bin/python"
test -d "$SRC"
test -s "$M6/mesh.frfsm"

SHORTROOT=$({ find "$CLOSE" -maxdepth 1 -type d -name 'm6_solution_compare_*' -printf '%T@ %p\n' 2>/dev/null || true; } | sort -nr | head -1 | cut -d' ' -f2-)
[[ -d "$SHORTROOT/M6_short_euclidean" ]] || { echo "M6_EUCLIDEAN_SHORT_RUN_NOT_FOUND"; exit 2; }
SHORTCASE="$SHORTROOT/M6_short_euclidean"
SOURCE=$({ find "$SHORTCASE" -maxdepth 1 -type f -name 'dist_p3b_M6_short_euclidean-*.frfss' -printf '%p\n' | sort -V | tail -1; })
[[ -s "$SOURCE" ]] || { echo "M6_EUCLIDEAN_SOURCE_NOT_FOUND"; exit 3; }
BASECFG="$SHORTCASE/p3b_M6_short_euclidean.ini"
test -s "$BASECFG"

read -r T0 < <("$ENV_DIR/bin/python" - "$SOURCE" <<'PY'
import re,sys
m=re.search(r'-([0-9]+(?:\.[0-9]+)?)\.frfss$',sys.argv[1])
if not m: raise SystemExit('cannot parse source time')
print(float(m.group(1)))
PY
)

curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/DGFS-BE-Solver/$REF/cases/jcp2019_fig14b_normal_shock/novelty/j14_state_stationarity.py" -o "$WORK/j14_state_stationarity.py"
curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/DGFS-BE-Solver/bc94b13364e55ff96fb6b32e2ecca2b19d7daac5/cases/jcp2019_fig14b_normal_shock/novelty/j14_solution_state_diagnostics.py" -o "$WORK/j14_solution_state_diagnostics.py"
"$ENV_DIR/bin/python" -m py_compile "$WORK/j14_state_stationarity.py" "$WORK/j14_solution_state_diagnostics.py"

cat > "$WORK/run.slurm" <<'SLURM'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="v100&x86_64"
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --job-name=dgfs-m6-euc
#SBATCH --output=SLURM_LOG_PLACEHOLDER
set -Eeuo pipefail
trap 'rc=$?; echo "M6_EUCLIDEAN_STEADY_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:?}; ENV_DIR=${DGFS_ENV:?}; SRC=${DGFS_SOLVER_SRC:?}; CLOSE=${DGFS_CLOSEOUT:?}
WORK=${DGFS_WORK:?}; OUT=${DGFS_OUTPUT_DIR:?}; SOURCE=${DGFS_SOURCE:?}; BASECFG=${DGFS_BASECFG:?}; MESH=${DGFS_MESH:?}
T0=${DGFS_T0:?}; SEG_DUR=${DGFS_SEG_DUR:?}; MAX_SEG=${DGFS_MAX_SEG:?}; DIST_DT=${DGFS_DIST_DT:?}; MOM_DT=${DGFS_MOM_DT:?}; TOL=${DGFS_STATION_TOL:?}

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

echo "M6_EUCLIDEAN_STEADY_START source=$SOURCE t0=$T0 node=${SLURMD_NODENAME:-unknown}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CURRENT="$SOURCE"
START="$T0"
CONVERGED=0
LAST_JSON=""
HISTORY=()

for ((SEG=1; SEG<=MAX_SEG; SEG++)); do
    END=$("$ENV_DIR/bin/python" -c 'import sys; print(float(sys.argv[1])+float(sys.argv[2]))' "$START" "$SEG_DUR")
    CASE="$WORK/segment_$SEG"
    mkdir -p "$CASE"

    "$ENV_DIR/bin/python" "$SRC/cases/jcp2019_fig14b_normal_shock/phase3/make_restart_configs.py" \
      --base "$BASECFG" --tstart "$START" --tend "$END" \
      --dist-dt-out "$DIST_DT" --mom-dt-out "$MOM_DT" \
      --residual-file kinetic_residual_euclidean.csv \
      --runs 'M6_euclidean_steady:32:6:euclidean' --out-dir "$CASE"

    cp "$MESH" "$CASE/mesh.frfsm"
    ln -sf "$CURRENT" "$CASE/dist-source.frfss"
    cd "$CASE"
    echo "M6_EUCLIDEAN_SEGMENT_START segment=$SEG interval=$START,$END input=$CURRENT"
    "$ENV_DIR/bin/python" -m frfs restart mesh.frfsm dist-source.frfss p3b_M6_euclidean_steady.ini -b cuda 2>&1 | tee solver.log

    mapfile -t SNAPS < <(find "$CASE" -maxdepth 1 -type f -name 'dist_p3b_M6_euclidean_steady-*.frfss' -printf '%p\n' | sort -V)
    [[ ${#SNAPS[@]} -ge 2 ]] || { echo "M6_EUCLIDEAN_NEED_TWO_SNAPSHOTS segment=$SEG count=${#SNAPS[@]}"; exit 4; }
    OLD=${SNAPS[$((${#SNAPS[@]}-2))]}
    NEW=${SNAPS[$((${#SNAPS[@]}-1))]}
    LAST_JSON="$CASE/stationarity.json"
    "$ENV_DIR/bin/python" "$WORK/j14_state_stationarity.py" \
      --config "$CASE/p3b_M6_euclidean_steady.ini" --old "$OLD" --new "$NEW" \
      --tolerance "$TOL" --output-json "$LAST_JSON" | tee "$CASE/stationarity.log"

    HISTORY+=("$NEW")
    CURRENT="$NEW"
    START="$END"

    if "$ENV_DIR/bin/python" - "$LAST_JSON" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['pass'] else 1)
PY
    then
        CONVERGED=1
        echo "M6_EUCLIDEAN_STATIONARY segment=$SEG final=$CURRENT"
        break
    fi

    # Keep the final state for recovery and the JSON evidence; remove the mid-segment full distribution.
    rm -f "$OLD"
    echo "M6_EUCLIDEAN_CONTINUE segment=$SEG next_start=$START"
done

# State-level diagnostic from the original t0 state to every retained segment-final state.
FINALCFG=$(dirname "$CURRENT")/p3b_M6_euclidean_steady.ini
"$ENV_DIR/bin/python" "$WORK/j14_solution_state_diagnostics.py" \
  --config "$FINALCFG" --mesh "$MESH" --label M6_euclidean_steady --t0 "$T0" \
  --snapshots "$SOURCE" "${HISTORY[@]}" --output-prefix "$OUT/M6_EUCLIDEAN_STEADY"

"$ENV_DIR/bin/python" - "$OUT" "$LAST_JSON" "$CURRENT" "$CONVERGED" <<'PY'
import json,pathlib,sys,zipfile
out=pathlib.Path(sys.argv[1]); sj=pathlib.Path(sys.argv[2]); current=sys.argv[3]; converged=bool(int(sys.argv[4]))
s=json.loads(sj.read_text())
g=json.loads((out/'M6_EUCLIDEAN_STEADY.json').read_text())
final=g['global_series'][-1]
lines=[
'# M6 Euclidean steady continuation','',
f'- stationary gate: {"PASS" if converged else "NOT YET"}',
f'- final snapshot: `{current}`',
f'- stationarity tolerance: {s["tolerance"]:.3e}',
f'- maximum state relative-L2 drift: {s["max_relative_l2"]:.6e}',
f'- final min(f): {final["min_f"]:.6e}',
f'- final max negative-mass fraction: {final["max_negative_mass_fraction"]:.6e}','',
'| field | relative L2 drift |','|---|---:|']
for k,v in s['relative_l2'].items(): lines.append(f'| {k} | {v:.6e} |')
(out/'M6_EUCLIDEAN_STEADY_SUMMARY.md').write_text('\n'.join(lines)+'\n')
summary={'stationary':converged,'final_snapshot':current,'stationarity':s,'final_global':final}
(out/'M6_EUCLIDEAN_STEADY_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
zp=out/'DGFS_M6_EUCLIDEAN_STEADY.zip'; zp.unlink(missing_ok=True)
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
    for name in ['M6_EUCLIDEAN_STEADY.json','M6_EUCLIDEAN_STEADY.points.csv','M6_EUCLIDEAN_STEADY.global.csv','M6_EUCLIDEAN_STEADY_SUMMARY.md','M6_EUCLIDEAN_STEADY_SUMMARY.json']:
        p=out/name
        if p.exists(): z.write(p,p.name)
print('\n'.join(lines)); print('BUNDLE='+str(zp))
PY

rm -rf "$CUDA_CACHE_PATH" || true
if (( CONVERGED )); then
    echo "M6_EUCLIDEAN_STEADY_COMPLETE=PASS"
else
    echo "M6_EUCLIDEAN_STEADY_COMPLETE=NEEDS_CONTINUATION"
fi
ls -lh "$OUT/DGFS_M6_EUCLIDEAN_STEADY.zip" "$OUT/M6_EUCLIDEAN_STEADY_SUMMARY.md"
SLURM
sed -i "s|SLURM_LOG_PLACEHOLDER|$OUT/dgfs-m6-euclidean-%j.out|" "$WORK/run.slurm"

JOB=$(sbatch --parsable \
  --export=ALL,DGFS_ROOT="$ROOT",DGFS_ENV="$ENV_DIR",DGFS_SOLVER_SRC="$SRC",DGFS_CLOSEOUT="$CLOSE",DGFS_WORK="$WORK",DGFS_OUTPUT_DIR="$OUT",DGFS_SOURCE="$SOURCE",DGFS_BASECFG="$BASECFG",DGFS_MESH="$M6/mesh.frfsm",DGFS_T0="$T0",DGFS_SEG_DUR="$SEG_DUR",DGFS_MAX_SEG="$MAX_SEG",DGFS_DIST_DT="$DIST_DT",DGFS_MOM_DT="$MOM_DT",DGFS_STATION_TOL="$STATION_TOL" \
  "$WORK/run.slurm")
JOB=${JOB%%;*}

echo "M6_EUCLIDEAN_STEADY_JOB=$JOB"
echo "M6_EUCLIDEAN_STEADY_WORK=$WORK"
echo "M6_EUCLIDEAN_STEADY_SOURCE=$SOURCE"
echo "M6_EUCLIDEAN_STEADY_FINAL_ZIP=$OUT/DGFS_M6_EUCLIDEAN_STEADY.zip"
echo "M6_EUCLIDEAN_STEADY_BOOTSTRAP_COMPLETE"
