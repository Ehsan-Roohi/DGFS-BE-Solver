#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "STEADY_PROJECTION_ALIGNED_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
POST_PY=${DGFS_POST_PYTHON:-$ENV_DIR/bin/python}
CLOSE=${DGFS_CLOSEOUT:-$ROOT/j14novclose_20260826_211439}
STEADY=${DGFS_STEADY_DIR:-$CLOSE/m16_steady_20260831_125816}
OUT=${DGFS_OUTPUT_DIR:-$PWD}
REF=${DGFS_REF:-a5002041976d8bc4b66c88b555bb44260d07077a}

M6R=$CLOSE/final_runs/run_M6_raw
M6F=$CLOSE/final_runs/run_M6_fplus
M16R=$STEADY/stage_1/M16_raw
M16F=$STEADY/stage_1/M16_fplus

latest_named () {
    local dir=$1 pat=$2
    find "$dir" -type f -name "$pat" -printf '%p\n' 2>/dev/null | sort -V | tail -1
}

M6R_S=$(latest_named "$M6R" 'dist_p3b_M6_raw-*.frfss')
M6F_S=$(latest_named "$M6F" 'dist_p3b_M6_fplus-*.frfss')

# Use newest Euclidean continuation that actually contains the converged t=340.25 snapshot.
EUC_S=$(find "$CLOSE" -type f -path '*/m6_euclidean_steady_*/segment_*/dist_p3b_M6_euclidean_steady-*.frfss' -printf '%p\n' 2>/dev/null | sort -V | tail -1)
[[ -n "$EUC_S" ]] || { echo "EUCLIDEAN_STEADY_SNAPSHOT_NOT_FOUND"; exit 2; }
EUC_DIR=$(dirname "$EUC_S")
EUC_CFG=$EUC_DIR/p3b_M6_euclidean_steady.ini

# M16 storage-safe campaign wrote full distributions sparsely (latest full checkpoint is
# expected near t=300.25) while cheap bulk moments continued through t=340.25.  For
# high-order moments/negativity we must use an actual full-distribution file, so resolve
# the newest one that exists rather than hard-coding t=340.25.
M16R_S=$(latest_named "$M16R" 'dist_p3b_M16_raw-*.frfss')
M16F_S=$(latest_named "$M16F" 'dist_p3b_M16_fplus-*.frfss')

for p in \
 "$M6R/p3b_M6_raw.ini" "$M6R/mesh.frfsm" "$M6R_S" \
 "$M6F/p3b_M6_fplus.ini" "$M6F/mesh.frfsm" "$M6F_S" \
 "$EUC_CFG" "$EUC_DIR/mesh.frfsm" "$EUC_S" \
 "$M16R/p3b_M16_raw.ini" "$M16R/mesh.frfsm" "$M16R_S" \
 "$M16F/p3b_M16_fplus.ini" "$M16F/mesh.frfsm" "$M16F_S"
do
    [[ -s "$p" ]] || { echo "MISSING_REQUIRED_FILE=$p"; exit 3; }
done

[[ -x "$POST_PY" ]] || { echo "POSTPROCESS_PYTHON_NOT_EXECUTABLE=$POST_PY"; exit 4; }
"$POST_PY" - <<'PY'
import h5py, matplotlib, numpy
print('POSTPROCESS_IMPORTS=PASS')
print('numpy='+numpy.__version__)
print('h5py='+h5py.__version__)
print('matplotlib='+matplotlib.__version__)
PY

mkdir -p "$OUT"
SCRIPT=$OUT/j14_steady_projection_compare_aligned.py
curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/DGFS-BE-Solver/$REF/cases/jcp2019_fig14b_normal_shock/novelty/j14_steady_projection_compare_aligned.py" -o "$SCRIPT"
"$POST_PY" -m py_compile "$SCRIPT"

echo "===== INPUT STEADY/FULL-DISTRIBUTION SNAPSHOTS ====="
echo "M6_raw=$M6R_S"
echo "M6_euclidean=$EUC_S"
echo "M6_fplus=$M6F_S"
echo "M16_raw=$M16R_S"
echo "M16_fplus=$M16F_S"
echo "NOTE: M16 full distributions are sparse storage-safe checkpoints; final M16 bulk-field stationarity was independently verified at 335.25->340.25."

echo "POSTPROCESS_PYTHON=$POST_PY"
cd "$OUT"
"$POST_PY" "$SCRIPT" \
 --M6_raw-config "$M6R/p3b_M6_raw.ini" --M6_raw-mesh "$M6R/mesh.frfsm" --M6_raw-snapshot "$M6R_S" \
 --M6_euclidean-config "$EUC_CFG" --M6_euclidean-mesh "$EUC_DIR/mesh.frfsm" --M6_euclidean-snapshot "$EUC_S" \
 --M6_fplus-config "$M6F/p3b_M6_fplus.ini" --M6_fplus-mesh "$M6F/mesh.frfsm" --M6_fplus-snapshot "$M6F_S" \
 --M16_raw-config "$M16R/p3b_M16_raw.ini" --M16_raw-mesh "$M16R/mesh.frfsm" --M16_raw-snapshot "$M16R_S" \
 --M16_fplus-config "$M16F/p3b_M16_fplus.ini" --M16_fplus-mesh "$M16F/mesh.frfsm" --M16_fplus-snapshot "$M16F_S" \
 --out-dir "$OUT"

echo "STEADY_PROJECTION_ALIGNED_COMPLETE"
ls -lh \
 "$OUT/DGFS_STEADY_PROJECTION_ALIGNED.zip" \
 "$OUT/STEADY_PROJECTION_ALIGNED_SUMMARY.md" \
 "$OUT/FIG_STEADY_HYDRO_ALIGNED.png" \
 "$OUT/FIG_STEADY_KINETIC_ALIGNED.png" \
 "$OUT/FIG_STEADY_SYMMETRY_ALIGNED.png" \
 "$OUT/FIG_STEADY_NEGATIVITY_ALIGNED.png"
