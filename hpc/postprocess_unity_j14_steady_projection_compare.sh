#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "STEADY_PROJECTION_COMPARE_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CLOSE=${DGFS_CLOSEOUT:-$ROOT/j14novclose_20260826_211439}
STEADY=${DGFS_STEADY_DIR:-$CLOSE/m16_steady_20260831_125816}
OUT=${DGFS_OUTPUT_DIR:-$PWD}
REF=${DGFS_REF:-1b716312021a812f158ee1b39d4d2eb9152c205a}

M6R=$CLOSE/final_runs/run_M6_raw
M6F=$CLOSE/final_runs/run_M6_fplus
M16R=$STEADY/stage_1/M16_raw
M16F=$STEADY/stage_1/M16_fplus

latest_named () {
    local dir=$1 pat=$2
    find "$dir" -maxdepth 2 -type f -name "$pat" -printf '%p\n' 2>/dev/null | sort -V | tail -1
}

M6R_S=$(latest_named "$M6R" 'dist_p3b_M6_raw-*.frfss')
M6F_S=$(latest_named "$M6F" 'dist_p3b_M6_fplus-*.frfss')

# Euclidean steady state: newest converged continuation directory, then highest-time retained snapshot.
EUCROOT=$(find "$CLOSE" -maxdepth 1 -type d -name 'm6_euclidean_steady_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
EUC_S=$(find "$EUCROOT" -type f -name 'dist_p3b_M6_euclidean_steady-*.frfss' -printf '%p\n' 2>/dev/null | sort -V | tail -1)
EUC_DIR=$(dirname "$EUC_S")
EUC_CFG=$EUC_DIR/p3b_M6_euclidean_steady.ini

M16R_S=$M16R/dist_p3b_M16_raw-340.25.frfss
M16F_S=$M16F/dist_p3b_M16_fplus-340.25.frfss

for p in \
 "$M6R/p3b_M6_raw.ini" "$M6R/mesh.frfsm" "$M6R_S" \
 "$M6F/p3b_M6_fplus.ini" "$M6F/mesh.frfsm" "$M6F_S" \
 "$EUC_CFG" "$EUC_DIR/mesh.frfsm" "$EUC_S" \
 "$M16R/p3b_M16_raw.ini" "$M16R/mesh.frfsm" "$M16R_S" \
 "$M16F/p3b_M16_fplus.ini" "$M16F/mesh.frfsm" "$M16F_S"
do
    [[ -s "$p" ]] || { echo "MISSING_REQUIRED_FILE=$p"; exit 2; }
done

mkdir -p "$OUT"
SCRIPT=$OUT/j14_steady_projection_compare.py
curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/DGFS-BE-Solver/$REF/cases/jcp2019_fig14b_normal_shock/novelty/j14_steady_projection_compare.py" -o "$SCRIPT"
"$ENV_DIR/bin/python" -m py_compile "$SCRIPT"

echo "===== STEADY SNAPSHOTS ====="
echo "M6_raw=$M6R_S"
echo "M6_euclidean=$EUC_S"
echo "M6_fplus=$M6F_S"
echo "M16_raw=$M16R_S"
echo "M16_fplus=$M16F_S"

cd "$OUT"
"$ENV_DIR/bin/python" "$SCRIPT" \
 --M6_raw-config "$M6R/p3b_M6_raw.ini" --M6_raw-mesh "$M6R/mesh.frfsm" --M6_raw-snapshot "$M6R_S" \
 --M6_euclidean-config "$EUC_CFG" --M6_euclidean-mesh "$EUC_DIR/mesh.frfsm" --M6_euclidean-snapshot "$EUC_S" \
 --M6_fplus-config "$M6F/p3b_M6_fplus.ini" --M6_fplus-mesh "$M6F/mesh.frfsm" --M6_fplus-snapshot "$M6F_S" \
 --M16_raw-config "$M16R/p3b_M16_raw.ini" --M16_raw-mesh "$M16R/mesh.frfsm" --M16_raw-snapshot "$M16R_S" \
 --M16_fplus-config "$M16F/p3b_M16_fplus.ini" --M16_fplus-mesh "$M16F/mesh.frfsm" --M16_fplus-snapshot "$M16F_S" \
 --out-dir "$OUT"

echo "STEADY_PROJECTION_COMPARE_COMPLETE"
ls -lh \
 "$OUT/DGFS_STEADY_PROJECTION_COMPARE.zip" \
 "$OUT/STEADY_PROJECTION_COMPARE_SUMMARY.md" \
 "$OUT/FIG_STEADY_KINETIC_MOMENTS.png" \
 "$OUT/FIG_STEADY_SYMMETRY.png" \
 "$OUT/FIG_STEADY_NEGATIVITY.png"
