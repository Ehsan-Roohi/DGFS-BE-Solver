#!/usr/bin/env bash
# Salvage and package a completed P4A run without rerunning the solver.
set -Eeuo pipefail
trap 'rc=$?; echo "P4A_PACKAGE_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

JOB=${1:-${JOB_ID:-}}
[[ -n "$JOB" ]] || { echo "usage: $0 JOB_ID"; exit 2; }
DGFS_ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
RUN_DIR=$(find "$DGFS_ROOT" -maxdepth 2 -type f -name "slurm-$JOB.out" -printf '%h\n' -quit)
[[ -d "$RUN_DIR" ]] || { echo "P4A_RUN_DIR_NOT_FOUND job=$JOB"; exit 3; }
cd "$RUN_DIR"

test -s p4a_comparison.json
test -s p4a_comparison.csv
test -s p4a_comparison.md
grep -q '^P4A_ASSESSMENT_PASS$' p4a_assess.log
for name in M16_raw M16_fplus M24_raw; do
    test -s "run_$name/dist_p3b_$name-1.00.frfss"
done

PLOT_PY=""
CANDIDATES=()
[[ -n "${DGFS_PLOT_PYTHON:-}" ]] && CANDIDATES+=("$DGFS_PLOT_PYTHON")
CANDIDATES+=("$DGFS_ROOT/dgfs_py310/bin/python")
if command -v python3 >/dev/null 2>&1; then CANDIDATES+=("$(command -v python3)"); fi
if command -v python >/dev/null 2>&1; then CANDIDATES+=("$(command -v python)"); fi
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base 2>/dev/null || true)
    [[ -n "$CONDA_BASE" ]] && CANDIDATES+=("$CONDA_BASE/bin/python")
fi
for py in "${CANDIDATES[@]}"; do
    if [[ -x "$py" ]] && "$py" -c 'import matplotlib, numpy' >/dev/null 2>&1; then
        PLOT_PY=$py
        break
    fi
done

if [[ -n "$PLOT_PY" ]]; then
    MPLCONFIGDIR="$RUN_DIR/.matplotlib" "$PLOT_PY" p3/p3c_physical_profiles.py \
      --comparison p4a_comparison.json --mach 1.59 \
      --profiles-png p4a_physical_profiles_24points.png \
      --profiles-pdf p4a_physical_profiles_24points.pdf \
      --differences-png p4a_physical_differences.png \
      --differences-pdf p4a_physical_differences.pdf \
      --csv p4a_physical_profiles_24points.csv
    echo "P4A_FIGURES_COMPLETE python=$PLOT_PY"
else
    echo "P4A_FIGURES_SKIPPED_NO_MATPLOTLIB"
fi

touch P4A_SUCCESS
ZIP="$DGFS_ROOT/p4a_${JOB}.zip"
STAGE=$(mktemp -d "$DGFS_ROOT/.p4a_${JOB}.XXXXXX")
TMP="$STAGE/p4a.zip"
trap 'rm -rf -- "$STAGE"' EXIT
zip -q -9 -r "$TMP" p4a_* P4A_SUCCESS gpu_layout_preflight.log configs run_* p3 solver_hook INPUT.txt \
  dgfs_fig14b.ini mesh.frfsm dist_dgfs_fig14b-0.0.frfss "slurm-$JOB.out" \
  -x '*/__pycache__/*' '*.pyc' 'run_*/dist_dgfs_fig14b-0.0.frfss'
zip -T "$TMP"
mv -f "$TMP" "$ZIP"
rm -rf -- "$STAGE"
trap - EXIT
sha256sum "$ZIP" > "$ZIP.sha256.txt"
echo P4A_VERIFIED_COMPLETE
echo "FIGURES=$([[ -n "$PLOT_PY" ]] && echo yes || echo no)"
echo "UPLOAD_ZIP=$ZIP"
echo "UPLOAD_SHA=$ZIP.sha256.txt"
ls -lh "$ZIP" "$ZIP.sha256.txt"
