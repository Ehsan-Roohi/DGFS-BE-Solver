#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "M6_EUCLIDEAN_CONTINUE_WRAPPER_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
CLOSE=${DGFS_CLOSEOUT:-$ROOT/j14novclose_20260826_211439}
BRANCH=${DGFS_BRANCH:-agent/j14nov-m16-storage-safe-v100}
MAX_SEG=${DGFS_EUCLIDEAN_MAX_SEGMENTS:-5}

# Find the latest retained final distribution from a previous Euclidean steady job.
SOURCE=$({ find "$CLOSE" -maxdepth 4 -type f \
  -path '*/m6_euclidean_steady_*/segment_*/dist_p3b_M6_euclidean_steady-*.frfss' \
  -printf '%T@ %p\n' 2>/dev/null || true; } | sort -nr | head -1 | cut -d' ' -f2-)
[[ -s "$SOURCE" ]] || { echo "M6_EUCLIDEAN_PREVIOUS_FINAL_NOT_FOUND"; exit 2; }
BASECFG=$(dirname "$SOURCE")/p3b_M6_euclidean_steady.ini
[[ -s "$BASECFG" ]] || { echo "M6_EUCLIDEAN_PREVIOUS_CONFIG_NOT_FOUND=$BASECFG"; exit 3; }

echo "M6_EUCLIDEAN_CONTINUE_SOURCE=$SOURCE"
echo "M6_EUCLIDEAN_CONTINUE_BASECFG=$BASECFG"
echo "M6_EUCLIDEAN_CONTINUE_MAX_SEGMENTS=$MAX_SEG"

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
curl -fsSL \
  "https://raw.githubusercontent.com/Ehsan-Roohi/DGFS-BE-Solver/$BRANCH/hpc/bootstrap_unity_j14_m6_euclidean_steady.sh" \
  -o "$TMP"

# Replace only the source-discovery block. Everything else, including V100-only
# scheduling, early stationarity stop, diagnostics, and output packaging, stays
# identical to the validated bootstrap.
python3 - "$TMP" "$SOURCE" "$BASECFG" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); source=sys.argv[2]; basecfg=sys.argv[3]
s=p.read_text()
a=s.index('SHORTROOT=$({ find "$CLOSE"')
b=s.index('read -r T0 < <(', a)
replacement=(
    f'SOURCE={source!r}\n'
    f'BASECFG={basecfg!r}\n'
    '[[ -s "$SOURCE" ]] || { echo "M6_EUCLIDEAN_SOURCE_NOT_FOUND"; exit 3; }\n'
    '[[ -s "$BASECFG" ]] || { echo "M6_EUCLIDEAN_BASECFG_NOT_FOUND"; exit 4; }\n\n'
)
p.write_text(s[:a]+replacement+s[b:])
PY

DGFS_EUCLIDEAN_MAX_SEGMENTS="$MAX_SEG" bash "$TMP"
