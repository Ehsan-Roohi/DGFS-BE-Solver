#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --job-name=j14-val
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -Eeuo pipefail
trap 'rc=$?; rm -f "${CAMPAIGN:-.}/VALIDATION_SUCCESS"; echo "J14_VALIDATION_FAILED rc=$rc line=$LINENO"; exit "$rc"' ERR

ROOT=${DGFS_ROOT:-/project/pi_roohie_umass_edu/DGFS_BE}
ENV_DIR=${DGFS_ENV:-$ROOT/dgfs_py310}
CAMPAIGN=${DGFS_VALIDATION_CAMPAIGN:?DGFS_VALIDATION_CAMPAIGN is required}
SOURCE=${DGFS_JCP14_SOURCE:?DGFS_JCP14_SOURCE is required}
SRC=${DGFS_SOLVER_SRC:?DGFS_SOLVER_SRC is required}
STAGE="$CAMPAIGN/input"
OUT="$CAMPAIGN/results"
cd "$CAMPAIGN"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"

for N in 4 8; do
    test -s "$STAGE/e$N/dgfs.ini"
    test -s "$STAGE/e$N/mesh.frfsm"
    test -s "$STAGE/e$N/bulk-final.frfss"
done

"$ENV_DIR/bin/python" "$SRC/cases/jcp2019_fig14b_validation/compare_fig14.py" \
    --campaign "$STAGE" \
    --reference "$SRC/cases/jcp2019_fig14b_validation/fig14_digitized.csv" \
    --output-dir "$OUT"
test -s "$OUT/fig14_validation.svg"
test -s "$OUT/fig14_metrics.csv"
test -s "$OUT/fig14_metrics.json"
test -s "$OUT/our_profiles.csv"

"$ENV_DIR/bin/python" - "$OUT" "$SOURCE" <<'PY'
import csv
import json
import math
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
source = pathlib.Path(sys.argv[2])
threshold = 2.0e-5

def residual_samples(path):
    samples = []
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            try:
                t = float(row['t'])
                raw = float(row['f'])
                normalized = float(row['f_normalized'])
            except (KeyError, TypeError, ValueError):
                continue
            if all(map(math.isfinite, (t, raw, normalized))):
                samples.append((t, raw, normalized))
    if not samples:
        raise SystemExit(f'NO_RESIDUAL_SAMPLES={path}')
    return samples

def at_or_before(samples, time):
    values = [row for row in samples if row[0] <= time + 1e-9]
    if not values:
        raise SystemExit(f'NO_RESIDUAL_AT_OR_BEFORE={time}')
    return max(values)

e4 = residual_samples(source / 'e4' / 'kinetic_residual.csv')
e8 = residual_samples(source / 'e8' / 'kinetic_residual.csv')
e4_final = at_or_before(e4, 170.0)
e8_snapshot = at_or_before(e8, 140.0)
e8_latest = max(e8)
status = {
    'benchmark': 'Jaiswal_Alexeenko_Hu_JCP378_2019_Figure14',
    'mach': 1.59,
    'kn_global_approx': 0.055,
    'convergence_threshold': threshold,
    'reference_data': {
        'alexeenko': 'digitized DGFS lines from JCP 2019 Figure 14',
        'ohwada': 'digitized 1993 symbols reproduced in JCP 2019 Figure 14',
    },
    'grid4': {
        'status': 'converged_final',
        'snapshot_time': 170.0,
        'residual_sample_time': e4_final[0],
        'paper_normalized_residual': e4_final[2],
        'converged': e4_final[2] < threshold,
        'source_bulk': str(source / 'e4' / 'bulk-final.frfss'),
    },
    'grid8': {
        'status': 'provisional_checkpoint_not_converged',
        'snapshot_time': 140.0,
        'snapshot_residual_sample_time': e8_snapshot[0],
        'snapshot_paper_normalized_residual': e8_snapshot[2],
        'latest_observed_time': e8_latest[0],
        'latest_observed_paper_normalized_residual': e8_latest[2],
        'converged': False,
        'timed_out': True,
        'source_bulk': str(source / 'e8' / 'bulk-140.0.frfss'),
    },
}
(out / 'VALIDATION_STATUS.json').write_text(json.dumps(status, indent=2) + '\n')

summary = out / 'SUMMARY.md'
original = summary.read_text()
notice = (
    '# Validation status\n\n'
    '- Grid 4: converged final solution at t = 170.\n'
    '- Grid 8: provisional complete checkpoint at t = 140; it is shown for inspection '
    'but is not labelled as converged.\n'
    '- References: digitized Alexeenko/Jaiswal DGFS lines and Ohwada (1993) symbols.\n\n'
)
summary.write_text(notice + original)

svg = out / 'fig14_validation.svg'
text = svg.read_text()
text = text.replace(
    'Mach 1.59 helium normal shock — exact JCP Figure 14 validation',
    'Mach 1.59 helium shock — grid 4 converged; grid 8 provisional at t=140')
svg.write_text(text)
print(f"GRID4_FINAL_RESIDUAL={e4_final[2]:.12e}")
print(f"GRID8_T140_RESIDUAL={e8_snapshot[2]:.12e}")
print(f"GRID8_LATEST_RESIDUAL={e8_latest[2]:.12e}")
PY

cp "$SRC/cases/jcp2019_fig14b_validation/fig14_digitized.csv" "$OUT/"
git -C "$SRC" rev-parse HEAD > "$OUT/GIT_COMMIT.txt"
{
    echo "source_campaign=$SOURCE"
    echo "grid4_snapshot=final_t170"
    echo "grid8_snapshot=bulk-140.0.frfss"
    echo "grid8_status=provisional_nonconverged"
} > "$OUT/PROVENANCE.txt"
(cd "$OUT" && find . -type f ! -name FILES.sha256 -print0 | sort -z | xargs -0 sha256sum > FILES.sha256)
touch "$CAMPAIGN/VALIDATION_SUCCESS"

ZIP="$ROOT/j14val_${SLURM_JOB_ID}.zip"
(cd "$OUT" && zip -9 -r "$ZIP" .)
zip -T "$ZIP"
sha256sum "$ZIP" > "$ZIP.sha256.txt"
echo "J14_MIXED_VALIDATION_COMPLETE"
echo "FIGURE=$OUT/fig14_validation.svg"
echo "SUMMARY=$OUT/SUMMARY.md"
echo "METRICS=$OUT/fig14_metrics.csv"
echo "UPLOAD_ZIP=$ZIP"
echo "UPLOAD_SHA=$ZIP.sha256.txt"
ls -lh "$ZIP" "$ZIP.sha256.txt"

