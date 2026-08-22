# DGFS kinetic diagnostics: first test step

This patch adds a read-only plugin for the single-species `dgfs`/`adgfs`
systems.  It does not change the numerical solution.  At a configurable step
interval it records:

- minimum distribution value and negative-DOF fraction;
- physically weighted negative-L1 fraction;
- positive-part discrete kinetic entropy;
- global mass, three momentum components, and kinetic energy;
- changes in these totals relative to the first sample;
- distribution, density, and temperature modal-tail sensors for each cell;
- a combined troubled-cell flag.

For an open normal-shock domain, changes in global invariants include physical
boundary fluxes; they are not a collision-conservation defect.  The entropy is
computed from the positive part of `f`, because the standard `f log(f)` is not
defined for negative `f`.  Negativity is reported separately.

Add this section to the normal-shock INI:

```ini
[soln-plugin-dgfskineticdiagnosticsstd]
nsteps = 50
file = kinetic_diagnostics.csv
cell-file = kinetic_cells.csv

; A cell is troubled when any modal sensor exceeds this value.
modal-threshold = -3.0

; Positivity criteria.  The relative test is scaled by max(f) in each cell.
negative-absolute-tolerance = 0.0
negative-relative-tolerance = 1e-10
negative-l1-tolerance = 1e-8

rho-floor = 1e-12
temperature-floor = 1e-12
entropy-floor = 1e-300
velocity-chunk = 256
distribution-modal-sensor = true
```

For a short smoke test, restart the Mach 1.592 case at `t = 6`, set
`tend = 6.02`, retain `dt = 0.001`, and run the standard restart command.  This
produces the initial/restart sample plus samples at steps 50 only if the run is
long enough; for the 20-step smoke test set `nsteps = 5`.

Validate the summary CSV with:

```bash
python3 tools/validate_kinetic_diagnostics.py kinetic_diagnostics.csv \
  --minimum-samples 2 --require-troubled-cells
```

The per-rank cell files are named `kinetic_cells_rank00000.csv`, etc.  Sort by
the maximum of the three modal sensors and inspect the flagged cells around
the shock.  The distribution sensor is the kinetic part of the detector; set
`distribution-modal-sensor = false` only for a cheaper preliminary run.  The
default threshold is intentionally conservative; use the first run to
calibrate it before implementing a limiter.

On Unity, the complete restart smoke test can be installed and submitted with
the one-line bootstrap command documented in the GitHub pull request.  The
bootstrap uses a dedicated checkout and run directory, refuses to overwrite a
dirty checkout, installs the checkout in editable mode, and restarts the
validated Mach 1.592 solution from `t = 6` to `t = 6.02`.
