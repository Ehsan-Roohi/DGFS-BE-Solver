# Exact JCP 2019 Figure 14 validation

This campaign reproduces only the Mach 1.59 helium normal shock in Figure 14
of Jaiswal, Alexeenko, and Hu, *Journal of Computational Physics* 378 (2019),
178-208, DOI `10.1016/j.jcp.2018.11.001`.

It deliberately contains **no Mach 2 case and no BGK/DVM comparison**.

## Paper configuration

| parameter | value |
|---|---:|
| gas/model | helium, VHS with `omega=0.5` (hard sphere) |
| upstream Mach | 1.59 |
| physical domain | `[-15,15] mm` |
| physical grids | 4 and 8 uniform line elements |
| DG order | 3 (polynomial degree 2) |
| limiter | none |
| time scheme | SSP-RK2, `dt=0.001` |
| velocity domain | `[-7,7]^3` |
| velocity grid | `32^3` |
| radial/angular rules | `Nrho=32`, Womersley `M=6` |
| molecular parameters | `dRef=2.17e-10 m`, `Tref=273 K` |
| convergence gate | paper-normalized distribution residual `<2e-5` |

The two physical grids run concurrently as one Slurm array.  Each task stays
inside one GPU allocation and advances in ten-time-unit restart segments until
the paper criterion is met.  This avoids the earlier continuation-submission
failures caused by Unity's per-user job-submission limit.  A dependent CPU job
then performs the comparison and packages the results.

## Reference data

`fig14_digitized.csv` contains:

- native vector paths of the authors' DGFS curves in panels 14(a) and 14(b);
- native vector marker centres for the Ohwada [59] finite-difference solution.

The extraction is reproducible with `digitize_fig14.py`; it reads vector paths
from page 14 of arXiv `1809.10186v2`.  It does not trace raster pixels.  The
source hash, axis calibration, citation, and uncertainty estimate are recorded
in `fig14_digitization_provenance.json`.  These data are used only after the
solver finishes and never enter the initial or boundary conditions.

The comparison output contains the present raw DG polynomials, exact GLL cell
averages, RMS/L-infinity errors against the paper curves and Ohwada symbols,
overshoot metrics, and a six-panel SVG figure.

## Unity one-line run

```bash
curl -fsSL https://raw.githubusercontent.com/Ehsan-Roohi/DGFS-BE-Solver/agent/jcp2019-fig14b-ohwada-validation/hpc/bootstrap_unity_jcp14.sh | bash
```

The campaign directory is short: `$DGFS_ROOT/jcp14_YYYYMMDD_HHMMSS`.  The final
log prints `UPLOAD_ZIP` and `UPLOAD_SHA`.
