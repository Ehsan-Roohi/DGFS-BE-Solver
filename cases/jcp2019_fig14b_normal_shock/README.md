# JCP 2019 Figure 14(b): helium normal shock

This case reproduces Figure 14(b) of Jaiswal, Alexeenko, and Hu,
*Journal of Computational Physics* **378** (2019), 178--208,
[doi:10.1016/j.jcp.2018.11.001](https://doi.org/10.1016/j.jcp.2018.11.001).

The values below are taken from Section 4.7, Table 2, and the Figure 14
caption of the paper.  Table 2 lists the four-element baseline; panel (b)
uses the stated eight-element refinement.  They deliberately differ from the
earlier `normal_shock_M1p592` exploratory run, whose physical domain was four
times longer.

| Quantity | Reproduction value |
|---|---:|
| Gas | helium |
| Upstream Mach number | 1.59 |
| Physical domain | `[-15, 15] mm` |
| Spatial elements | 8 uniform one-dimensional elements |
| DG approximation | third order (`order = 2`, polynomial degree 2) |
| Limiter | none |
| Velocity domain | `[-7, 7]^3` |
| Velocity grid | `32^3` |
| Radial quadrature | `Nr = N = 32` |
| Spherical design | Womersley, `M = 6` |
| VHS parameters | `omega = 0.5`, `dRef = 2.17e-10 m`, `Tref = 273 K` |
| Upstream state | `rho=1.916e-5 kg/m^3`, `ux=1398.771 m/s`, `T=223 K` |
| Downstream state | `rho=3.505e-5 kg/m^3`, `ux=764.659 m/s`, `T=354.762 K` |
| Upstream mean free path | `1.648 mm` |

Section 4.7 explicitly identifies the benchmark as a one-dimensional steady
case.  The mesh extent is `x/H0 in [-0.5, 0.5]`, with `H0 = 30 mm`; therefore
its physical length is exactly 30 mm and `Delta x/lambda = 2.2755`, as in
Figure 14(b).

## Unity one-line submission

Run this on a Unity login node:

```bash
curl -fsSL https://raw.githubusercontent.com/Ehsan-Roohi/DGFS-BE-Solver/agent/jcp2019-fig14b-exact/hpc/bootstrap_unity_jcp2019_fig14b.sh | bash
```

The bootstrap creates a timestamped, self-contained run directory under
`/project/pi_roohie_umass_edu/DGFS_BE/runs`, checks every paper parameter,
and submits one GPU job.  The job writes the final distribution and moments,
exports a VTU file, records checksums, and produces a ZIP archive.  If
Matplotlib is available, it also creates `fig14b_raw_and_cell_average.png`.
The residual plugin is sampled every Euler step and reports the paper's
normalized convergence measure, using the second-step relative change as its
denominator.  The job is accepted only when that measure is below `2e-5`.

To validate this case without launching the solver:

```bash
python cases/jcp2019_fig14b_normal_shock/verify_case.py
```
