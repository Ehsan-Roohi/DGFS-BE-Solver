# Phase 5: honest normal-shock validation gate

This stage distinguishes three different claims:

1. **Exact physics audit:** Rankine--Hugoniot mass, momentum, and total-energy
   flux consistency of the imposed helium end states.
2. **Internal numerical audit:** comparison with the `M_omega = 24` angular
   result. This is resolution sensitivity only; it is not an independent
   validation.
3. **Independent validation:** shock-centred comparison against raw,
   same-physics DSMC/DVM data. This gate cannot pass without an external CSV.

The official DGFS normal-shock test-case repository publishes a DGFS/DSMC
comparison image but no machine-readable DSMC profile. We therefore do not
digitize pixels silently or present the internal `M_omega = 24` curve as an
independent solution.

## External-reference schema

Use SI units and these columns:

```text
x_mm,rho_kg_m3,ux_m_s,T_K,qx_W_m2,Pxx_minus_p_Pa
```

Place the file at `$DGFS_ROOT/reference/jcp2019_dsmc.csv`, or set
`DGFS_P5_REFERENCE_CSV=/absolute/path/reference.csv`. A matching provenance file
named `jcp2019_dsmc.provenance.json` is mandatory; it records DSMC/DVM method,
source, independence, and the CSV SHA-256. A reference numerically identical to
the internal DGFS curve is rejected. Without these files, the job still
packages the exact and internal audits but reports
`INCOMPLETE_EXTERNAL_REFERENCE_REQUIRED`.

## Unity

```bash
curl -fsSL https://raw.githubusercontent.com/Ehsan-Roohi/DGFS-BE-Solver/agent/phase5-independent-validation/hpc/bootstrap_unity_p5.sh | bash
```
