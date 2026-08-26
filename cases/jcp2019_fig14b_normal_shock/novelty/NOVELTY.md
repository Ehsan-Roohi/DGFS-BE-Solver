# Innovation stage: conservative fast-spectral DGFS collision operator

## Research question

For the exact Jaiswal–Alexeenko–Hu JCP 2019 Figure 14 helium shock
(Mach 1.59, global Kn approximately 0.055, 8 spatial elements, third-order DG),
can a weighted five-moment correction make the discrete fast-spectral collision
term conservative without degrading the validated shock profiles?

## Method

For each spatial DG solution point, form the collision-invariant basis

B = [1, vx, vy, vz, 0.5 |v|^2]

and the raw discrete moments m = cw B Q. The proposed correction is

Qc = Q - w B^T lambda,    w = max(f, 0),

where lambda solves the weighted 5 by 5 Gram system so that cw B Qc = 0.
The operation modifies the collision term Q only. It does not clip f and it
must not be described as a proof of global positivity.

The converged 8-element Figure 14 distribution is used as the common initial
state. Four otherwise identical 0.25-time-unit restarts are performed:

1. angular order M_omega = 6, raw collision;
2. angular order M_omega = 6, conservative fplus projection;
3. angular order M_omega = 12, raw collision;
4. angular order M_omega = 12, conservative fplus projection.

## Evidence collected

- Direct raw and corrected collision-invariant defects at all 24 DG points.
- Collision-kernel timing and projection overhead.
- Full physical density, velocity, and temperature profiles.
- RMS and maximum errors against digitized Alexeenko DGFS lines and Ohwada
  symbols reproduced in JCP 2019 Figure 14.
- Shock location, macroscopic overshoot, negative distribution mass, integral
  drift, residual history, and solver wall time.

## Predeclared claim gate

A publication claim passes only if both angular orders satisfy:

- corrected maximum collision-invariant defect <= 5e-12;
- collision correction overhead <= 50%;
- mean paper-reference RMS is not degraded by more than 10% plus 1e-6;
- maximum density/velocity/temperature overshoot is not worsened by more than
  0.005 of the normalized jump;
- all reported metrics are finite.

The batch job always packages the scientific result. It writes
CLAIM_GATE_PASS or CLAIM_GATE_FAIL instead of deleting a negative result.

## Defensible novelty statement

If the gate passes, the supported statement is:

> A positive-node-weighted, five-moment projection restores discrete
> conservation of the fast-spectral Boltzmann collision term to numerical
> precision in the validated Mach 1.59 normal-shock benchmark, across two
> angular orders, with bounded overhead and no material loss of agreement with
> the Alexeenko and Ohwada reference profiles.

No stronger positivity, entropy, geometry, or all-flow-regime claim follows
from this experiment.

