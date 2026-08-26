# Steady-state closeout for the conservative-collision audit

The original J14 novelty campaign established the operator-level result: the
weighted five-moment projection reduces the collision-invariant defect to
roundoff at both supported angular orders while adding less than one percent
kernel overhead.  Its four global restarts, however, covered only
`t=160.00` to `160.25`.  The M6 raw case remained steady, but the modified
cases had normalized kinetic residuals near 95--97.  Those short trajectories
are sensitivity checks, not converged global validations.

The closeout continues only `M6_fplus`, `M16_raw`, and `M16_fplus`.  The
already-converged `M6_raw` result is reused.  GPU work is split into bounded
segments ending at `t=168.25`, `176.25`, and `180.25`; later segments skip a
case automatically once its normalized residual is at most one.  The final
claim gate requires all four cases to satisfy that steady-state criterion in
addition to the existing conservation, overhead, paper-accuracy, overshoot,
and finite-value gates.

The `fplus` label means a weighted conservative projection of the collision
term `Q`.  It is not a clipping operation and does not claim positivity of the
distribution `f`.
