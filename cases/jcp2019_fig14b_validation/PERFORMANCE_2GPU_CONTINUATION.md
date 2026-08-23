# Verified two-GPU continuation for JCP Figure 14

The velocity-batch autotuner selected `NvBatchSize=256` with full-distribution
and bulk-moment equivalence and a measured `1.43986x` speedup over 64.

This second performance gate snapshots the newest complete checkpoint from the
existing exact Mach 1.59 campaign.  It advances the same checkpoint by 0.5 time
units once on one GPU and once on two MPI ranks/two GPUs.  The two-GPU result is
recombined to one partition and compared against the one-GPU result.  Long
continuation is allowed only if distribution and bulk errors pass strict
tolerances and measured speedup is at least 1.10.  Physics, DG order, time step,
velocity grid, angular rule, collision model, boundaries, and paper convergence
criterion are unchanged.

The source campaign is read-only.  It is never cancelled, modified, or deleted.
