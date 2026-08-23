# JCP Figure 14 velocity-batch autotuner

This is a non-destructive performance gate for the exact Mach 1.59 benchmark.
It runs the same four-element problem to `t=0.1` with velocity batch sizes 64,
256, and 1024.  The original double-precision `Nv=32`, `Nrho=32`, `M=6`, DG
order, time step, collision model, initial conditions, and boundary conditions
are unchanged.

The gate compares the full distribution and all written bulk moments against
the batch-64 baseline with a relative tolerance of `2e-12`.  Failed or
non-equivalent variants cannot be recommended.  The fastest passing batch is
written to `RECOMMENDATION.env`.  This job does not cancel, modify, or restart
the production Figure 14 campaign.

The next gate will use the selected batch in a checkpoint-preserving two-GPU
MPI continuation and verify the repartitioned solution before production use.
