# -*- coding: utf-8 -*-

import numpy as np

from frfs.mpiutil import get_comm_rank_root, get_mpi
from frfs.plugins.base import BasePlugin, init_csv


class DGFSResidualStdPlugin(BasePlugin):
    name = 'dgfsresidualstd'
    systems = ['dgfs', 'adgfs']
    formulations = ['std']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        comm, rank, root = get_comm_rank_root()

        # Output frequency
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')
        self.isoutf = self.cfg.getint(cfgsect, 'output-file', 0)
        self.normalise = self.cfg.getbool(cfgsect, 'normalise', False)
        self._normalisation_resid = None
        if self.cfg.hasopt(cfgsect, 'normalisation-resid'):
            reference = self.cfg.getfloat(cfgsect, 'normalisation-resid')
            if not np.isfinite(reference) or reference <= 0:
                raise ValueError('normalisation-resid must be positive')
            self._normalisation_resid = np.array([reference])

        # The root rank needs to open the output file
        if rank == root and self.isoutf:
            header = ['t', 'f']
            if self.normalise:
                header.append('f_normalized')

            # Open
            self.outf = init_csv(self.cfg, cfgsect, ','.join(header))

        # Call ourself in case output is needed after the first step
        self(intg)

    def __call__(self, intg):
        # If an output is due this step
        if intg.nacptsteps % self.nsteps == 0 and intg.nacptsteps:
            # MPI info
            comm, rank, root = get_comm_rank_root()

            # Previous and current solution
            prev = self._prev
            curr = [s[intg._idxcurr].get() for s in 
                    intg.system.eles_scal_upts_inb_full]

            # Square of the residual vector [pad 0 for communication]
            resid_num = np.array([sum(np.linalg.norm(c - p)**2
                        for p, c in zip(prev, curr)), 0.])
            resid_den = np.array([sum(np.linalg.norm(p)**2
                        for p in prev), 0.])

            # Reduce and, if we are the root rank, output
            if rank != root:
                comm.Reduce(resid_num, None, op=get_mpi('sum'), root=root)
                comm.Reduce(resid_den, None, op=get_mpi('sum'), root=root)
            else:
                comm.Reduce(get_mpi('in_place'), resid_num, op=get_mpi('sum'),
                            root=root)
                comm.Reduce(get_mpi('in_place'), resid_den, op=get_mpi('sum'),
                            root=root)

                # Normalise [Remove the padded 0]
                resid = np.sqrt(resid_num[:-1]/resid_den[:-1])

                # Optionally reproduce the normal-shock convergence measure
                # used by Jaiswal, Alexeenko, and Hu (JCP 378, 2019):
                #   r_n / r_1,  r_n = ||f^{n+1} - f^n|| / ||f^n||.
                # With one-based paper indexing, the reference is the change
                # between accepted steps one and two.
                if self.normalise:
                    if (intg.nacptsteps == 2 and
                            self._normalisation_resid is None):
                        self._normalisation_resid = resid.copy()

                    if self._normalisation_resid is None:
                        normresid = np.full_like(resid, np.nan)
                    else:
                        normresid = resid/self._normalisation_resid

                    row = [intg.tcurr] + resid.tolist() + normresid.tolist()
                else:
                    row = [intg.tcurr] + resid.tolist()

                # Write
                print(' ', self.name, ': ', 
                    ', '.join("{0:.3e}".format(r) for r in row))

                # Flush to disk
                if(self.isoutf):
                    print(','.join(str(r) for r in row), file=self.outf)
                    self.outf.flush()

            del self._prev

        # If an output is due next step
        if (intg.nacptsteps + 1) % self.nsteps == 0:
            self._prev = [s[intg._idxcurr].get() for s in 
                            intg.system.eles_scal_upts_inb_full]
