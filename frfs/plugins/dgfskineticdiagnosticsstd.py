# -*- coding: utf-8 -*-

"""Kinetic diagnostics and troubled-cell sensing for single-species DGFS.

This plugin is deliberately read-only: it copies the current distribution to
the host, computes diagnostics, and writes CSV files.  It does not alter the
time integrator, the collision operator, or any solution register.
"""

import os

import numpy as np

from frfs.mpiutil import get_comm_rank_root, get_mpi
from frfs.plugins.base import BasePlugin, init_csv
from frfs.quadrules import get_quadrule


def modal_tail_sensor(field, vdm, high_mode_mask, tiny=1.0e-300):
    """Return log10 of the highest-mode energy fraction in each element."""
    modal = np.linalg.solve(vdm.T, field)
    total = np.sum(modal*modal, axis=0)
    high = np.sum(modal[high_mode_mask]*modal[high_mode_mask], axis=0)

    return np.log10(np.maximum(high/np.maximum(total, tiny), tiny))


def highest_mode_mask(shape, degrees, degree):
    """Identify the P_p/P_{p-1} modal shell for each element family."""
    if shape in {'line', 'quad', 'hex'}:
        shell = [max(d) == degree for d in degrees]
    elif shape in {'tri', 'tet'}:
        shell = [sum(d) == degree for d in degrees]
    elif shape == 'pri':
        shell = [sum(d[:2]) == degree or d[2] == degree for d in degrees]
    elif shape == 'pyr':
        shell = [max(d[:2]) + d[2] == degree for d in degrees]
    else:
        shell = [max(d) == degree for d in degrees]

    return np.array(shell, dtype=bool)


class DGFSKineticDiagnosticsStdPlugin(BasePlugin):
    """Monitor positivity, moments, entropy, and modal smoothness."""

    name = 'dgfskineticdiagnosticsstd'
    systems = ['dgfs', 'adgfs']
    formulations = ['std']

    _summary_header = [
        't', 'step', 'f_min', 'negative_dof_fraction',
        'negative_l1_fraction', 'entropy_positive_part', 'mass',
        'momentum_x', 'momentum_y', 'momentum_z', 'energy',
        'drift_mass_ref', 'drift_momentum_x_refscale',
        'drift_momentum_y_refscale', 'drift_momentum_z_refscale',
        'drift_energy_ref', 'troubled_cells', 'total_cells',
        'troubled_fraction', 'max_distribution_modal_sensor',
        'max_density_modal_sensor', 'max_temperature_modal_sensor',
        'max_modal_sensor'
    ]

    _cell_header = [
        't', 'step', 'rank', 'element_type', 'element',
        'center_x', 'center_y', 'center_z', 'f_min',
        'negative_dof_fraction', 'negative_l1_fraction', 'rho_min',
        'temperature_min', 'density_modal_sensor',
        'temperature_modal_sensor', 'distribution_modal_sensor', 'troubled'
    ]

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        comm, rank, root = get_comm_rank_root()

        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')
        self.modal_threshold = self.cfg.getfloat(
            cfgsect, 'modal-threshold', -3.0
        )
        self.negative_absolute_tolerance = self.cfg.getfloat(
            cfgsect, 'negative-absolute-tolerance', 0.0
        )
        self.negative_relative_tolerance = self.cfg.getfloat(
            cfgsect, 'negative-relative-tolerance', 1.0e-10
        )
        self.negative_l1_tolerance = self.cfg.getfloat(
            cfgsect, 'negative-l1-tolerance', 1.0e-8
        )
        self.rho_floor = self.cfg.getfloat(cfgsect, 'rho-floor', 1.0e-12)
        self.temperature_floor = self.cfg.getfloat(
            cfgsect, 'temperature-floor', 1.0e-12
        )
        self.entropy_floor = self.cfg.getfloat(
            cfgsect, 'entropy-floor', 1.0e-300
        )
        self.velocity_chunk = self.cfg.getint(
            cfgsect, 'velocity-chunk', 256
        )
        self.distribution_sensor = self.cfg.getbool(
            cfgsect, 'distribution-modal-sensor', True
        )

        if self.nsteps <= 0:
            raise ValueError('nsteps must be positive')
        if self.velocity_chunk <= 0:
            raise ValueError('velocity-chunk must be positive')
        if min(self.negative_absolute_tolerance,
               self.negative_relative_tolerance,
               self.negative_l1_tolerance) < 0.0:
            raise ValueError('negative tolerances must be non-negative')
        if min(self.rho_floor, self.temperature_floor,
               self.entropy_floor) <= 0.0:
            raise ValueError('diagnostic floors must be positive')

        self.outf = None
        if rank == root:
            self.outf = init_csv(
                self.cfg, cfgsect, ','.join(self._summary_header)
            )

        self.cell_outf = None
        cell_file = self.cfg.get(cfgsect, 'cell-file', '').strip()
        if cell_file:
            if not cell_file.endswith('.csv'):
                cell_file += '.csv'

            base, ext = os.path.splitext(cell_file)
            rank_file = '{0}_rank{1:05d}{2}'.format(base, rank, ext)
            self.cell_outf = open(rank_file, 'a')

            if os.path.getsize(rank_file) == 0:
                print(','.join(self._cell_header), file=self.cell_outf)

        vm = intg.system.vm
        self.cv = vm.cv()
        self.cv2 = np.sum(self.cv*self.cv, axis=0)
        self.cw = vm.cw()
        self.vsize = vm.vsize()

        self.element_data = []
        for etype, ele in intg.system.ele_map.items():
            basis = ele.basis

            # Integrate the nodal polynomial using a rule that is exact for
            # at least twice the DG degree.  For affine elements this makes
            # the conserved-moment totals exact to the represented solution.
            qrule = get_quadrule(
                basis.name, qdeg=max(2*basis.order + 1, 1)
            )
            interp = basis.ubasis.nodal_basis_at(qrule.pts)
            ref_weights = np.dot(qrule.wts, interp)
            phys_weights = ref_weights[:, None]*ele.djac_at_np('upts')

            high_mode_mask = highest_mode_mask(
                basis.name, basis.ubasis.degrees, basis.order
            )
            if not np.any(high_mode_mask):
                high_mode_mask[-1] = True

            centers = intg.system.ele_ploc_upts[
                len(self.element_data)
            ].mean(axis=0).T

            self.element_data.append({
                'etype': etype,
                'vdm': basis.ubasis.vdm,
                'high_mode_mask': high_mode_mask,
                'weights': phys_weights,
                'abs_weights': np.abs(phys_weights),
                'centers': centers
            })

        self.reference = None
        self.last_sample_step = None

        # Record the initial or restart state before the first advance.
        self._sample(intg)

    def __call__(self, intg):
        if intg.nacptsteps % self.nsteps:
            return
        if intg.nacptsteps == self.last_sample_step:
            return

        self._sample(intg)

    def _analyse_element_type(self, f, edata):
        nupts, vsize, neles = f.shape
        if vsize != self.vsize:
            raise ValueError('Unexpected velocity-space size')

        weights = edata['weights']
        abs_weights = edata['abs_weights']

        cell_f_min = np.min(f, axis=(0, 1))
        cell_f_max = np.max(f, axis=(0, 1))
        cell_negative_count = np.zeros(neles, dtype=np.float64)
        cell_negative_l1 = np.zeros(neles)
        cell_absolute_l1 = np.zeros(neles)
        distribution_total_energy = np.zeros(neles)
        distribution_high_energy = np.zeros(neles)

        negative_count = 0.0
        negative_l1 = 0.0
        absolute_l1 = 0.0
        entropy = 0.0

        for start in range(0, vsize, self.velocity_chunk):
            stop = min(start + self.velocity_chunk, vsize)
            fb = f[:, start:stop, :]
            negative = fb < 0.0

            cell_negative_count += np.sum(negative, axis=(0, 1))
            cell_negative_l1 += np.sum(
                np.where(negative, -fb, 0.0), axis=(0, 1)
            )
            cell_absolute_l1 += np.sum(np.abs(fb), axis=(0, 1))

            negative_count += float(np.sum(negative))
            negative_l1 += self.cw*np.sum(
                abs_weights[:, None, :]*np.where(negative, -fb, 0.0)
            )
            absolute_l1 += self.cw*np.sum(
                abs_weights[:, None, :]*np.abs(fb)
            )

            positive_part = np.where(fb > 0.0, fb, 0.0)
            entropy += self.cw*np.sum(
                weights[:, None, :]*positive_part*
                np.log(np.maximum(positive_part, self.entropy_floor))
            )

            if self.distribution_sensor:
                modal_f = np.linalg.solve(
                    edata['vdm'].T, fb.reshape(nupts, -1)
                ).reshape(nupts, stop - start, neles)
                distribution_total_energy += np.sum(
                    modal_f*modal_f, axis=(0, 1)
                )
                high_modal_f = modal_f[edata['high_mode_mask']]
                distribution_high_energy += np.sum(
                    high_modal_f*high_modal_f, axis=(0, 1)
                )

        rho = self.cw*np.sum(f, axis=1)
        momentum = np.array([
            self.cw*np.einsum('uve,v->ue', f, self.cv[d])
            for d in range(3)
        ])
        energy_density = 0.5*self.cw*np.einsum(
            'uve,v->ue', f, self.cv2
        )

        velocity_squared = np.zeros_like(rho)
        valid_rho = rho > self.rho_floor
        for d in range(3):
            velocity_component = np.zeros_like(rho)
            velocity_component[valid_rho] = (
                momentum[d][valid_rho]/rho[valid_rho]
            )
            velocity_squared += velocity_component*velocity_component

        temperature = np.full_like(rho, -np.inf)
        temperature[valid_rho] = (2.0/3.0)*(
            2.0*energy_density[valid_rho]/rho[valid_rho]
            - velocity_squared[valid_rho]
        )

        density_sensor = modal_tail_sensor(
            rho, edata['vdm'], edata['high_mode_mask']
        )
        temperature_sensor = modal_tail_sensor(
            np.where(np.isfinite(temperature),
                     temperature, self.temperature_floor),
            edata['vdm'], edata['high_mode_mask']
        )
        if self.distribution_sensor:
            distribution_sensor = np.log10(np.maximum(
                distribution_high_energy/np.maximum(
                    distribution_total_energy, self.entropy_floor
                ),
                self.entropy_floor
            ))
        else:
            distribution_sensor = np.full(neles, -300.0)

        cell_negative_fraction = (
            cell_negative_count/float(nupts*vsize)
        )
        cell_negative_l1_fraction = (
            cell_negative_l1/np.maximum(cell_absolute_l1,
                                        self.entropy_floor)
        )
        rho_min = np.min(rho, axis=0)
        temperature_min = np.min(temperature, axis=0)

        positivity_limit = -(
            self.negative_absolute_tolerance
            + self.negative_relative_tolerance*np.maximum(cell_f_max, 0.0)
        )
        troubled = (
            (cell_f_min < positivity_limit)
            | (cell_negative_l1_fraction > self.negative_l1_tolerance)
            | (rho_min <= self.rho_floor)
            | (temperature_min <= self.temperature_floor)
            | (density_sensor > self.modal_threshold)
            | (temperature_sensor > self.modal_threshold)
            | (distribution_sensor > self.modal_threshold)
        )

        totals = np.array([
            negative_count,
            float(f.size),
            negative_l1,
            absolute_l1,
            entropy,
            np.sum(weights*rho),
            np.sum(weights*momentum[0]),
            np.sum(weights*momentum[1]),
            np.sum(weights*momentum[2]),
            np.sum(weights*energy_density),
            float(np.count_nonzero(troubled)),
            float(neles)
        ])

        cells = {
            'f_min': cell_f_min,
            'negative_fraction': cell_negative_fraction,
            'negative_l1_fraction': cell_negative_l1_fraction,
            'rho_min': rho_min,
            'temperature_min': temperature_min,
            'density_sensor': density_sensor,
            'temperature_sensor': temperature_sensor,
            'distribution_sensor': distribution_sensor,
            'troubled': troubled
        }

        return totals, cells

    def _write_cells(self, intg, rank, edata, cells):
        if self.cell_outf is None:
            return

        centers = edata['centers']
        for eidx in range(len(cells['f_min'])):
            center = [np.nan, np.nan, np.nan]
            center[:self.ndims] = centers[eidx, :self.ndims]

            row = [
                intg.tcurr, intg.nacptsteps, rank, edata['etype'], eidx,
                center[0], center[1], center[2], cells['f_min'][eidx],
                cells['negative_fraction'][eidx],
                cells['negative_l1_fraction'][eidx],
                cells['rho_min'][eidx], cells['temperature_min'][eidx],
                cells['density_sensor'][eidx],
                cells['temperature_sensor'][eidx],
                cells['distribution_sensor'][eidx],
                int(cells['troubled'][eidx])
            ]
            print(','.join(str(value) for value in row), file=self.cell_outf)

        self.cell_outf.flush()

    def _sample(self, intg):
        comm, rank, root = get_comm_rank_root()

        full_solution = [
            bank[intg._idxcurr].get()
            for bank in intg.system.eles_scal_upts_inb_full
        ]

        local_totals = np.zeros(12)
        local_f_min = np.inf
        local_max_distribution_sensor = -300.0
        local_max_density_sensor = -300.0
        local_max_temperature_sensor = -300.0

        for f, edata in zip(full_solution, self.element_data):
            totals, cells = self._analyse_element_type(f, edata)
            local_totals += totals
            local_f_min = min(local_f_min, np.min(cells['f_min']))
            local_max_distribution_sensor = max(
                local_max_distribution_sensor,
                np.max(cells['distribution_sensor'])
            )
            local_max_density_sensor = max(
                local_max_density_sensor, np.max(cells['density_sensor'])
            )
            local_max_temperature_sensor = max(
                local_max_temperature_sensor,
                np.max(cells['temperature_sensor'])
            )
            self._write_cells(intg, rank, edata, cells)

        global_totals = np.empty_like(local_totals) if rank == root else None
        comm.Reduce(
            local_totals, global_totals, op=get_mpi('sum'), root=root
        )

        local_extrema = np.array([
            local_f_min,
            local_max_distribution_sensor,
            local_max_density_sensor,
            local_max_temperature_sensor
        ])
        global_f_min = np.empty(1) if rank == root else None
        global_max_sensors = np.empty(3) if rank == root else None
        comm.Reduce(
            local_extrema[:1], global_f_min,
            op=get_mpi('min'), root=root
        )
        comm.Reduce(
            local_extrema[1:], global_max_sensors,
            op=get_mpi('max'), root=root
        )

        if rank == root:
            g = global_totals
            invariants = g[5:10].copy()

            if self.reference is None:
                self.reference = invariants.copy()

            mass_scale = max(abs(self.reference[0]), self.entropy_floor)
            momentum_scale = max(
                np.sqrt(2.0*abs(self.reference[0]*self.reference[4])),
                self.entropy_floor
            )
            energy_scale = max(abs(self.reference[4]), self.entropy_floor)
            delta = invariants - self.reference
            max_modal_sensor = np.max(global_max_sensors)

            row = [
                intg.tcurr,
                intg.nacptsteps,
                global_f_min[0],
                g[0]/max(g[1], 1.0),
                g[2]/max(g[3], self.entropy_floor),
                g[4],
                invariants[0],
                invariants[1],
                invariants[2],
                invariants[3],
                invariants[4],
                delta[0]/mass_scale,
                delta[1]/momentum_scale,
                delta[2]/momentum_scale,
                delta[3]/momentum_scale,
                delta[4]/energy_scale,
                int(g[10]),
                int(g[11]),
                g[10]/max(g[11], 1.0),
                global_max_sensors[0],
                global_max_sensors[1],
                global_max_sensors[2],
                max_modal_sensor
            ]

            print('  {0}: t={1:.6e}, fmin={2:.3e}, negL1={3:.3e}, '
                  'troubled={4}/{5}, sensor={6:.3e}'.format(
                      self.name, intg.tcurr, global_f_min[0], row[4],
                      int(g[10]), int(g[11]), max_modal_sensor
                  ))
            print(','.join(str(value) for value in row), file=self.outf)
            self.outf.flush()

        self.last_sample_step = intg.nacptsteps
