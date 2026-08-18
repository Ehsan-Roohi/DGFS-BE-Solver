from __future__ import annotations

import configparser
import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "cases"
    / "jcp2019_fig14b_normal_shock"
    / "diagnose_fig14b.py"
)
SPEC = importlib.util.spec_from_file_location("fig14b_diagnostics", SCRIPT)
diag = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(diag)


def test_velocity_mesh_matches_midpoint_construction():
    cfg = configparser.ConfigParser()
    cfg.read_dict({
        "constants": {"Nv": "4"},
        "non-dim": {"T0": "1", "molarMass0": str(2 * diag.R_UNIVERSAL)},
        "velocity-mesh": {"cmax": "0", "Tmax": "1", "dev": "2"},
    })
    cv, cw = diag.velocity_mesh(cfg)
    assert cv.shape == (3, 64)
    assert np.isclose(cv[0].min(), -1.5)
    assert np.isclose(cv[0].max(), 1.5)
    assert np.isclose(cw, 1.0)


def test_inventory_detects_negative_tail_without_hiding_mass():
    soln = np.array([
        [[1.0], [2.0]],
        [[1.0], [-0.5]],
        [[1.0], [2.0]],
    ])
    xweights = np.array([[0.25], [0.5], [0.25]])
    cv = np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    result = diag.phase_space_inventory(soln, xweights, cv, cw=1.0)
    assert result["negative_count"] == 1
    assert result["min_f"] == -0.5
    assert result["negative_mass"] > 0.0
    assert result["negative_mass_fraction"] > 0.0


def _gll_nodes(left, right):
    return np.vstack((left, 0.5 * (left + right), right))


def test_phase_alignment_removes_a_pure_translation():
    left = np.array([-1.0, 0.0])
    right = np.array([0.0, 1.0])
    nodes = _gll_nodes(left, right)
    velocities = np.array([0.0, 1.0, 2.0, 3.0])
    delta = 0.08

    def distribution(x, center):
        return (1.0 + (x[:, None, :] - center) ** 2) * (
            1.0 + 0.1 * velocities[None, :, None]
        )

    previous = distribution(nodes, 0.0)
    current = distribution(nodes, delta)
    result = diag.distribution_residual(
        previous,
        current,
        previous_center=0.0,
        current_center=delta,
        left=left,
        right=right,
        cw=1.0,
        h0=2.0,
        chunk_size=2,
        nquad=8,
    )
    assert result["unaligned_full_f_L2"] > 1.0e-3
    assert result["phase_aligned_full_f_L2"] < 1.0e-12
    assert np.isclose(
        result["shock_displacement_lambda"],
        delta * 2.0 / diag.MEAN_FREE_PATH_M,
    )


def test_exact_gll_cell_average_of_quadratic():
    # Values of x^2 at GLL nodes -1, 0, 1 integrate to an average of 1/3.
    moments = np.zeros((3, 1, 1))
    moments[:, 0, 0] = [1.0, 0.0, 1.0]
    _, _, average = diag.dense_bulk_profile(
        moments, np.array([-1.0]), np.array([1.0]), points_per_element=9
    )
    assert np.isclose(average[0, 0], 1.0 / 3.0)
