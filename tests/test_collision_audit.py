from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "cases"
    / "jcp2019_fig14b_normal_shock"
    / "audit_collision.py"
)
SPEC = importlib.util.spec_from_file_location("collision_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def symmetric_velocity_grid():
    c0 = np.array([-2.0, -1.0, 1.0, 2.0])
    cx, cy, cz = np.meshgrid(c0, c0, c0, indexing="ij")
    return np.vstack((cx.ravel(), cy.ravel(), cz.ravel()))


def test_constant_collision_source_fails_mass_invariant():
    cv = symmetric_velocity_grid()
    q = np.ones(cv.shape[1])
    result = audit.collision_invariants(q, cv, cw=0.5)
    assert result["relative_cancellation_defect"]["mass"] == 1.0
    assert result["signed"]["mass"] > 0.0


def test_nullspace_source_preserves_all_five_invariants():
    cv = symmetric_velocity_grid()
    c2 = np.sum(cv * cv, axis=0)
    basis = np.vstack((np.ones(cv.shape[1]), cv, 0.5 * c2))
    _, _, vh = np.linalg.svd(basis, full_matrices=True)
    q = vh[-1]
    result = audit.collision_invariants(q, cv, cw=0.25)
    assert max(result["relative_cancellation_defect"].values()) < 1.0e-13


def test_maxwellian_is_positive_and_centered():
    cv = symmetric_velocity_grid()
    f = audit.maxwellian(cv)
    moments = audit.distribution_moments(f, cv, cw=1.0)
    assert np.all(f > 0.0)
    assert np.allclose(moments["velocity"], 0.0, atol=1.0e-14)
    assert moments["negative_count"] == 0
    assert audit.entropy_production(np.zeros_like(f), f, 1.0) == 0.0


def test_point_selection_starts_near_shock_center():
    soln = np.zeros((3, 4, 2))
    x = np.array([[-1.0, 0.2], [-0.5, 0.5], [-0.2, 1.0]])
    selected = audit.select_points(soln, x, max_points=2)
    assert selected == [(2, 0), (0, 1)]
