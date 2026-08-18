#!/usr/bin/env python3
"""Fail unless the committed case matches JCP 378 (2019), Figure 14(b)."""

from __future__ import annotations

import configparser
import importlib.util
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CFG_PATH = HERE / "dgfs_fig14b.ini"
MESH_PATH = HERE / "mesh_fig14b_8elem.msh"


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise AssertionError(f"{label}: got {actual!r}, expected {expected!r}")


def section(lines: list[str], name: str) -> list[str]:
    start = lines.index(f"${name}")
    end = lines.index(f"$End{name}")
    return lines[start + 1 : end]


def verify_1d_solver_core() -> None:
    """Audit the point faces and 1D metric used by the actual solver."""
    import frfs
    from frfs.inifile import Inifile
    from frfs.shapes import LineShape
    from frfs.util import pad_vtk_points

    solver_cfg = Inifile.load(str(CFG_PATH))
    basis = LineShape(2, solver_cfg)
    np.testing.assert_allclose(basis.fpts[:, 0], [-1.0, 1.0])
    np.testing.assert_allclose(basis.fpts_wts, [1.0, 1.0])
    if basis.gbasis_coeffs.shape != (2, 3):
        raise AssertionError(f"invalid 1D correction-basis shape: {basis.gbasis_coeffs.shape}")

    # Load BaseElements without importing optional runtime plugins.  This
    # keeps the preflight inexpensive while exercising the production metric
    # implementation from the selected solver checkout.
    elements_path = Path(frfs.__file__).parent / "solvers/base/elements.py"
    spec = importlib.util.spec_from_file_location("dgfs_elements_audit", elements_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class AuditElements(module.BaseElements):
        privarmap = {1: ["u"]}
        convarmap = privarmap

        @property
        def _scratch_bufs(self):
            return set()

        @staticmethod
        def pri_to_con(pris, cfg):
            return pris

        def set_backend(self, backend, nscal_upts, nonce):
            raise NotImplementedError

    edges = np.linspace(-0.5, 0.5, 9)
    elements = np.stack((edges[:-1], edges[1:]), axis=0)[..., None]
    audit = AuditElements(LineShape, elements, solver_cfg)
    smats, djacs = audit._smats_djacs_mpts
    np.testing.assert_allclose(smats, 1.0)
    np.testing.assert_allclose(djacs, 0.0625)
    np.testing.assert_allclose(audit.rcpdjac_at_np("upts"), 16.0)

    # A one-dimensional VTK file still requires x/y/z point triples.
    points = np.arange(6.0).reshape(2, 3, 1)
    padded = pad_vtk_points(points)
    if padded.shape != (2, 3, 3):
        raise AssertionError(f"invalid padded VTK point shape: {padded.shape}")
    np.testing.assert_allclose(padded[..., 0], points[..., 0])
    np.testing.assert_allclose(padded[..., 1:], 0.0)


def main() -> None:
    cfg = configparser.ConfigParser()
    cfg.read(CFG_PATH)

    expected = {
        ("constants", "Nv"): 32.0,
        ("constants", "Nrho"): 32.0,
        ("non-dim", "H0"): 30e-3,
        ("velocity-mesh", "dev"): 7.0,
        ("spherical-design-rule", "M"): 6.0,
        ("scattering-model", "omega"): 0.5,
        ("scattering-model", "dRef"): 2.17e-10,
        ("scattering-model", "Tref"): 273.0,
        ("solver", "order"): 2.0,
        ("soln-bcs-left", "rho"): 1.916e-5,
        ("soln-bcs-left", "T"): 223.0,
        ("soln-bcs-left", "ux"): 1398.771,
        ("soln-bcs-right", "rho"): 3.505e-5,
        ("soln-bcs-right", "T"): 354.762,
        ("soln-bcs-right", "ux"): 764.659,
    }
    for (group, key), value in expected.items():
        close(cfg.getfloat(group, key), value, f"[{group}] {key}")

    if cfg.get("solver-interfaces-point", "flux-pts") != "gauss-legendre":
        raise AssertionError("1D point-interface quadrature is missing or invalid")
    if cfg.get("solver-elements-line", "soln-pts") != "gauss-legendre-lobatto":
        raise AssertionError("1D solution-point rule is missing or invalid")

    lines = [line.strip() for line in MESH_PATH.read_text().splitlines()]
    node_lines = section(lines, "Nodes")[1:]
    element_lines = section(lines, "Elements")[1:]
    nodes = {int(row.split()[0]): tuple(map(float, row.split()[1:4])) for row in node_lines}
    fluid_lines = [row for row in element_lines if int(row.split()[1]) == 1]
    xs = [xyz[0] for xyz in nodes.values()]

    close(min(xs), -0.5, "mesh x minimum")
    close(max(xs), 0.5, "mesh x maximum")
    if len(fluid_lines) != 8:
        raise AssertionError(f"mesh has {len(fluid_lines)} line elements; expected 8")

    h0 = cfg.getfloat("non-dim", "H0")
    physical_length = (max(xs) - min(xs)) * h0
    dx_over_lambda = physical_length / len(fluid_lines) / 1.648e-3
    close(physical_length, 30e-3, "physical domain length")
    close(dx_over_lambda, 2.275485436893204, "Delta x/lambda")

    if any("limiter" in s.lower() for s in cfg.sections()):
        raise AssertionError("a limiter section is present; the paper case uses no limiter")
    if cfg.getint("soln-plugin-dgfsresidualstd", "nsteps") != 1:
        raise AssertionError("the paper convergence criterion must be sampled every step")
    if not cfg.getboolean("soln-plugin-dgfsresidualstd", "normalise"):
        raise AssertionError("the paper convergence normalization is disabled")
    if cfg.get("solver-time-integrator", "scheme") != "dgfs-tvd-rk2":
        raise AssertionError("the exact case must use the SSP-RK2 DGFS stepper")

    verify_1d_solver_core()

    print("JCP2019_FIG14B_CASE_VERIFIED")
    print(f"physical_length_mm={physical_length * 1e3:.6f}")
    print(f"elements={len(fluid_lines)}")
    print(f"dx_over_lambda={dx_over_lambda:.9f}")
    print("one_dimensional_solver_core=verified")
    print("time_integrator=SSP-RK2")


if __name__ == "__main__":
    main()
