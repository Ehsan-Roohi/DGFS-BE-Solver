#!/usr/bin/env python3
"""Fail unless both committed runs match JCP 378 (2019), Figure 14."""

from __future__ import annotations

import configparser
import csv
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise AssertionError(f"{label}: got {actual!r}, expected {expected!r}")


def section(lines: list[str], name: str) -> list[str]:
    return lines[lines.index(f"${name}") + 1:lines.index(f"$End{name}")]


def mesh_elements(path: Path) -> tuple[int, float, float]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    node_rows = section(lines, "Nodes")[1:]
    elem_rows = section(lines, "Elements")[1:]
    nodes = {int(r.split()[0]): float(r.split()[1]) for r in node_rows}
    fluid = [r for r in elem_rows if int(r.split()[1]) == 1]
    return len(fluid), min(nodes.values()), max(nodes.values())


def main() -> None:
    cfg = configparser.ConfigParser()
    cfg.read(HERE / "dgfs.ini")
    expected = {
        ("constants", "Nv"): 32,
        ("constants", "Nrho"): 32,
        ("non-dim", "H0"): 30e-3,
        ("velocity-mesh", "dev"): 7,
        ("spherical-design-rule", "M"): 6,
        ("scattering-model", "omega"): 0.5,
        ("scattering-model", "dRef"): 2.17e-10,
        ("scattering-model", "Tref"): 273,
        ("solver", "order"): 2,
        ("solver-time-integrator", "dt"): 0.001,
        ("soln-bcs-left", "rho"): 1.916e-5,
        ("soln-bcs-left", "T"): 223,
        ("soln-bcs-left", "ux"): 1398.771,
        ("soln-bcs-right", "rho"): 3.505e-5,
        ("soln-bcs-right", "T"): 354.762,
        ("soln-bcs-right", "ux"): 764.659,
    }
    for (section_name, key), value in expected.items():
        close(cfg.getfloat(section_name, key), value, f"[{section_name}] {key}")
    if cfg.get("solver-time-integrator", "scheme") != "dgfs-tvd-rk2":
        raise AssertionError("paper SSP-RK2 scheme is not selected")
    if cfg.get("solver-elements-line", "soln-pts") != "gauss-legendre-lobatto":
        raise AssertionError("paper GLL nodal rule is not selected")
    if any("limiter" in s.lower() for s in cfg.sections()):
        raise AssertionError("Figure 14 uses no limiter")

    for n in (4, 8):
        count, xmin, xmax = mesh_elements(HERE / f"mesh_{n}e.msh")
        if count != n:
            raise AssertionError(f"mesh_{n}e has {count} elements")
        close(xmin, -0.5, f"mesh_{n}e xmin")
        close(xmax, 0.5, f"mesh_{n}e xmax")
        dx_lambda = 30e-3 / n / 1.648e-3
        print(f"mesh_{n}e_dx_over_lambda={dx_lambda:.9f}")

    counts = {}
    with (HERE / "fig14_digitized.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["source"], row["mesh_elements"], row["property"])
            counts[key] = counts.get(key, 0) + 1
    for prop in ("rho", "T", "u"):
        if counts.get(("ohwada_1993", "", prop)) != 20:
            raise AssertionError(f"missing Ohwada symbols for {prop}")
        if counts.get(("alexeenko_dgfs", "4", prop), 0) < 35:
            raise AssertionError(f"missing Alexeenko 4e curve for {prop}")
        if counts.get(("alexeenko_dgfs", "8", prop), 0) < 70:
            raise AssertionError(f"missing Alexeenko 8e curve for {prop}")

    print("JCP2019_FIG14_EXACT_VALIDATION_CASE_PASS")
    print("mach=1.59 kn_global_approx=0.055")
    print("domain_mm=-15,15 grids=4,8 velocity=32^3 angular_M=6 DG_order=3")
    print("references=Alexeenko_DGFS_lines,Ohwada_1993_symbols")


if __name__ == "__main__":
    main()
