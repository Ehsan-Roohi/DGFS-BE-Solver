#!/usr/bin/env python3
"""Generate the phase-3b short-restart configs (t = 30 -> 30.1) from dgfs_fig14b.ini.

    python make_restart_configs.py --base dgfs_fig14b.ini --residual-csv kinetic_residual.csv \
        --tstart 30 --tend 30.1 --dt-out 0.05 --out-dir configs \
        --runs M6_raw:32:6:none,M16_raw:16:16:none,M16_fplus:16:16:fplus,M24_raw:16:24:none

Each run spec is  name:Nrho:M:projection  with projection in none|euclidean|f|fplus.
The residual normalisation uses the first finite raw residual of the production
run (same convention as run.slurm), so f_normalized stays comparable.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import math
from pathlib import Path

DEFAULT_RUNS = "M6_raw:32:6:none,M16_raw:16:16:none,M16_fplus:16:16:fplus,M24_raw:16:24:none"


def baseline_residual(path: Path) -> str | None:
    if not path or not path.is_file():
        return None
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                raw = float(row["f"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(raw):
                return f"{raw:.17g}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--residual-csv", type=Path, default=None)
    ap.add_argument("--tstart", default="30")
    ap.add_argument("--tend", default="30.1")
    ap.add_argument("--dt-out", default="0.05")
    ap.add_argument("--dt", default=None, help="override dt (default: keep base)")
    ap.add_argument("--residual-file", default="kinetic_residual_p3b.csv")
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--out-dir", type=Path, default=Path("configs"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline = baseline_residual(args.residual_csv) if args.residual_csv else None

    for spec in args.runs.split(","):
        name, nrho, m, proj = spec.split(":")
        cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        cfg.optionxform = str
        with args.base.open() as stream:
            cfg.read_file(stream)
        cfg.set("constants", "Nrho", nrho)
        cfg.set("spherical-design-rule", "M", m)
        cfg.set("scattering-model", "projection", proj)
        cfg.set("scattering-model", "projection-solve", "device")
        cfg.set("solver-time-integrator", "tstart", str(args.tstart))
        cfg.set("solver-time-integrator", "tend", str(args.tend))
        if args.dt:
            cfg.set("solver-time-integrator", "dt", str(args.dt))
        for sect in ("soln-plugin-dgfsdistwriterstd", "soln-plugin-dgfsmomwriterstd"):
            cfg.set(sect, "dt-out", str(args.dt_out))
            cfg.set(sect, "basedir", ".")
        cfg.set("soln-plugin-dgfsdistwriterstd", "basename", "dist_p3b_%s-{t:.2f}" % name)
        cfg.set("soln-plugin-dgfsmomwriterstd", "basename", "bulksol_p3b_%s-{t:.2f}" % name)
        cfg.set("soln-plugin-dgfsresidualstd", "file", args.residual_file)
        if baseline:
            cfg.set("soln-plugin-dgfsresidualstd", "normalisation-resid", baseline)
        else:
            cfg.remove_option("soln-plugin-dgfsresidualstd", "normalisation-resid")
        out = args.out_dir / f"p3b_{name}.ini"
        with out.open("w") as stream:
            cfg.write(stream)
        print(f"{name}: Nrho={nrho} M={m} projection={proj} -> {out}")


if __name__ == "__main__":
    main()
