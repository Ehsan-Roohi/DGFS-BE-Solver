#!/usr/bin/env python3
"""Generate phase-3b restart configs from a completed DGFS run.

Distribution and bulk-moment output cadences may be controlled independently.
This is important for long M16 steady extensions where full distribution
checkpoints are large but bulk fields and residual histories are cheap.
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
    ap.add_argument("--dt-out", default="0.05", help="legacy cadence used for both writers unless split cadences are supplied")
    ap.add_argument("--dist-dt-out", default=None, help="full distribution checkpoint cadence")
    ap.add_argument("--mom-dt-out", default=None, help="bulk-moment output cadence")
    ap.add_argument("--dt", default=None, help="override dt (default: keep base)")
    ap.add_argument("--residual-file", default="kinetic_residual_p3b.csv")
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--out-dir", type=Path, default=Path("configs"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline = baseline_residual(args.residual_csv) if args.residual_csv else None
    dist_dt_out = args.dist_dt_out if args.dist_dt_out is not None else args.dt_out
    mom_dt_out = args.mom_dt_out if args.mom_dt_out is not None else args.dt_out

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
        cfg.set("soln-plugin-dgfsdistwriterstd", "dt-out", str(dist_dt_out))
        cfg.set("soln-plugin-dgfsdistwriterstd", "basedir", ".")
        cfg.set("soln-plugin-dgfsmomwriterstd", "dt-out", str(mom_dt_out))
        cfg.set("soln-plugin-dgfsmomwriterstd", "basedir", ".")
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
        print(
            f"{name}: Nrho={nrho} M={m} projection={proj} "
            f"dist_dt_out={dist_dt_out} mom_dt_out={mom_dt_out} -> {out}"
        )


if __name__ == "__main__":
    main()
