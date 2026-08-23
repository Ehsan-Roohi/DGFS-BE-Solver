#!/usr/bin/env python3
"""Create a fixed-physics JCP14 continuation configuration."""

from __future__ import annotations

import argparse
import configparser
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--output-span", required=True)
    ap.add_argument("--normalisation", required=True)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    with args.source.open() as stream:
        cfg.read_file(stream)
    cfg.set("constants", "NvBatchSize", str(args.batch))
    cfg.set("solver-time-integrator", "tstart", args.start)
    cfg.set("solver-time-integrator", "tend", args.end)
    cfg.set("soln-plugin-dgfsdistwriterstd", "dt-out", args.output_span)
    cfg.set("soln-plugin-dgfsmomwriterstd", "dt-out", args.output_span)
    cfg.set("soln-plugin-dgfsresidualstd", "normalisation-resid", args.normalisation)
    if not cfg.has_section("backend-cuda"):
        cfg.add_section("backend-cuda")
    cfg.set("backend-cuda", "device-id", "0")
    cfg.set("backend-cuda", "mpi-type", "standard")
    with args.output.open("w") as stream:
        cfg.write(stream)


if __name__ == "__main__":
    main()
