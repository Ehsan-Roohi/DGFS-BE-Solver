#!/usr/bin/env python3
"""Copy the newest complete JCP14 checkpoint without touching its live run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path

import h5py
import numpy as np


STAMP = re.compile(r"^dist-([0-9]+(?:\.[0-9]+)?)\.frfss$")


def scalar(h5: h5py.File, key: str) -> str:
    value = h5[key][()]
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
        return value.decode() if isinstance(value, bytes) else str(value)
    return str(value)


def valid_pair(mesh: Path, dist: Path, bulk: Path) -> bool:
    try:
        with h5py.File(mesh, "r") as hm, h5py.File(dist, "r") as hd, h5py.File(bulk, "r") as hb:
            uuid = scalar(hm, "mesh_uuid")
            if scalar(hd, "mesh_uuid") != uuid or scalar(hb, "mesh_uuid") != uuid:
                return False
            dsets = [k for k in hd if k.startswith("soln_")]
            msets = [k for k in hb if k.startswith("moments_")]
            if not dsets or not msets:
                return False
            for h5, names in ((hd, dsets), (hb, msets)):
                for name in names:
                    data = h5[name]
                    if not data.size or not np.all(np.isfinite(data[...])):
                        return False
    except (OSError, KeyError, ValueError):
        return False
    return True


def first_residual(path: Path) -> float:
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                raw = float(row["f"])
                normalized = float(row["f_normalized"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(raw) and math.isfinite(normalized):
                return raw
    raise RuntimeError(f"no finite residual in {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    mesh = args.source / "mesh.frfsm"
    cfg = args.source / "dgfs.ini"
    residual = args.source / "kinetic_residual.csv"
    for path in (mesh, cfg, residual):
        if not path.is_file():
            raise SystemExit(f"required source missing: {path}")

    candidates = []
    for dist in args.source.glob("dist-*.frfss"):
        match = STAMP.match(dist.name)
        if not match:
            continue
        time = float(match.group(1))
        bulk = args.source / f"bulk-{time:.1f}.frfss"
        if bulk.is_file() and valid_pair(mesh, dist, bulk):
            candidates.append((time, dist, bulk))
    if not candidates:
        raise SystemExit(f"NO_COMPLETE_CHECKPOINT_FOUND source={args.source}")
    time, dist, bulk = max(candidates, key=lambda item: item[0])

    args.output.mkdir(parents=True, exist_ok=True)
    copies = {
        mesh: args.output / "source_mesh.frfsm",
        dist: args.output / "source_dist.frfss",
        bulk: args.output / "source_bulk.frfss",
        cfg: args.output / "dgfs.ini",
        residual: args.output / "source_kinetic_residual.csv",
    }
    for source, target in copies.items():
        shutil.copy2(source, target)

    baseline = first_residual(residual)
    metadata = {
        "source_directory": str(args.source.resolve()),
        "source_mesh": str(mesh.resolve()),
        "source_distribution": str(dist.resolve()),
        "source_bulk": str(bulk.resolve()),
        "checkpoint_time": time,
        "normalisation_residual": baseline,
    }
    (args.output / "SOURCE.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"JCP14_SNAPSHOT_COMPLETE source={args.source} time={time:.1f}")


if __name__ == "__main__":
    main()
