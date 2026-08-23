#!/usr/bin/env python3
"""Deterministically repartition a native 1-D line mesh and DGFS checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from frfs.partitioners.base import BasePartitioner
from frfs.readers.native import NativeReader


class ContiguousLinePartitioner(BasePartitioner):
    dflt_opts: dict[str, object] = {}
    int_opts: set[str] = set()
    enum_opts: dict[str, object] = {}

    def _partition_graph(self, graph, partwts):
        if len(graph.vwts) < len(partwts):
            raise ValueError("more MPI ranks than physical elements")
        parts = np.empty(len(graph.vwts), dtype=np.int32)
        for rank, indices in enumerate(np.array_split(np.arange(len(parts)), len(partwts))):
            parts[indices] = rank
        return parts


def write_h5(path: Path, data: dict[str, object]) -> None:
    with h5py.File(path, "w") as h5:
        for key, value in data.items():
            h5[key] = value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, required=True)
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--solution", type=Path, action="append", default=[])
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.parts < 1:
        raise SystemExit("parts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.mesh, "r") as h5:
        spts = [k for k in h5 if k.startswith("spt_")]
        if not spts or any("_line_" not in k for k in spts):
            raise SystemExit("CONTIGUOUS_PARTITION_REQUIRES_LINE_MESH")

    partitioner = ContiguousLinePartitioner([1] * args.parts, elewts={"line": 1})
    mesh, convert = partitioner.partition(NativeReader(str(args.mesh)))
    write_h5(args.output_dir / args.mesh.name, mesh)
    for solution in args.solution:
        write_h5(args.output_dir / solution.name, convert(NativeReader(str(solution))))
    print(f"LINE_PARTITION_COMPLETE parts={args.parts} mesh={args.mesh}")


if __name__ == "__main__":
    main()
