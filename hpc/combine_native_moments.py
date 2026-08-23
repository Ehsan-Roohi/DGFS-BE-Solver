#!/usr/bin/env python3
"""Combine partitioned DGFS moment output into the ordering of a one-part mesh."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np


PART = re.compile(r"^spt_line_p([0-9]+)$")


def scalar(h5: h5py.File, key: str):
    value = h5[key][()]
    return value.item() if isinstance(value, np.ndarray) and value.shape == () else value


def centres(points: np.ndarray) -> np.ndarray:
    return 0.5 * (points[:, :, 0].min(axis=0) + points[:, :, 0].max(axis=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-mesh", type=Path, required=True)
    ap.add_argument("--source-bulk", type=Path, required=True)
    ap.add_argument("--target-mesh", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    pieces = []
    with h5py.File(args.source_mesh, "r") as hm, h5py.File(args.source_bulk, "r") as hb:
        if scalar(hm, "mesh_uuid") != scalar(hb, "mesh_uuid"):
            raise SystemExit("SOURCE_BULK_MESH_UUID_MISMATCH")
        for key in hm:
            match = PART.match(key)
            if not match:
                continue
            rank = match.group(1)
            moment_key = f"moments_line_p{rank}"
            if moment_key not in hb:
                raise SystemExit(f"missing {moment_key}")
            x = centres(hm[key][...])
            moments = hb[moment_key][...]
            if moments.shape[-1] != len(x):
                raise SystemExit(f"element mismatch for partition {rank}")
            pieces.extend((float(xx), moments[..., i]) for i, xx in enumerate(x))
        metadata = {key: hb[key][()] for key in hb if not key.startswith("moments_") and key != "mesh_uuid"}

    if not pieces:
        raise SystemExit("NO_PARTITIONED_MOMENTS_FOUND")
    with h5py.File(args.target_mesh, "r") as ht:
        target_points = ht["spt_line_p0"][...]
        target_centres = centres(target_points)
        target_uuid = ht["mesh_uuid"][()]

    source_centres = np.array([item[0] for item in pieces])
    source_moments = np.stack([item[1] for item in pieces], axis=-1)
    order_source = np.argsort(source_centres)
    order_target = np.argsort(target_centres)
    if not np.allclose(source_centres[order_source], target_centres[order_target], rtol=0, atol=1e-13):
        raise SystemExit("SOURCE_TARGET_ELEMENT_CENTRES_DIFFER")
    combined = np.empty_like(source_moments)
    combined[..., order_target] = source_moments[..., order_source]

    with h5py.File(args.output, "w") as out:
        out["moments_line_p0"] = combined
        for key, value in metadata.items():
            out[key] = value
        out["mesh_uuid"] = target_uuid
    print(f"MOMENTS_COMBINE_COMPLETE elements={combined.shape[-1]} output={args.output}")


if __name__ == "__main__":
    main()
