#!/usr/bin/env python3
"""Extract Figure 14 curves/markers from the vector arXiv PDF.

The PDF contains native vector paths.  We therefore recover path vertices and
marker centres directly instead of tracing raster pixels.  The committed CSV
is the output for arXiv:1809.10186v2, page 14 (PDF page number, 1 based).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET


NS = "{http://www.w3.org/2000/svg}"
NUM = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?")
COLORS = {
    "rho": "rgb(82.421875%, 0%, 0%)",
    "u": "rgb(0%, 0%, 0%)",
    "T": "rgb(17.601013%, 17.601013%, 100%)",
}


def points(path: ET.Element) -> list[tuple[float, float]]:
    values = [float(x) for x in NUM.findall(path.attrib.get("d", ""))]
    return list(zip(values[0::2], values[1::2]))


def axis_map(root: ET.Element, transform: str):
    candidates = []
    for path in root.iter(NS + "path"):
        if (path.attrib.get("transform") == transform and
                path.attrib.get("stroke") == "rgb(0%, 0%, 0%)"):
            pts = points(path)
            if len(pts) == 2:
                candidates.append(pts)
    horizontal = max((p for p in candidates if p[0][0] == p[1][0]),
                     key=lambda p: abs(p[1][1] - p[0][1]))
    vertical = max((p for p in candidates if p[0][1] == p[1][1]),
                   key=lambda p: abs(p[1][0] - p[0][0]))
    y_bottom = horizontal[0][0]
    x_left, x_right = sorted((horizontal[0][1], horizontal[1][1]))
    y_top = min(vertical[0][0], vertical[1][0])

    def convert(point: tuple[float, float]) -> tuple[float, float]:
        raw_y, raw_x = point  # the PDF transform swaps the two coordinates
        x = -8.0 + 14.0 * (raw_x - x_left) / (x_right - x_left)
        y = (y_bottom - raw_y) / (y_bottom - y_top)
        return x, y

    return convert, {"raw_x_left": x_left, "raw_x_right": x_right,
                     "raw_y_top": y_top, "raw_y_bottom": y_bottom}


def is_marker(prop: str, pts: list[tuple[float, float]]) -> bool:
    if not pts:
        return False
    dx = max(x for x, _ in pts) - min(x for x, _ in pts)
    dy = max(y for _, y in pts) - min(y for _, y in pts)
    if prop == "rho":
        return len(pts) >= 10 and 70 < dx < 120 and 70 < dy < 120
    if prop == "T":
        return len(pts) == 4 and 60 < dx < 100 and 60 < dy < 100
    return len(pts) == 5 and 60 < dx < 80 and 60 < dy < 80


def main() -> None:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf", type=Path)
    source.add_argument("--svg", type=Path)
    ap.add_argument("--csv", type=Path, default=Path("fig14_digitized.csv"))
    ap.add_argument("--provenance", type=Path,
                    default=Path("fig14_digitization_provenance.json"))
    args = ap.parse_args()

    tmp = None
    svg = args.svg
    if args.pdf:
        tmp = tempfile.TemporaryDirectory(prefix="fig14-vector-")
        svg = Path(tmp.name) / "page14.svg"
        subprocess.run(["pdftocairo", "-f", "14", "-l", "14", "-svg",
                        str(args.pdf), str(svg)], check=True)
    assert svg is not None
    root = ET.parse(svg).getroot()

    groups: dict[str, list[ET.Element]] = {}
    for path in root.iter(NS + "path"):
        transform = path.attrib.get("transform", "")
        if (path.attrib.get("stroke") in COLORS.values() and
                "48.459" in transform and "0.0389616" in transform):
            groups.setdefault(transform, []).append(path)
    if len(groups) != 2:
        raise RuntimeError(f"expected two Figure 14 panels, found {len(groups)}")
    panels = sorted(groups, key=lambda t: float(t.rsplit(",", 2)[-2].strip()))

    rows: list[dict[str, object]] = []
    calibration = {}
    for panel_index, transform in enumerate(panels):
        mesh = 4 if panel_index == 0 else 8
        convert, calibration[f"panel_{mesh}e"] = axis_map(root, transform)
        for prop, color in COLORS.items():
            segment = 0
            for path in groups[transform]:
                if path.attrib.get("stroke") != color:
                    continue
                pts = points(path)
                width = path.attrib.get("stroke-width")
                if width == "28.75":
                    segment += 1
                    for index, point in enumerate(pts):
                        x, y = convert(point)
                        rows.append({"source": "alexeenko_dgfs", "mesh_elements": mesh,
                                     "property": prop, "kind": "line",
                                     "segment": segment, "point": index,
                                     "x_over_lambda": x, "normalized_value": y})
                elif mesh == 8 and width == "23" and is_marker(prop, pts):
                    centre = ((min(x for x, _ in pts) + max(x for x, _ in pts)) / 2,
                              (min(y for _, y in pts) + max(y for _, y in pts)) / 2)
                    x, y = convert(centre)
                    rows.append({"source": "ohwada_1993", "mesh_elements": "",
                                 "property": prop, "kind": "symbol",
                                 "segment": 0, "point": 0,
                                 "x_over_lambda": x, "normalized_value": y})

    rows.sort(key=lambda r: (str(r["source"]), str(r["mesh_elements"]),
                             str(r["property"]), int(r["segment"]),
                             float(r["x_over_lambda"])))
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**row,
                          "x_over_lambda": f'{float(row["x_over_lambda"]):.9f}',
                          "normalized_value": f'{float(row["normalized_value"]):.9f}'}
                         for row in rows)

    pdf_hash = hashlib.sha256(args.pdf.read_bytes()).hexdigest() if args.pdf else None
    provenance = {
        "paper": "Jaiswal, Alexeenko & Hu, JCP 378 (2019) 178-208",
        "doi": "10.1016/j.jcp.2018.11.001",
        "arxiv": "1809.10186v2",
        "source_url": "https://arxiv.org/pdf/1809.10186",
        "pdf_sha256": pdf_hash,
        "pdf_page_1_based": 14,
        "figure": "14(a,b)",
        "method": "native SVG path extraction; no raster tracing",
        "meaning": {"symbols": "Ohwada (1993) [59]", "lines": "paper DGFS"},
        "axis": {"x": "x/lambda, [-8,6]", "y": "normalized property, [0,1]"},
        "estimated_reading_uncertainty": {"x_over_lambda": 0.01,
                                           "normalized_value": 0.002},
        "calibration": calibration,
        "rows": len(rows),
    }
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"FIG14_DIGITIZATION_COMPLETE rows={len(rows)} csv={args.csv}")


if __name__ == "__main__":
    main()
