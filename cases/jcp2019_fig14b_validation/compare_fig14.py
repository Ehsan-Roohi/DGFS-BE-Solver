#!/usr/bin/env python3
"""Compare two converged reproductions with vector-digitized Figure 14 data."""

from __future__ import annotations

import argparse
import configparser
import csv
import html
import json
import math
from pathlib import Path

import h5py
import numpy as np


PROPS = ("rho", "T", "u")
FIELD = {"rho": "rho", "T": "T", "u": "U:x"}
COLOR = {"rho": "#d7191c", "T": "#2c4cff", "u": "#111111"}
LABEL = {"rho": "density rho'", "T": "temperature T'", "u": "velocity u'"}
LAMBDA_UPSTREAM = 1.648e-3


def ini_from_h5(value: object) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read_string(value.decode() if isinstance(value, bytes) else str(value))
    return cfg


def basis(r: np.ndarray) -> np.ndarray:
    return np.vstack((0.5 * r * (r - 1), 1 - r * r, 0.5 * r * (r + 1)))


def crossing(segments: list[tuple[np.ndarray, np.ndarray]], target: float) -> float:
    hits = []
    for x, y in segments:
        idx = np.flatnonzero((y[:-1] - target) * (y[1:] - target) <= 0)
        for j in idx:
            if y[j + 1] == y[j]:
                hits.append(float(x[j]))
            else:
                hits.append(float(x[j] + (target - y[j]) *
                                  (x[j + 1] - x[j]) / (y[j + 1] - y[j])))
    if not hits:
        raise RuntimeError("density midpoint crossing not found")
    return min(hits, key=abs)


def load_case(case_dir: Path) -> dict[str, object]:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(case_dir / "dgfs.ini")
    h0 = cfg.getfloat("non-dim", "H0")
    bounds = {
        "rho": (cfg.getfloat("soln-bcs-left", "rho"),
                cfg.getfloat("soln-bcs-right", "rho")),
        "T": (cfg.getfloat("soln-bcs-left", "T"),
              cfg.getfloat("soln-bcs-right", "T")),
        "u": (cfg.getfloat("soln-bcs-left", "ux"),
              cfg.getfloat("soln-bcs-right", "ux")),
    }
    with h5py.File(case_dir / "mesh.frfsm", "r") as h5:
        mesh = h5["spt_line_p0"][()]
    with h5py.File(case_dir / "bulk-final.frfss", "r") as h5:
        moments = h5["moments_line_p0"][()]
        stats = ini_from_h5(h5["stats"][()])
    fields = [x.strip() for x in stats["data"]["fields"].split(",")]
    index = {prop: fields.index(FIELD[prop]) for prop in PROPS}
    left = mesh[:, :, 0].min(axis=0)
    right = mesh[:, :, 0].max(axis=0)
    order = np.argsort(left)
    left, right, moments = left[order], right[order], moments[:, :, order]
    r = np.linspace(-1, 1, 241)
    b = basis(r)
    physical: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {p: [] for p in PROPS}
    for elem in range(moments.shape[2]):
        x = 0.5 * ((1 - r) * left[elem] + (1 + r) * right[elem])
        for prop in PROPS:
            physical[prop].append((x, b.T @ moments[:, index[prop], elem]))
    rho_mid = 0.5 * sum(bounds["rho"])
    x_shock = crossing(physical["rho"], rho_mid)

    normalized: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {p: [] for p in PROPS}
    for prop in PROPS:
        upstream, downstream = bounds[prop]
        for x, values in physical[prop]:
            x_lambda = (x - x_shock) * h0 / LAMBDA_UPSTREAM
            if prop == "u":
                y = (values - downstream) / (upstream - downstream)
            else:
                y = (values - upstream) / (downstream - upstream)
            normalized[prop].append((x_lambda, y))

    weights = np.array([1.0, 4.0, 1.0]) / 6.0
    centres = 0.5 * (left + right)
    cell_average = {}
    for prop in PROPS:
        upstream, downstream = bounds[prop]
        values = np.einsum("i,ie->e", weights, moments[:, index[prop], :])
        if prop == "u":
            values = (values - downstream) / (upstream - downstream)
        else:
            values = (values - upstream) / (downstream - upstream)
        cell_average[prop] = ((centres - x_shock) * h0 / LAMBDA_UPSTREAM, values)
    return {"segments": normalized, "cell_average": cell_average,
            "shock_center_mm": x_shock * h0 * 1e3,
            "elements": moments.shape[2]}


def load_reference(path: Path):
    lines: dict[tuple[int, str], dict[int, list[tuple[float, float]]]] = {}
    symbols: dict[str, list[tuple[float, float]]] = {p: [] for p in PROPS}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            point = (float(row["x_over_lambda"]), float(row["normalized_value"]))
            if row["source"] == "ohwada_1993":
                symbols[row["property"]].append(point)
            else:
                key = (int(row["mesh_elements"]), row["property"])
                lines.setdefault(key, {}).setdefault(int(row["segment"]), []).append(point)
    line_lists = {}
    for key, segments in lines.items():
        line_lists[key] = [sorted(values) for _, values in sorted(segments.items())]
    for prop in PROPS:
        symbols[prop].sort()
    return line_lists, symbols


def evaluate(segments: list[tuple[np.ndarray, np.ndarray]], xs: np.ndarray) -> np.ndarray:
    result = np.full(xs.shape, np.nan)
    for i, target in enumerate(xs):
        candidates = []
        for x, y in segments:
            lo, hi = min(x[0], x[-1]), max(x[0], x[-1])
            if lo - 1e-9 <= target <= hi + 1e-9:
                candidates.append(float(np.interp(target, x, y)))
        if candidates:
            result[i] = float(np.mean(candidates))
    return result


def metric(case_segments, points):
    x = np.array([p[0] for p in points])
    ref = np.array([p[1] for p in points])
    current = evaluate(case_segments, x)
    mask = np.isfinite(current)
    error = current[mask] - ref[mask]
    if not len(error):
        raise RuntimeError("no overlapping comparison points")
    return {"n": int(len(error)), "rms": float(np.sqrt(np.mean(error ** 2))),
            "linf": float(np.max(np.abs(error))),
            "bias": float(np.mean(error))}


def polyline(points, sx, sy) -> str:
    return " ".join(("M" if i == 0 else "L") + f"{sx(x):.2f},{sy(y):.2f}"
                    for i, (x, y) in enumerate(points))


def write_svg(path: Path, cases, ref_lines, symbols) -> None:
    width, height = 1320, 760
    margin_x, top = 76, 94
    gap_x, gap_y = 34, 64
    panel_w = (width - 2 * margin_x - 2 * gap_x) / 3
    panel_h = (height - top - 70 - gap_y) / 2
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111}.tick{font-size:12px}.title{font-size:17px;font-weight:600}.axis{stroke:#222;stroke-width:1.2}.grid{stroke:#ddd;stroke-width:.8}.ours{fill:none;stroke-width:2.8}.paper{fill:none;stroke-width:1.8;stroke-dasharray:6 4}</style>',
           '<text x="660" y="30" text-anchor="middle" font-size="22" font-weight="700">Mach 1.59 helium normal shock — exact JCP Figure 14 validation</text>',
           '<line x1="120" y1="58" x2="170" y2="58" stroke="#444" stroke-width="3"/><text x="180" y="63" font-size="14">present raw DG</text>',
           '<circle cx="365" cy="58" r="4" fill="#444"/><text x="378" y="63" font-size="14">present exact cell average</text>',
           '<line x1="615" y1="58" x2="665" y2="58" stroke="#444" stroke-width="2" stroke-dasharray="6 4"/><text x="675" y="63" font-size="14">Alexeenko DGFS (digitized line)</text>',
           '<circle cx="1030" cy="58" r="5" fill="white" stroke="#444" stroke-width="1.8"/><text x="1043" y="63" font-size="14">Ohwada 1993 (digitized symbols)</text>']
    for row, nelem in enumerate((4, 8)):
        for col, prop in enumerate(PROPS):
            x0 = margin_x + col * (panel_w + gap_x)
            y0 = top + row * (panel_h + gap_y)
            sx = lambda x, x0=x0: x0 + (x + 8) / 14 * panel_w
            sy = lambda y, y0=y0: y0 + panel_h - (y + 0.03) / 1.06 * panel_h
            for xt in range(-8, 7, 2):
                xx = sx(xt)
                out.append(f'<line class="grid" x1="{xx:.2f}" y1="{y0:.2f}" x2="{xx:.2f}" y2="{y0+panel_h:.2f}"/>')
                out.append(f'<text class="tick" x="{xx:.2f}" y="{y0+panel_h+18:.2f}" text-anchor="middle">{xt}</text>')
            for yt in np.linspace(0, 1, 6):
                yy = sy(float(yt))
                out.append(f'<line class="grid" x1="{x0:.2f}" y1="{yy:.2f}" x2="{x0+panel_w:.2f}" y2="{yy:.2f}"/>')
                if col == 0:
                    out.append(f'<text class="tick" x="{x0-9:.2f}" y="{yy+4:.2f}" text-anchor="end">{yt:.1f}</text>')
            out.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{panel_w:.2f}" height="{panel_h:.2f}" fill="none" class="axis"/>')
            out.append(f'<text class="title" x="{x0+panel_w/2:.2f}" y="{y0-10:.2f}" text-anchor="middle">{nelem} elements — {html.escape(LABEL[prop])}</text>')
            for segment in ref_lines[(nelem, prop)]:
                out.append(f'<path class="paper" stroke="{COLOR[prop]}" d="{polyline(segment, sx, sy)}"/>')
            for x, y in symbols[prop]:
                xx, yy = sx(x), sy(y)
                if prop == "rho":
                    out.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="4.5" fill="white" stroke="{COLOR[prop]}" stroke-width="1.7"/>')
                elif prop == "T":
                    out.append(f'<path d="M{xx:.2f},{yy-5:.2f} L{xx-4.8:.2f},{yy+4:.2f} L{xx+4.8:.2f},{yy+4:.2f} Z" fill="white" stroke="{COLOR[prop]}" stroke-width="1.7"/>')
                else:
                    out.append(f'<rect x="{xx-4:.2f}" y="{yy-4:.2f}" width="8" height="8" fill="white" stroke="{COLOR[prop]}" stroke-width="1.7"/>')
            for x, y in cases[nelem]["segments"][prop]:
                pts = list(zip(x, y))
                out.append(f'<path class="ours" stroke="{COLOR[prop]}" d="{polyline(pts, sx, sy)}"/>')
            avg_x, avg_y = cases[nelem]["cell_average"][prop]
            for x, y in zip(avg_x, avg_y):
                out.append(f'<circle cx="{sx(float(x)):.2f}" cy="{sy(float(y)):.2f}" r="3.2" fill="{COLOR[prop]}" stroke="white" stroke-width=".8"/>')
            if row == 1:
                out.append(f'<text x="{x0+panel_w/2:.2f}" y="{y0+panel_h+43:.2f}" text-anchor="middle" font-size="15">(x - x_s) / lambda</text>')
        out.append(f'<text x="18" y="{top+row*(panel_h+gap_y)+panel_h/2:.2f}" text-anchor="middle" font-size="15" transform="rotate(-90 18 {top+row*(panel_h+gap_y)+panel_h/2:.2f})">normalized property</text>')
    out.append('</svg>')
    path.write_text("\n".join(out) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = {n: load_case(args.campaign / f"e{n}") for n in (4, 8)}
    ref_lines, symbols = load_reference(args.reference)

    rows = []
    report = {"case": "JCP2019_Figure14_Mach1.59", "digitized_reference": str(args.reference),
              "shock_center_mm": {}, "metrics": []}
    for nelem in (4, 8):
        report["shock_center_mm"][str(nelem)] = cases[nelem]["shock_center_mm"]
        for prop in PROPS:
            alex_points = [p for segment in ref_lines[(nelem, prop)] for p in segment]
            for source, points in (("alexeenko_dgfs", alex_points),
                                   ("ohwada_1993", symbols[prop])):
                values = metric(cases[nelem]["segments"][prop], points)
                row = {"mesh_elements": nelem, "property": prop,
                       "reference": source, **values}
                rows.append(row)
                report["metrics"].append(row)
            all_values = np.concatenate([y for _, y in cases[nelem]["segments"][prop]])
            report.setdefault("overshoot", {}).setdefault(str(nelem), {})[prop] = {
                "minimum": float(np.min(all_values)), "maximum": float(np.max(all_values)),
                "magnitude": float(max(0.0, -np.min(all_values), np.max(all_values) - 1.0)),
            }

    with (args.output_dir / "fig14_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "fig14_metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    with (args.output_dir / "our_profiles.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mesh_elements", "property", "kind", "segment", "point",
                         "x_over_lambda", "normalized_value"])
        for nelem in (4, 8):
            for prop in PROPS:
                for seg, (x, y) in enumerate(cases[nelem]["segments"][prop], 1):
                    for point, (xx, yy) in enumerate(zip(x, y)):
                        writer.writerow([nelem, prop, "raw_dg", seg, point,
                                         f"{xx:.10g}", f"{yy:.10g}"])
                avg_x, avg_y = cases[nelem]["cell_average"][prop]
                for point, (xx, yy) in enumerate(zip(avg_x, avg_y)):
                    writer.writerow([nelem, prop, "exact_gll_cell_average", 0, point,
                                     f"{xx:.10g}", f"{yy:.10g}"])

    write_svg(args.output_dir / "fig14_validation.svg", cases, ref_lines, symbols)
    lines = ["# JCP 2019 Figure 14 validation", "",
             "Exact Mach 1.59 helium hard-sphere case; no Mach 2/DVM data are used.", "",
             "| elements | property | reference | RMS | L_inf | bias | n |",
             "|---:|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['mesh_elements']} | {row['property']} | {row['reference']} | "
                     f"{row['rms']:.4e} | {row['linf']:.4e} | {row['bias']:+.4e} | {row['n']} |")
    (args.output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("JCP2019_FIG14_COMPARISON_COMPLETE")
    print(args.output_dir / "fig14_validation.svg")


if __name__ == "__main__":
    main()

