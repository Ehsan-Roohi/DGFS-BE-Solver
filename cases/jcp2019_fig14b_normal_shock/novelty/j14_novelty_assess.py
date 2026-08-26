#!/usr/bin/env python3
"""Merge collision, restart, and digitized-paper evidence for the novelty gate."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np

import compare_fig14 as fig14

RUNS = ("M6_raw", "M6_fplus", "M16_raw", "M16_fplus")
STYLE = {
    "M6_raw": ("#b54a45", ""),
    "M6_fplus": ("#315ca8", "8 4"),
    "M16_raw": ("#111111", ""),
    "M16_fplus": ("#15956f", "8 4"),
}


def flatten_alex(ref_lines, prop):
    return [p for segment in ref_lines[(8, prop)] for p in segment]


def line(points, sx, sy):
    return " ".join(
        ("M" if i == 0 else "L") + f"{sx(float(x)):.2f},{sy(float(y)):.2f}"
        for i, (x, y) in enumerate(points)
    )


def write_svg(path, cases, ref_lines, symbols):
    width, height = 1320, 500
    ml, top, gap = 75, 82, 35
    pw = (width - 2*ml - 2*gap)/3
    ph = 335
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111}.grid{stroke:#ddd;stroke-width:.8}.axis{stroke:#222;stroke-width:1.1;fill:none}.tick{font-size:12px}</style>',
        '<text x="660" y="27" text-anchor="middle" font-size="21" font-weight="700">Mach 1.59, Kn ≈ 0.055 — conservative collision correction</text>',
    ]
    lx = 145
    for name in RUNS:
        color, dash = STYLE[name]
        dashattr = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<line x1="{lx}" y1="53" x2="{lx+38}" y2="53" stroke="{color}" stroke-width="2.5"{dashattr}/>')
        out.append(f'<text x="{lx+45}" y="58" font-size="13">{html.escape(name.replace("_", " "))}</text>')
        lx += 180
    out.extend([
        '<line x1="865" y1="53" x2="903" y2="53" stroke="#777" stroke-width="1.8" stroke-dasharray="3 4"/><text x="910" y="58" font-size="13">Alexeenko line</text>',
        '<circle cx="1100" cy="53" r="4" fill="white" stroke="#777"/><text x="1110" y="58" font-size="13">Ohwada symbols</text>',
    ])
    for col, prop in enumerate(fig14.PROPS):
        x0 = ml + col*(pw+gap)
        sx = lambda x, x0=x0: x0 + (x+8)/14*pw
        sy = lambda y: top + ph - (y+0.03)/1.06*ph
        for xt in range(-8, 7, 2):
            xx = sx(xt)
            out.append(f'<line class="grid" x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{top+ph}"/>')
            out.append(f'<text class="tick" x="{xx:.2f}" y="{top+ph+18}" text-anchor="middle">{xt}</text>')
        for yt in np.linspace(0, 1, 6):
            yy = sy(float(yt))
            out.append(f'<line class="grid" x1="{x0:.2f}" y1="{yy:.2f}" x2="{x0+pw:.2f}" y2="{yy:.2f}"/>')
            if col == 0:
                out.append(f'<text class="tick" x="{x0-8}" y="{yy+4:.2f}" text-anchor="end">{yt:.1f}</text>')
        out.append(f'<rect class="axis" x="{x0:.2f}" y="{top}" width="{pw:.2f}" height="{ph}"/>')
        out.append(f'<text x="{x0+pw/2:.2f}" y="{top-10}" text-anchor="middle" font-size="16" font-weight="600">{html.escape(fig14.LABEL[prop])}</text>')
        for segment in ref_lines[(8, prop)]:
            out.append(f'<path d="{line(segment, sx, sy)}" fill="none" stroke="#777" stroke-width="1.8" stroke-dasharray="3 4"/>')
        for x, y in symbols[prop]:
            out.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.3" fill="white" stroke="#777" stroke-width="1.3"/>')
        for name in RUNS:
            color, dash = STYLE[name]
            dashattr = f' stroke-dasharray="{dash}"' if dash else ""
            for x, y in cases[name]["segments"][prop]:
                out.append(f'<path d="{line(zip(x, y), sx, sy)}" fill="none" stroke="{color}" stroke-width="2.1"{dashattr}/>')
        out.append(f'<text x="{x0+pw/2:.2f}" y="{top+ph+43}" text-anchor="middle" font-size="14">(x − x_s) / λ</text>')
    out.append('</svg>')
    path.write_text("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", type=Path, required=True)
    ap.add_argument("--audit-m6", type=Path, required=True)
    ap.add_argument("--audit-m16", type=Path, required=True)
    ap.add_argument("--paper-cases", type=Path, required=True)
    ap.add_argument("--paper-reference", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison = json.loads(args.comparison.read_text())
    audits = {
        6: json.loads(args.audit_m6.read_text()),
        16: json.loads(args.audit_m16.read_text()),
    }
    technical = {r["run"].removeprefix("run_"): r for r in comparison["runs"]}
    cases = {name: fig14.load_case(args.paper_cases/name) for name in RUNS}
    ref_lines, symbols = fig14.load_reference(args.paper_reference)

    rows = []
    metrics = {}
    for name in RUNS:
        M = 6 if name.startswith("M6_") else 16
        mode = "fplus" if name.endswith("_fplus") else "raw"
        paper = {}
        vals = []
        for prop in fig14.PROPS:
            alex = fig14.metric(cases[name]["segments"][prop], flatten_alex(ref_lines, prop))
            ohwada = fig14.metric(cases[name]["segments"][prop], symbols[prop])
            paper[prop] = {"alexeenko_dgfs": alex, "ohwada_1993": ohwada}
            vals.extend((alex["rms"], ohwada["rms"]))
        tech = technical[name]
        collision = audits[M]["summary"]
        defect = collision[f"{mode}_max_defect"]
        tms = collision[
            "median_fplus_collision_ms" if mode == "fplus"
            else "median_raw_collision_ms"
        ]
        over = collision["median_projection_overhead_ratio"] if mode == "fplus" else 1.0
        row = {
            "run": name,
            "M_omega": M,
            "projection": mode,
            "paper_mean_rms": float(np.mean(vals)),
            "paper_max_rms": float(np.max(vals)),
            "collision_max_defect": defect,
            "collision_median_ms": tms,
            "collision_overhead_ratio": over,
            "wall_seconds": tech["wall_seconds_job"],
            "rho_overshoot": tech["monotone_overshoot"]["rho"]["fraction_of_jump"],
            "u_overshoot": tech["monotone_overshoot"]["ux"]["fraction_of_jump"],
            "T_overshoot": tech["monotone_overshoot"]["T"]["fraction_of_jump"],
            "negative_mass_fraction": tech["max_negative_mass_fraction"],
            "min_f": tech["min_f"],
            "shock_position_mm": cases[name]["shock_center_mm"],
        }
        rows.append(row)
        metrics[name] = {"paper": paper, "summary": row}

    by = {r["run"]: r for r in rows}
    gates = {
        "M6_fplus_conserves_Q": audits[6]["summary"]["fplus_max_defect"] <= 5e-12,
        "M16_fplus_conserves_Q": audits[16]["summary"]["fplus_max_defect"] <= 5e-12,
        "M6_projection_overhead_le_50pct": audits[6]["summary"]["median_projection_overhead_ratio"] <= 1.50,
        "M16_projection_overhead_le_50pct": audits[16]["summary"]["median_projection_overhead_ratio"] <= 1.50,
        "M6_paper_accuracy_not_degraded_10pct": by["M6_fplus"]["paper_mean_rms"] <= 1.10*by["M6_raw"]["paper_mean_rms"] + 1e-6,
        "M16_paper_accuracy_not_degraded_10pct": by["M16_fplus"]["paper_mean_rms"] <= 1.10*by["M16_raw"]["paper_mean_rms"] + 1e-6,
        "M6_overshoot_not_worse": max(by["M6_fplus"][f"{p}_overshoot"] for p in ("rho", "u", "T")) <= max(by["M6_raw"][f"{p}_overshoot"] for p in ("rho", "u", "T")) + 5e-3,
        "M16_overshoot_not_worse": max(by["M16_fplus"][f"{p}_overshoot"] for p in ("rho", "u", "T")) <= max(by["M16_raw"][f"{p}_overshoot"] for p in ("rho", "u", "T")) + 5e-3,
    }
    finite = all(
        math.isfinite(float(v))
        for r in rows for k, v in r.items()
        if k not in ("run", "projection") and v is not None
    )
    gates["all_metrics_finite"] = finite
    passed = all(gates.values())

    report = {
        "schema_version": 1,
        "case": "JCP2019_Figure14_Mach1.59_Kn0.055_grid8",
        "novelty": (
            "weighted five-moment conservative projection of the fast-spectral "
            "collision term, audited across angular order and validated against "
            "digitized Alexeenko DGFS lines and Ohwada symbols"
        ),
        "scope_warning": (
            "fplus projects Q with w=max(f,0); it neither clips nor guarantees "
            "global positivity of the distribution f"
        ),
        "claim_gate_pass": passed,
        "gates": gates,
        "collision_audits": {str(M): audits[M]["summary"] for M in (6, 16)},
        "runs": metrics,
    }
    (args.output_dir/"novelty_report.json").write_text(json.dumps(report, indent=2)+"\n")
    with (args.output_dir/"novelty_metrics.csv").open("w", newline="") as stream:
        wr = csv.DictWriter(stream, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)

    lines = [
        "# JCP Figure 14 conservative-collision novelty audit",
        "",
        f"**Claim gate:** {'PASS' if passed else 'FAIL'}",
        "",
        "The fplus operation is a weighted five-moment projection of the collision term Q; it is not a positivity claim for f.",
        "",
        "| run | M_omega | projection | paper mean RMS | max collision defect | collision ms | overhead | run wall s | max overshoot | negative mass |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        ov = max(r[f"{p}_overshoot"] for p in ("rho", "u", "T"))
        lines.append(
            f"| {r['run']} | {r['M_omega']} | {r['projection']} | "
            f"{r['paper_mean_rms']:.4e} | {r['collision_max_defect']:.3e} | "
            f"{r['collision_median_ms']:.3f} | {r['collision_overhead_ratio']:.3f} | "
            f"{r['wall_seconds']:.1f} | {ov:.3e} | {r['negative_mass_fraction']:.3e} |"
        )
    lines += ["", "## Gate details", ""]
    lines += [f"- {'PASS' if ok else 'FAIL'} — {key}" for key, ok in gates.items()]
    (args.output_dir/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    write_svg(args.output_dir/"novelty_profiles.svg", cases, ref_lines, symbols)
    (args.output_dir/("CLAIM_GATE_PASS" if passed else "CLAIM_GATE_FAIL")).touch()
    print("\n".join(lines))
    print(f"J14NOV_CLAIM_GATE={'PASS' if passed else 'FAIL'}")
    print("J14NOV_ASSESSMENT_COMPLETE")


if __name__ == "__main__":
    main()
