#!/usr/bin/env python3
"""Assess clean-start transverse-velocity generation in Phase 4A."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REQUIRED = {"run_M16_raw", "run_M16_fplus", "run_M24_raw"}
LABELS = {
    "run_M16_raw": "angular M_omega=16, raw",
    "run_M16_fplus": "angular M_omega=16, conservative fplus",
    "run_M24_raw": "angular M_omega=24, raw",
}


def cell_profile(record, key):
    """Exact GLL cell averages; safe to connect between cell centres."""
    return record["cell_averages"][key]


def svg(path, report):
    W, H = 1000, 620
    L, R, T, B = 110, 35, 55, 90
    pw, ph = W - L - R, H - T - B
    ne = len(report["reference"]["cell_averages"]["uz"])
    xs = [-15.0 + 30.0 * (e + 0.5) / ne for e in range(ne)]
    u0 = report["nondim"]["u0"]
    colors = {"run_M16_raw": "#bc6c25", "run_M16_fplus": "#355fa3", "run_M24_raw": "#111111"}
    curves = [("clean t=0 reference", [v * u0 for v in cell_profile(report["reference"], "uz")], "#888", "5,5")]
    curves += [(LABELS[r["run"]], [v * u0 for v in cell_profile(r, "uz")], colors[r["run"]], "")
               for r in report["runs"]]
    lo = min(min(y) for _, y, _, _ in curves)
    hi = max(max(y) for _, y, _, _ in curves)
    pad = max(0.07 * (hi - lo), 1e-10)
    lo, hi = lo - pad, hi + pad
    xp = lambda x: L + (x + 15) / 30 * pw
    yp = lambda y: T + (hi - y) / (hi - lo) * ph
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Arial;fill:#111}.t{font-size:17px}.l{font-size:22px}.g{font-size:17px}</style>',
           '<text class="l" x="500" y="30" text-anchor="middle">Normal shock Ma=1.59: clean-start transverse velocity, t=1</text>']
    for x in (-15, -10, -5, 0, 5, 10, 15):
        out += [f'<line x1="{xp(x):.2f}" y1="{T}" x2="{xp(x):.2f}" y2="{T+ph}" stroke="#ddd"/>',
                f'<text class="t" x="{xp(x):.2f}" y="{T+ph+30}" text-anchor="middle">{x}</text>']
    for i in range(6):
        y = lo + i * (hi - lo) / 5
        out += [f'<line x1="{L}" y1="{yp(y):.2f}" x2="{L+pw}" y2="{yp(y):.2f}" stroke="#ddd"/>',
                f'<text class="t" x="{L-12}" y="{yp(y)+6:.2f}" text-anchor="end">{y:.3e}</text>']
    out += [f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" fill="none" stroke="#111"/>',
            f'<text class="l" x="{L+pw/2}" y="{H-25}" text-anchor="middle">x [mm]</text>',
            f'<text class="l" transform="translate(28 {T+ph/2}) rotate(-90)" text-anchor="middle">u_z [m s^-1]</text>']
    for name, y, color, dash in curves:
        pts = " ".join(f"{xp(x):.2f},{yp(v):.2f}" for x, v in zip(xs, y))
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.6"{ds}/>')
    for i, (name, _, color, dash) in enumerate(curves):
        y = T + 25 + 27 * i
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        out += [f'<line x1="{L+18}" y1="{y}" x2="{L+56}" y2="{y}" stroke="{color}" stroke-width="3"{ds}/>',
                f'<text class="g" x="{L+66}" y="{y+6}">{name}</text>']
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", type=Path, default=Path("p4a_comparison.json"))
    ap.add_argument("--json", type=Path, default=Path("p4a_metrics.json"))
    ap.add_argument("--csv", type=Path, default=Path("p4a_metrics.csv"))
    ap.add_argument("--md", type=Path, default=Path("p4a_metrics.md"))
    ap.add_argument("--svg", type=Path, default=Path("p4a_uz_generation.svg"))
    args = ap.parse_args()
    report = json.loads(args.comparison.read_text())
    if {r["run"] for r in report["runs"]} != REQUIRED:
        raise SystemExit("P4A_RUN_SET_INVALID")
    reference_uz = float(report["reference"]["max_abs_uz_m_per_s"])
    if reference_uz > 1e-10:
        raise SystemExit(f"P4A_REFERENCE_NOT_CLEAN max_abs_uz={reference_uz:.16e}")

    rows = []
    for r in report["runs"]:
        overshoot = max(r["monotone_overshoot"][k]["fraction_of_jump"] for k in ("rho", "ux", "T"))
        row = {
            "run": r["run"], "M": r["M"], "projection": r["projection"],
            "tcurr": r["tcurr"], "wall_seconds": r["wall_seconds_job"],
            "max_abs_uz_m_per_s": r["max_abs_uz_m_per_s"],
            "max_rho_ux_T_overshoot_fraction": overshoot,
            "rms_rho_vs_M24": r["profile_rms_diff_vs_M24_raw"]["rho"],
            "rms_ux_vs_M24": r["profile_rms_diff_vs_M24_raw"]["ux"],
            "rms_T_vs_M24": r["profile_rms_diff_vs_M24_raw"]["T"],
            "rms_qx_vs_M24": r["profile_rms_diff_vs_M24_raw"]["qx"],
            "rms_stress_vs_M24": r["profile_rms_diff_vs_M24_raw"]["Pxx_minus_p"],
            "rms_uz_vs_M24": r["profile_rms_diff_vs_M24_raw"]["uz"],
        }
        if not all(math.isfinite(float(v)) for k, v in row.items() if k not in ("run", "projection")):
            raise SystemExit(f"P4A_NONFINITE {r['run']}")
        if abs(float(r["tcurr"]) - 1.0) > 2e-8:
            raise SystemExit(f"P4A_BAD_FINAL_TIME {r['run']}")
        if overshoot > 1e-8:
            raise SystemExit(f"P4A_MONOTONE_OVERSHOOT {r['run']} {overshoot}")
        rows.append(row)

    raw = next(x for x in rows if x["run"] == "run_M16_raw")
    plus = next(x for x in rows if x["run"] == "run_M16_fplus")
    metrics = {
        "reference_max_abs_uz_m_per_s": reference_uz,
        "runs": rows,
        "raw_to_fplus_max_uz_ratio": raw["max_abs_uz_m_per_s"] / max(plus["max_abs_uz_m_per_s"], 1e-300),
        "fplus_to_raw_uz_rms_ratio": plus["rms_uz_vs_M24"] / max(raw["rms_uz_vs_M24"], 1e-300),
        "fplus_closer_to_M24_in_uz": plus["rms_uz_vs_M24"] < raw["rms_uz_vs_M24"],
    }
    args.json.write_text(json.dumps(metrics, indent=2) + "\n")
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| run | max abs uz [m/s] | max overshoot | RMS uz vs Momega24 | wall [s] |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['run']} | {row['max_abs_uz_m_per_s']:.6e} | "
                     f"{row['max_rho_ux_T_overshoot_fraction']:.3e} | "
                     f"{row['rms_uz_vs_M24']:.3e} | {row['wall_seconds']:.1f} |")
    args.md.write_text("\n".join(lines) + "\n")
    svg(args.svg, report)
    print("\n".join(lines))
    print(f"CLEAN_REFERENCE_MAX_ABS_UZ={reference_uz:.16e}")
    print(f"RAW_TO_FPLUS_MAX_UZ_RATIO={metrics['raw_to_fplus_max_uz_ratio']:.8e}")
    print(f"FPLUS_CLOSER_TO_M24_IN_UZ={str(metrics['fplus_closer_to_M24_in_uz']).lower()}")
    print("P4A_ASSESSMENT_PASS")


if __name__ == "__main__":
    main()
