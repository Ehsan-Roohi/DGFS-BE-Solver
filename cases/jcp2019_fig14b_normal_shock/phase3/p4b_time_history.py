#!/usr/bin/env python3
"""Extract clean-start P4A time histories without rerunning the solver."""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_restarts as cr


RUNS = (
    ("run_M16_raw", 16, "none"),
    ("run_M16_fplus", 16, "fplus"),
    ("run_M24_raw", 24, "none"),
)
FIELDS = ("rho", "ux", "T", "qx", "Pxx_minus_p", "uz")


def time_from_name(path: Path) -> float:
    match = re.search(r"-([0-9]+(?:\.[0-9]+)?)\.frfss$", path.name)
    if not match:
        raise ValueError(f"cannot parse time from {path}")
    return float(match.group(1))


def profile(record: dict, key: str) -> np.ndarray:
    return np.asarray([[p[key] for p in row] for row in record["points"]], dtype=float)


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="completed P4A run directory")
    ap.add_argument("--out-dir", type=Path, default=Path("p4b_results"))
    args = ap.parse_args()

    root = args.root.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = root / "dgfs_fig14b.ini"
    mesh_path = root / "mesh.frfsm"
    ref_path = root / "dist_dgfs_fig14b-0.0.frfss"
    for path in (cfg_path, mesh_path, ref_path):
        if not path.is_file():
            raise SystemExit(f"P4B_INPUT_MISSING {path}")

    cfg = cr.read_ini(cfg_path)
    cv, cw, nd = cr.velocity_mesh(cfg)
    x, order, left, right = cr.load_mesh(mesh_path)
    xw = cr.GLL2_WEIGHTS[:, None] * (right - left)[None, :]
    h0 = cfg.getfloat("non-dim", "H0")
    rho_mid = 0.5 * (
        cfg.getfloat("soln-bcs-left", "rho") + cfg.getfloat("soln-bcs-right", "rho")
    ) / nd["rho0"]

    def analyse(path: Path) -> dict:
        result = cr.analyse_snapshot(path, cv, cw, x, xw, order, nd)
        result["shock_position_nondim"] = cr.shock_position(x, profile(result, "rho"), rho_mid)
        return result

    reference = analyse(ref_path)
    if abs(float(reference["tcurr"] or 0.0)) > 1e-12:
        raise SystemExit(f"P4B_REFERENCE_NOT_T0 t={reference['tcurr']}")
    if reference["max_abs_uz_m_per_s"] > 1e-10:
        raise SystemExit(f"P4B_REFERENCE_NOT_CLEAN uz={reference['max_abs_uz_m_per_s']:.16e}")

    snapshots: dict[str, dict[float, dict]] = {}
    common_times: set[float] | None = None
    for name, _, _ in RUNS:
        run_dir = root / name
        files = sorted(run_dir.glob("dist_p3b_*.frfss"), key=time_from_name)
        mapping = {round(time_from_name(path), 10): analyse(path) for path in files}
        mapping[0.0] = reference
        snapshots[name] = mapping
        common_times = set(mapping) if common_times is None else common_times & set(mapping)
    times = sorted(common_times or ())
    if times != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise SystemExit(f"P4B_TIME_SET_INVALID {times}")

    scales = {
        "rho": nd["rho0"],
        "ux": nd["u0"],
        "T": nd["T0"],
        "qx": nd["rho0"] * nd["u0"] ** 3,
        "Pxx_minus_p": nd["rho0"] * nd["u0"] ** 2,
        "uz": nd["u0"],
    }
    units = {
        "rho": "kg_m3", "ux": "m_s", "T": "K", "qx": "W_m2",
        "Pxx_minus_p": "Pa", "uz": "m_s",
    }

    rows = []
    records = []
    for t in times:
        benchmark = snapshots["run_M24_raw"][t]
        for name, M, projection in RUNS:
            record = snapshots[name][t]
            row = {
                "time": t,
                "run": name,
                "angular_Momega": M,
                "projection": projection,
                "max_abs_uz_m_s": record["max_abs_uz_m_per_s"],
                "max_rho_ux_T_overshoot_fraction": max(
                    record["monotone_overshoot"][key]["fraction_of_jump"]
                    for key in ("rho", "ux", "T")
                ),
                "min_f": record["min_f"],
                "max_negative_mass_fraction": record["max_negative_mass_fraction"],
                "shock_position_mm": (
                    record["shock_position_nondim"] * h0 * 1e3
                    if record["shock_position_nondim"] is not None else math.nan
                ),
                "qx_min_W_m2": record["extrema"]["qx"]["minimum"] * scales["qx"],
                "stress_max_Pa": record["extrema"]["Pxx_minus_p"]["maximum"] * scales["Pxx_minus_p"],
            }
            for key in FIELDS:
                row[f"rms_{key}_vs_M24_{units[key]}"] = rms(
                    profile(record, key), profile(benchmark, key)
                ) * scales[key]
            rows.append(row)
            records.append({"time": t, "run": name, "summary": row})

    csv_path = out / "p4b_time_history.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    final = {row["run"]: row for row in rows if row["time"] == 1.0}
    raw, plus = final["run_M16_raw"], final["run_M16_fplus"]
    improvements = {}
    for key in FIELDS:
        field = f"rms_{key}_vs_M24_{units[key]}"
        improvements[key] = raw[field] / max(plus[field], 1e-300)
    report = {
        "case": {"Mach": 1.59, "times": times, "reference_tcurr": reference["tcurr"]},
        "nondim": nd,
        "rows": rows,
        "final_raw_to_fplus_rms_improvement": improvements,
        "final_raw_to_fplus_max_abs_uz_ratio": (
            raw["max_abs_uz_m_s"] / max(plus["max_abs_uz_m_s"], 1e-300)
        ),
    }
    (out / "p4b_time_history.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# P4B clean-start time-history audit",
        "",
        "| quantity | raw/fplus RMS improvement at t=1 |",
        "|---|---:|",
    ]
    for key in FIELDS:
        lines.append(f"| {key} | {improvements[key]:.6f} |")
    lines += [
        "",
        f"- raw/fplus max abs uz ratio: {report['final_raw_to_fplus_max_abs_uz_ratio']:.6f}",
        f"- clean reference max abs uz: {reference['max_abs_uz_m_per_s']:.6e} m/s",
        f"- time samples: {', '.join(f'{t:g}' for t in times)}",
    ]
    (out / "p4b_summary.md").write_text("\n".join(lines) + "\n")
    print("P4B_TIME_HISTORY_PASS")
    print(f"P4B_TIMES={','.join(f'{t:g}' for t in times)}")
    print(f"P4B_RAW_TO_FPLUS_MAX_UZ_RATIO={report['final_raw_to_fplus_max_abs_uz_ratio']:.8e}")
    for key in FIELDS:
        print(f"P4B_IMPROVEMENT_{key}={improvements[key]:.8e}")


if __name__ == "__main__":
    main()
