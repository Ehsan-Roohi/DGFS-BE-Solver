#!/usr/bin/env python3
"""P4E: time-history validation of the symmetry-aware transverse projection."""
from __future__ import annotations

import argparse
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
    ("run_M16_raw", 16, "none", "baseline"),
    ("run_M16_fplus", 16, "fplus", "baseline"),
    ("run_M16_transverse", 16, "fplus-transverse", "transverse"),
    ("run_M24_raw", 24, "none", "baseline"),
)
FIELDS = ("rho", "ux", "T", "qx", "Pxx_minus_p", "uz")
UNITS = {
    "rho": "kg_m3", "ux": "m_s", "T": "K", "qx": "W_m2",
    "Pxx_minus_p": "Pa", "uz": "m_s",
}


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
    ap.add_argument("--baseline-root", type=Path, required=True)
    ap.add_argument("--transverse-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("p4e_results"))
    args = ap.parse_args()

    baseline = args.baseline_root.resolve()
    transverse = args.transverse_root.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = baseline / "dgfs_fig14b.ini"
    mesh_path = baseline / "mesh.frfsm"
    ref_path = baseline / "dist_dgfs_fig14b-0.0.frfss"
    for path in (cfg_path, mesh_path, ref_path):
        if not path.is_file():
            raise SystemExit(f"P4E_INPUT_MISSING {path}")

    cfg = cr.read_ini(cfg_path)
    cv, cw, nd = cr.velocity_mesh(cfg)
    x, order, left, right = cr.load_mesh(mesh_path)
    xw = cr.GLL2_WEIGHTS[:, None] * (right - left)[None, :]
    h0 = cfg.getfloat("non-dim", "H0")
    rho_mid = 0.5 * (
        cfg.getfloat("soln-bcs-left", "rho") + cfg.getfloat("soln-bcs-right", "rho")
    ) / nd["rho0"]
    x_mm = x.T.ravel() * h0 * 1e3

    def analyse(path: Path) -> dict:
        result = cr.analyse_snapshot(path, cv, cw, x, xw, order, nd)
        result["shock_position_nondim"] = cr.shock_position(
            x, profile(result, "rho"), rho_mid
        )
        return result

    reference = analyse(ref_path)
    if abs(float(reference["tcurr"] or 0.0)) > 1e-12:
        raise SystemExit(f"P4E_REFERENCE_NOT_T0 t={reference['tcurr']}")
    if reference["max_abs_uz_m_per_s"] > 1e-10:
        raise SystemExit(
            f"P4E_REFERENCE_NOT_SYMMETRIC uz={reference['max_abs_uz_m_per_s']:.16e}"
        )

    roots = {"baseline": baseline, "transverse": transverse}
    snapshots: dict[str, dict[float, dict]] = {}
    common_times: set[float] | None = None
    for name, _, _, source in RUNS:
        run_dir = roots[source] / name
        files = sorted(run_dir.glob("dist_p3b_*.frfss"), key=time_from_name)
        mapping = {round(time_from_name(path), 10): analyse(path) for path in files}
        mapping[0.0] = reference
        snapshots[name] = mapping
        common_times = set(mapping) if common_times is None else common_times & set(mapping)
    times = sorted(common_times or ())
    expected = [0.0, 0.25, 0.5, 0.75, 1.0]
    if times != expected:
        raise SystemExit(f"P4E_TIME_SET_INVALID got={times} expected={expected}")

    scales = {
        "rho": nd["rho0"], "ux": nd["u0"], "T": nd["T0"],
        "qx": nd["rho0"] * nd["u0"] ** 3,
        "Pxx_minus_p": nd["rho0"] * nd["u0"] ** 2,
        "uz": nd["u0"],
    }

    rows: list[dict] = []
    physical_rows: list[dict] = []
    for time in times:
        benchmark = snapshots["run_M24_raw"][time]
        for name, momega, projection, _ in RUNS:
            record = snapshots[name][time]
            row = {
                "time": time, "run": name, "angular_Momega": momega,
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
            }
            for key in FIELDS:
                row[f"rms_{key}_vs_M24_{UNITS[key]}"] = rms(
                    profile(record, key), profile(benchmark, key)
                ) * scales[key]
            rows.append(row)

            values = {key: profile(record, key).T.ravel() * scales[key] for key in FIELDS}
            for index, xpos in enumerate(x_mm):
                physical_rows.append({
                    "time": time, "run": name, "angular_Momega": momega,
                    "projection": projection, "dg_point_index": index, "x_mm": xpos,
                    **{f"{key}_{UNITS[key]}": values[key][index] for key in FIELDS},
                })

    summary_path = out / "p4e_time_history.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    profile_path = out / "p4e_physical_profiles.csv"
    with profile_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(physical_rows[0]))
        writer.writeheader(); writer.writerows(physical_rows)

    by_time = {(r["time"], r["run"]): r for r in rows}
    audit = []
    for time in times[1:]:
        raw = by_time[(time, "run_M16_raw")]
        plus = by_time[(time, "run_M16_fplus")]
        trans = by_time[(time, "run_M16_transverse")]
        ratios = {}
        for key in FIELDS:
            field = f"rms_{key}_vs_M24_{UNITS[key]}"
            ratios[f"raw_over_transverse_{key}"] = raw[field] / max(trans[field], 1e-300)
            ratios[f"fplus_over_transverse_{key}"] = plus[field] / max(trans[field], 1e-300)
        ratios["transverse_over_raw_mean_core_error"] = float(np.mean([
            trans[f"rms_{key}_vs_M24_{UNITS[key]}"] / max(
                raw[f"rms_{key}_vs_M24_{UNITS[key]}"], 1e-300
            ) for key in FIELDS if key != "uz"
        ]))
        ratios["raw_over_transverse_max_abs_uz"] = (
            raw["max_abs_uz_m_s"] / max(trans["max_abs_uz_m_s"], 1e-300)
        )
        audit.append({"time": time, **ratios})

    supported = all(
        item["raw_over_transverse_uz"] >= 3.0
        and item["transverse_over_raw_mean_core_error"] <= 1.10
        and by_time[(item["time"], "run_M16_transverse")][
            "max_rho_ux_T_overshoot_fraction"
        ] <= 1e-10
        for item in audit
    )
    report = {
        "case": {"Mach": 1.59, "times": times, "angular_orders": [16, 24]},
        "reference_max_abs_uz_m_s": reference["max_abs_uz_m_per_s"],
        "rows": rows, "audit": audit,
        "transverse_time_history_supported": supported,
    }
    (out / "p4e_time_history.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# P4E transverse-projection time-history validation", "",
        "Physical Mach number: **Ma = 1.59**. `M_omega` is angular quadrature order, not Mach number.", "",
        "| t | raw/transverse uz RMS | raw/transverse max|uz| | transverse/raw mean core error | max overshoot |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in audit:
        over = by_time[(item["time"], "run_M16_transverse")]["max_rho_ux_T_overshoot_fraction"]
        lines.append(
            f"| {item['time']:.2f} | {item['raw_over_transverse_uz']:.4f} | "
            f"{item['raw_over_transverse_max_abs_uz']:.4f} | "
            f"{item['transverse_over_raw_mean_core_error']:.4f} | {over:.3e} |"
        )
    lines += ["", f"TRANSVERSE_TIME_HISTORY_SUPPORTED={'yes' if supported else 'no'}"]
    (out / "p4e_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("P4E_TIME_HISTORY_COMPLETE")


if __name__ == "__main__":
    main()
