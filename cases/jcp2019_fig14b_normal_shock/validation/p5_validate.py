#!/usr/bin/env python3
"""Phase 5 physics-validation gate for the JCP-2019 helium normal shock.

The script deliberately separates exact conservation checks, internal angular
refinement, and comparison with an independent DSMC/DVM data set.  It never
reports an independent-validation PASS when the external data are absent.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

FIELDS = ("rho_kg_m3", "ux_m_s", "T_K", "qx_W_m2", "Pxx_minus_p_Pa")
REQUIRED_EXTERNAL = ("x_mm",) + FIELDS


def read_csv(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def finite_float(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}: {row[key]}")
    return value


def profile(rows, run, time):
    selected = [r for r in rows if r["run"] == run and abs(float(r["time"]) - time) < 5e-10]
    if not selected:
        raise ValueError(f"no rows for run={run}, time={time:g}")
    selected.sort(key=lambda r: float(r["x_mm"]))
    return {key: np.asarray([finite_float(r, key) for r in selected])
            for key in ("x_mm",) + FIELDS + ("uz_m_s",)}


def shock_center(p):
    x, rho = p["x_mm"], p["rho_kg_m3"]
    target = 0.5*(np.mean(rho[:2]) + np.mean(rho[-2:]))
    idx = int(np.argmin(np.abs(rho - target)))
    if idx == 0 or idx == len(x) - 1:
        return float(x[idx])
    j = idx - 1 if (rho[idx] - target)*(rho[idx - 1] - target) <= 0 else idx + 1
    if rho[j] == rho[idx]:
        return float(x[idx])
    return float(x[idx] + (target - rho[idx])*(x[j] - x[idx])/(rho[j] - rho[idx]))


def norms(a, b, scale):
    d = np.asarray(a) - np.asarray(b)
    scale = max(abs(float(scale)), 1e-300)
    return {"L1": float(np.mean(np.abs(d))/scale),
            "L2": float(np.sqrt(np.mean(d*d))/scale),
            "Linf": float(np.max(np.abs(d))/scale)}


def field_scale(ref, key):
    y = ref[key]
    if key in ("rho_kg_m3", "ux_m_s", "T_K"):
        return abs(float(np.mean(y[-2:]) - np.mean(y[:2])))
    return float(np.max(np.abs(y)))


def compare(candidate, reference):
    xc = candidate["x_mm"] - shock_center(candidate)
    xr = reference["x_mm"] - shock_center(reference)
    mask = (xc >= xr.min()) & (xc <= xr.max())
    result = {}
    for key in FIELDS:
        interpolated = np.interp(xc[mask], xr, reference[key])
        result[key] = norms(candidate[key][mask], interpolated, field_scale(reference, key))
    return result


def external_profile(path):
    rows = read_csv(path)
    if not rows:
        raise ValueError("external reference CSV is empty")
    missing = [k for k in REQUIRED_EXTERNAL if k not in rows[0]]
    if missing:
        raise ValueError("external reference is missing columns: " + ", ".join(missing))
    rows.sort(key=lambda r: float(r["x_mm"]))
    return {key: np.asarray([finite_float(r, key) for r in rows])
            for key in REQUIRED_EXTERNAL}


def external_provenance(data_path, provenance_path):
    provenance = json.loads(Path(provenance_path).read_text())
    required = ("method", "source", "independent_of_dgfs", "same_physical_case", "sha256")
    missing = [key for key in required if key not in provenance]
    if missing:
        raise ValueError("external provenance is missing: " + ", ".join(missing))
    if str(provenance["method"]).upper() not in ("DSMC", "DVM"):
        raise ValueError("external method must be DSMC or DVM")
    if provenance["independent_of_dgfs"] is not True:
        raise ValueError("independent_of_dgfs must be true")
    if provenance["same_physical_case"] is not True:
        raise ValueError("same_physical_case must be true")
    digest = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    if digest.lower() != str(provenance["sha256"]).lower():
        raise ValueError("external CSV SHA-256 does not match provenance")
    provenance["verified_sha256"] = digest
    return provenance


def suspiciously_identical(candidate, external):
    if len(candidate["x_mm"]) != len(external["x_mm"]):
        return False
    if not np.allclose(candidate["x_mm"], external["x_mm"], rtol=0.0, atol=1e-12):
        return False
    return all(np.allclose(candidate[key], external[key], rtol=1e-12, atol=1e-14)
               for key in FIELDS)


def rh_audit():
    molar_mass = 4.0047e-3
    gas_constant = 8.31446261815324/molar_mass
    gamma = 5.0/3.0
    left = dict(rho=1.916e-5, u=1398.771, T=223.0)
    right = dict(rho=3.505e-5, u=764.659, T=354.762)

    def flux(s):
        p = s["rho"]*gas_constant*s["T"]
        mass = s["rho"]*s["u"]
        momentum = s["rho"]*s["u"]**2 + p
        enthalpy = gamma/(gamma - 1.0)*gas_constant*s["T"]
        energy = mass*(enthalpy + 0.5*s["u"]**2)
        return dict(mass=mass, momentum=momentum, energy=energy)

    fl, fr = flux(left), flux(right)
    mismatch = {k: abs(fl[k] - fr[k])/max(abs(fl[k]), abs(fr[k])) for k in fl}
    return {"left_flux": fl, "right_flux": fr, "relative_mismatch": mismatch,
            "tolerance": 2.0e-3,
            "pass": max(mismatch.values()) < 2.0e-3}


def markdown(report):
    lines = ["# P5 normal-shock validation audit", "",
             f"- Audit time in restart coordinates: `{report['time']}`",
             f"- Rankine-Hugoniot gate: **{report['rankine_hugoniot']['pass']}**",
             f"- Independent reference: **{report['independent_status']}**", ""]
    lines += ["## Rankine-Hugoniot relative flux mismatch", "",
              "| mass | momentum | total energy |", "|---:|---:|---:|",
              "| {mass:.3e} | {momentum:.3e} | {energy:.3e} |".format(
                  **report["rankine_hugoniot"]["relative_mismatch"]), ""]
    lines += ["## Internal angular-refinement check", "",
              "This is a numerical sensitivity check, not an independent validation.", "",
              "| run | field | normalized L2 | normalized Linf |",
              "|---|---|---:|---:|"]
    for run, fields in report["internal_comparison"].items():
        for field, values in fields.items():
            lines.append(f"| {run} | {field} | {values['L2']:.3e} | {values['Linf']:.3e} |")
    lines += ["", "## Scientific verdict", "", report["verdict"], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--external-reference")
    parser.add_argument("--external-provenance")
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--external-l2-limit", type=float, default=0.05)
    args = parser.parse_args()

    rows = read_csv(args.profiles)
    runs = sorted({r["run"] for r in rows if abs(float(r["time"]) - args.time) < 5e-10})
    if "run_M24_raw" not in runs:
        raise SystemExit("P5_M24_REFERENCE_MISSING")
    reference = profile(rows, "run_M24_raw", args.time)
    internal = {run: compare(profile(rows, run, args.time), reference)
                for run in runs if run != "run_M24_raw"}

    report = {"schema_version": 1, "time": args.time,
              "rankine_hugoniot": rh_audit(),
              "internal_reference": "run_M24_raw (not independent)",
              "internal_comparison": internal}

    if args.external_reference:
        if not args.external_provenance:
            raise SystemExit("P5_EXTERNAL_PROVENANCE_REQUIRED")
        ext = external_profile(args.external_reference)
        provenance = external_provenance(args.external_reference, args.external_provenance)
        if suspiciously_identical(reference, ext):
            raise SystemExit("P5_EXTERNAL_REFERENCE_DUPLICATES_INTERNAL_DGFS")
        external = {run: compare(profile(rows, run, args.time), ext) for run in runs}
        worst_l2 = max(v["L2"] for fields in external.values() for v in fields.values())
        passed = report["rankine_hugoniot"]["pass"] and worst_l2 <= args.external_l2_limit
        report.update(external_reference=str(Path(args.external_reference).resolve()),
                      external_provenance=provenance,
                      external_comparison=external, external_worst_normalized_L2=worst_l2,
                      independent_status="PASS" if passed else "FAIL")
        report["verdict"] = ("Independent same-physics comparison passed the configured gates."
                             if passed else "Independent data were supplied, but at least one gate failed.")
    else:
        report["independent_status"] = "INCOMPLETE_EXTERNAL_REFERENCE_REQUIRED"
        report["verdict"] = ("The run passes only endpoint/conservation and internal-resolution auditing. "
                             "It is not independently validated until raw same-physics DSMC/DVM data are supplied.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/"p5_validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out/"p5_validation.md").write_text(markdown(report) + "\n")
    (out/"P5_AUDIT_COMPLETE").touch()
    if report["independent_status"] == "PASS":
        (out/"P5_INDEPENDENT_VALIDATION_PASS").touch()
    print("P5_AUDIT_COMPLETE")
    print("P5_RH_PASS=" + str(report["rankine_hugoniot"]["pass"]).lower())
    print("P5_INDEPENDENT_STATUS=" + report["independent_status"])


if __name__ == "__main__":
    main()
