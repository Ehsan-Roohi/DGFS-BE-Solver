#!/usr/bin/env python3
"""Rebuild the J14 publication gate using the existing steady M16 data.

No solver is run.  M6 profiles/residuals come from the completed closeout;
M16 profiles use the t=340.25 bulk snapshots and the residual histories from
the long steady extension.  Outputs are written directly to --output-dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

RUNS = ("M6_raw", "M6_fplus", "M16_raw", "M16_fplus")


def clean_residual(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                item = (float(row["t"]), float(row["f"]), float(row["f_normalized"]))
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(v) for v in item):
                rows.append(item)
    arr = np.asarray(rows, dtype=float)
    if len(arr) < 3:
        return arr
    keep = np.ones(len(arr), dtype=bool)
    # Output/checkpoint synchronization produces isolated machine-zero rows.
    for i in range(1, len(arr) - 1):
        if arr[i, 1] < 1.0e-3 * min(arr[i - 1, 1], arr[i + 1, 1]):
            keep[i] = False
    return arr[keep]


def sustained_crossing(arr: np.ndarray, threshold: float = 1.0, hold: float = 0.5):
    if arr.size == 0:
        return None
    for i in np.flatnonzero(arr[:, 2] <= threshold):
        j = np.searchsorted(arr[:, 0], arr[i, 0] + hold, side="left")
        if j < len(arr) and np.all(arr[i:j + 1, 2] <= threshold):
            return [float(arr[i, 0]), float(arr[i, 2])]
    return None


def profile_overshoot(case: dict[str, object]) -> dict[str, dict[str, float]]:
    ans = {}
    for prop, key in (("rho", "rho"), ("u", "ux"), ("T", "T")):
        values = np.concatenate([y for _, y in case["segments"][prop]])
        ans[key] = {
            "fraction_of_jump": float(max(0.0, -float(np.min(values)), float(np.max(values)) - 1.0))
        }
    return ans


def copy_case(dst: Path, ini: Path, mesh: Path, bulk: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ini, dst / "dgfs.ini")
    shutil.copy2(mesh, dst / "mesh.frfsm")
    shutil.copy2(bulk, dst / "bulk-final.frfss")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--closeout", type=Path, required=True)
    ap.add_argument("--steady", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path.cwd())
    args = ap.parse_args()

    closeout = args.closeout.resolve()
    steady = args.steady.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = steady / "src"

    validation = source / "cases/jcp2019_fig14b_validation"
    phase3 = source / "cases/jcp2019_fig14b_normal_shock/phase3"
    novelty = source / "cases/jcp2019_fig14b_normal_shock/novelty"
    sys.path[:0] = [str(validation), str(phase3), str(novelty), str(source)]
    import compare_fig14 as fig14

    with tempfile.TemporaryDirectory(prefix="j14_final_gate_") as td:
        tmp = Path(td)
        paper = tmp / "paper_cases"

        for name in ("M6_raw", "M6_fplus"):
            src = closeout / "paper_cases" / name
            copy_case(paper / name, src / "dgfs.ini", src / "mesh.frfsm", src / "bulk-final.frfss")

        for name in ("M16_raw", "M16_fplus"):
            src = steady / "stage_1" / name
            copy_case(
                paper / name,
                src / f"p3b_{name}.ini",
                src / "mesh.frfsm",
                src / f"bulksol_p3b_{name}-340.25.frfss",
            )

        cases = {name: fig14.load_case(paper / name) for name in RUNS}
        comparison = json.loads((closeout / "results/restart_comparison.json").read_text())
        technical = {r["run"].removeprefix("run_"): r for r in comparison["runs"]}

        residuals = {
            "M6_raw": closeout / "final_runs/run_M6_raw/kinetic_residual_p3b.csv",
            "M6_fplus": closeout / "final_runs/run_M6_fplus/kinetic_residual_p3b.csv",
            "M16_raw": steady / "stage_1/M16_raw/kinetic_residual_p3b.csv",
            "M16_fplus": steady / "stage_1/M16_fplus/kinetic_residual_p3b.csv",
        }
        residual_summary = {}
        for name, path in residuals.items():
            arr = clean_residual(path)
            if arr.size == 0:
                raise RuntimeError(f"no finite residual data for {name}")
            final = arr[-1]
            cross = sustained_crossing(arr)
            technical[name]["residual_final"] = {
                "t": float(final[0]), "raw": float(final[1]), "normalized": float(final[2])
            }
            technical[name]["residual_sustained_crossing"] = cross
            residual_summary[name] = {
                "final": technical[name]["residual_final"], "sustained_crossing": cross
            }

        # Replace the old transient M16 overshoot metrics by the t=340.25 steady profiles.
        for name in ("M16_raw", "M16_fplus"):
            technical[name]["monotone_overshoot"] = profile_overshoot(cases[name])

        comparison_path = tmp / "restart_comparison_final.json"
        comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")

        env = dict(os.environ)
        env["PYTHONPATH"] = ":".join([str(validation), str(phase3), str(source), env.get("PYTHONPATH", "")])
        assess = novelty / "j14_novelty_assess.py"
        gate = novelty / "j14_novelty_closeout_gate.py"
        reference = validation / "fig14_digitized.csv"

        subprocess.run([
            sys.executable, str(assess), "--comparison", str(comparison_path),
            "--audit-m6", str(closeout / "audit/M6.json"),
            "--audit-m16", str(closeout / "audit/M16.json"),
            "--paper-cases", str(paper), "--paper-reference", str(reference),
            "--output-dir", str(output),
        ], check=True, env=env)
        subprocess.run([
            sys.executable, str(gate), "--comparison", str(comparison_path),
            "--report", str(output / "novelty_report.json"),
            "--summary", str(output / "SUMMARY.md"), "--output-dir", str(output),
            "--steady-threshold", "1.0",
        ], check=True, env=env)

    report = json.loads((output / "novelty_report.json").read_text())
    shutil.copy2(output / "novelty_profiles.svg", output / "FIG2_MACH1P59_REFERENCE_PROFILES.svg")
    failed = [key for key, value in report["gates"].items() if not value]
    summary = [
        "# DGFS final steady claim gate", "",
        "Benchmark: helium normal shock, Mach 1.59, Kn ≈ 0.055, 8 spatial elements, third-order DG.",
        "M6/M16 denote angular orders M_omega=6/16; they are not Mach numbers.", "",
        f"**FINAL CLAIM GATE: {'PASS' if report['claim_gate_pass'] else 'FAIL'}**", "",
        "## Failed gates", *( ["- none"] if not failed else [f"- {x}" for x in failed] ), "",
        "## Residual evidence",
    ]
    for name in RUNS:
        item = residual_summary[name]
        summary.append(f"- {name}: final={item['final']}; sustained_crossing={item['sustained_crossing']}")
    (output / "FINAL_STEADY_GATE_SUMMARY.md").write_text("\n".join(summary) + "\n")
    (output / "FINAL_STEADY_RESIDUALS.json").write_text(json.dumps(residual_summary, indent=2) + "\n")

    print(f"FINAL_CLAIM_GATE={'PASS' if report['claim_gate_pass'] else 'FAIL'}")
    print("FAILED_GATES=" + ",".join(failed))
    print(f"OUTPUT_DIR={output}")


if __name__ == "__main__":
    main()
