#!/usr/bin/env python3
"""Add steady-state requirements to the J14 novelty claim gate."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


RUNS = ("M6_raw", "M6_fplus", "M16_raw", "M16_fplus")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steady-threshold", type=float, default=1.0)
    args = parser.parse_args()

    comparison = json.loads(args.comparison.read_text())
    report = json.loads(args.report.read_text())
    technical = {
        item["run"].removeprefix("run_"): item for item in comparison["runs"]
    }

    steady = {}
    for name in RUNS:
        item = technical.get(name, {})
        residual = item.get("residual_final") or {}
        normalized = residual.get("normalized")
        ok = (
            normalized is not None
            and math.isfinite(float(normalized))
            and float(normalized) <= args.steady_threshold
        )
        steady[name] = {
            "converged": ok,
            "time": residual.get("t"),
            "raw_residual": residual.get("raw"),
            "normalized_residual": normalized,
            "threshold": args.steady_threshold,
        }
        report["gates"][f"{name}_steady_state"] = ok

    report["gates"]["all_runs_steady_state"] = all(
        item["converged"] for item in steady.values()
    )
    passed = all(bool(value) for value in report["gates"].values())
    report["schema_version"] = 2
    report["claim_gate_pass"] = passed
    report["validation_state"] = (
        "steady_state_closeout_complete" if passed else "steady_state_closeout_failed"
    )
    report["steady_state"] = steady
    report["run_wall_scope"] = (
        "Continuation wall times are diagnostic only and are not cross-case cost "
        "comparisons; collision-kernel timings define projection overhead."
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    text = args.summary.read_text()
    text = re.sub(
        r"\*\*Claim gate:\*\*\s+(PASS|FAIL)",
        f"**Claim gate:** {'PASS' if passed else 'FAIL'}",
        text,
        count=1,
    )
    lines = [
        "",
        "## Steady-state closeout",
        "",
        f"Acceptance requires the normalized kinetic residual to be at most {args.steady_threshold:g}.",
        "",
        "| run | final time | raw residual | normalized residual | steady |",
        "|---|---:|---:|---:|---|",
    ]
    for name in RUNS:
        item = steady[name]
        time = item["time"]
        raw = item["raw_residual"]
        norm = item["normalized_residual"]
        lines.append(
            f"| {name} | {float(time):.3f} | {float(raw):.4e} | "
            f"{float(norm):.4e} | {'PASS' if item['converged'] else 'FAIL'} |"
            if time is not None and raw is not None and norm is not None
            else f"| {name} | n/a | n/a | n/a | FAIL |"
        )
    lines += [
        "",
        "Continuation wall times are not compared across cases; the audited collision-kernel timings define projection overhead.",
    ]
    args.summary.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for marker in ("CLAIM_GATE_PASS", "CLAIM_GATE_FAIL"):
        (args.output_dir / marker).unlink(missing_ok=True)
    (args.output_dir / ("CLAIM_GATE_PASS" if passed else "CLAIM_GATE_FAIL")).touch()
    status = {
        "claim_gate_pass": passed,
        "steady_threshold": args.steady_threshold,
        "steady_state": steady,
        "failed_gates": [key for key, value in report["gates"].items() if not value],
    }
    (args.output_dir / "CLOSEOUT_STATUS.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )
    print(f"J14NOV_CLOSEOUT_CLAIM_GATE={'PASS' if passed else 'FAIL'}")
    for name, item in steady.items():
        print(
            f"J14NOV_CLOSEOUT_STEADY case={name} pass={int(item['converged'])} "
            f"normalized={item['normalized_residual']}"
        )


if __name__ == "__main__":
    main()
