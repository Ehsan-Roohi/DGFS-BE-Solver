#!/usr/bin/env python3
"""Shared per-point analysis, per-quadrature summaries, gates and CSV output for
the phase-3 quadrature/projection audit (used by the GPU audit on Unity and by
the offline CPU study)."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from conservative_projection import (INVARIANT_NAMES, WEIGHTINGS, ConservativeProjector,
                                     bulk_moments, entropy_indicator,
                                     euler_update_negativity, negativity)

OPERATORS = ("raw",) + WEIGHTINGS      # raw, euclidean, f, fplus, maxwellian
INTERIOR_ABS_X = 0.40                  # |x| < 0.40 excludes the two boundary elements' outer points


def _named(values) -> dict:
    return {name: float(v) for name, v in zip(INVARIANT_NAMES, values)}


def analyse_point(label: str, f: np.ndarray, q_raw: np.ndarray, P: ConservativeProjector,
                  dt: float, cv: np.ndarray, cw: float, x: float | None = None,
                  upt: int | None = None, elem: int | None = None,
                  gpu_results: dict | None = None, collision_timing: dict | None = None) -> dict:
    """Full per-point record.  ``gpu_results`` (optional) maps weighting ->
    {"Qc": array from the GPU kernels, "timing_ms": {...}, "lam": array|None}."""
    f = np.asarray(f, dtype=float)
    q_raw = np.asarray(q_raw, dtype=float)
    rho, u, T = bulk_moments(f, cv, cw)
    neg_f = negativity(f, cw)
    f_raw_update = f + dt * q_raw
    rec = {
        "label": label, "solution_point": upt, "element": elem, "x_nondim": x,
        "input": {"density": rho, "velocity": [float(v) for v in u], "temperature": T,
                  **neg_f},
        "collision_timing_ms": collision_timing,
        "operators": {},
    }

    def operator_block(q: np.ndarray, ref_update: np.ndarray | None) -> dict:
        m = P.moments(q)
        blk = {
            "moments_signed": _named(m),
            "cancellation_defect": _named(P.cancellation_defects(q)),
            "state_defect": _named(P.state_defects(q, f)),
            "L1": float(cw * np.sum(np.abs(q))),
            "L2": float(math.sqrt(cw * np.dot(q, q))),
            "Linf": float(np.max(np.abs(q))),
            "uz_source_rate": float(m[3] / rho),           # d(u_z)/dt  (nondim)
            "ux_source_rate": float(m[1] / rho),
            "euler_update": euler_update_negativity(f, q, dt, cw, ref_update),
            "entropy_indicator": entropy_indicator(q, f, cw),
        }
        return blk

    rec["operators"]["raw"] = operator_block(q_raw, None)
    for w in WEIGHTINGS:
        try:
            r = P.project(q_raw, f, w)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            rec["operators"][w] = {"error": str(exc)}
            continue
        blk = operator_block(r.Qc, f_raw_update)
        blk.update({
            "rel_correction_l2": float(r.rel_correction_l2),
            "lambda": [float(v) for v in r.lam],
            "scaled_condition": float(r.scaled_condition),
            "max_abs_basis_lam": float(r.max_abs_basis_lam),
            "dt_max_abs_basis_lam": float(dt * r.max_abs_basis_lam),
            "weight_min": r.weight_min, "weight_max": r.weight_max,
        })
        if gpu_results and w in gpu_results:
            g = gpu_results[w]
            qg = np.asarray(g["Qc"], dtype=float)
            blk["gpu"] = {
                "timing_ms": g.get("timing_ms"),
                "max_abs_diff_vs_numpy": float(np.max(np.abs(qg - r.Qc))),
                "max_abs_diff_vs_numpy_rel_Linf": float(np.max(np.abs(qg - r.Qc))
                                                         / max(np.max(np.abs(r.Qc)), 1e-300)),
                "cancellation_defect": _named(P.cancellation_defects(qg)),
                "state_defect": _named(P.state_defects(qg, f)),
                "lambda": ([float(v) for v in g["lam"]] if g.get("lam") is not None else None),
                "euler_update": euler_update_negativity(f, qg, dt, cw, f_raw_update),
            }
        rec["operators"][w] = blk
    return rec


def _maxover(records: list[dict], getter, interior: bool = False) -> float:
    vals = []
    for r in records:
        if interior and (r["x_nondim"] is None or abs(r["x_nondim"]) >= INTERIOR_ABS_X):
            continue
        try:
            vals.append(float(getter(r)))
        except (KeyError, TypeError):
            continue
    return max(vals) if vals else float("nan")


def summarise_quadrature(records: list[dict], dt: float) -> dict:
    """Aggregate over the snapshot points of one (Nrho, M) configuration."""
    snap = [r for r in records if r["element"] is not None]
    out = {"snapshot_points": len(snap), "raw": {}, "projected": {}, "timing": {}}
    raw = out["raw"]
    for norm in ("cancellation_defect", "state_defect"):
        raw[f"max_{norm}_all"] = {name: _maxover(
            snap, lambda r, n=name, k=norm: r["operators"]["raw"][k][n]) for name in INVARIANT_NAMES}
        raw[f"max_{norm}_interior"] = {name: _maxover(
            snap, lambda r, n=name, k=norm: r["operators"]["raw"][k][n], True) for name in INVARIANT_NAMES}
    raw["max_abs_uz_source_rate"] = _maxover(snap, lambda r: abs(r["operators"]["raw"]["uz_source_rate"]))
    raw["max_abs_momentum_z_state_defect"] = raw["max_state_defect_all"]["momentum_z"]
    raw["uz_source_rate_by_point"] = {r["label"]: r["operators"]["raw"]["uz_source_rate"] for r in snap}
    raw["max_negative_mass_fraction_euler_update"] = _maxover(
        snap, lambda r: r["operators"]["raw"]["euler_update"]["negative_mass_fraction"])
    for w in WEIGHTINGS:
        blk = {}
        blk["max_post_cancellation_defect"] = _maxover(
            snap, lambda r: max(r["operators"][w]["cancellation_defect"].values()))
        blk["max_post_state_defect"] = _maxover(
            snap, lambda r: max(r["operators"][w]["state_defect"].values()))
        blk["max_rel_correction_l2_all"] = _maxover(snap, lambda r: r["operators"][w]["rel_correction_l2"])
        blk["max_rel_correction_l2_interior"] = _maxover(
            snap, lambda r: r["operators"][w]["rel_correction_l2"], True)
        blk["median_rel_correction_l2_all"] = float(np.median([
            r["operators"][w]["rel_correction_l2"] for r in snap if "rel_correction_l2" in r["operators"][w]]))
        eu = lambda r, key: r["operators"][w]["euler_update"][key]
        blk["max_negative_mass_ratio_vs_raw"] = _maxover(snap, lambda r: eu(r, "negative_mass_ratio_vs_reference"))
        blk["max_newly_negative_vs_raw"] = _maxover(snap, lambda r: eu(r, "newly_negative_vs_reference"))
        blk["max_min_ratio_vs_raw"] = _maxover(snap, lambda r: eu(r, "min_ratio_vs_reference"))
        blk["max_dt_max_abs_basis_lam"] = _maxover(snap, lambda r: r["operators"][w]["dt_max_abs_basis_lam"])
        blk["max_scaled_condition"] = _maxover(snap, lambda r: r["operators"][w]["scaled_condition"])
        gpu_recs = [r for r in snap if "gpu" in r["operators"].get(w, {})]
        if gpu_recs:
            blk["gpu"] = {
                "max_post_cancellation_defect": _maxover(
                    gpu_recs, lambda r: max(r["operators"][w]["gpu"]["cancellation_defect"].values())),
                "max_abs_diff_vs_numpy": _maxover(gpu_recs, lambda r: r["operators"][w]["gpu"]["max_abs_diff_vs_numpy"]),
                "max_abs_diff_vs_numpy_rel_Linf": _maxover(
                    gpu_recs, lambda r: r["operators"][w]["gpu"]["max_abs_diff_vs_numpy_rel_Linf"]),
                "max_negative_mass_ratio_vs_raw": _maxover(
                    gpu_recs, lambda r: r["operators"][w]["gpu"]["euler_update"]["negative_mass_ratio_vs_reference"]),
                "median_projection_ms": float(np.median([
                    r["operators"][w]["gpu"]["timing_ms"]["median"] for r in gpu_recs
                    if r["operators"][w]["gpu"].get("timing_ms")] or [float("nan")])),
            }
        out["projected"][w] = blk
    tim = [r["collision_timing_ms"]["median"] for r in snap if r.get("collision_timing_ms")]
    if tim:
        out["timing"]["median_collision_ms_per_point"] = float(np.median(tim))
        out["timing"]["estimated_serial_24_point_collision_ms"] = float(24.0 * np.median(tim))
    return out


def evaluate_gates(quads: list[dict], gates: dict) -> list[dict]:
    """quads: list of {"name", "Nrho", "M", "summary", "role"} in the order
    baseline, proposed, reference (roles taken from the 'role' field)."""
    results = []
    byrole = {q.get("role"): q for q in quads}
    base, prop = byrole.get("baseline"), byrole.get("proposed")

    def add(name, value, threshold, ok, note=""):
        results.append({"gate": name, "value": value, "threshold": threshold,
                        "pass": bool(ok), "note": note})

    for q in quads:
        s = q["summary"]
        for w in ("euclidean", "fplus"):
            v = s["projected"][w]["max_post_cancellation_defect"]
            add(f"G1 post-projection defect [{q['name']}, {w}, numpy]", v, gates["post_defect"],
                v < gates["post_defect"])
            g = s["projected"][w].get("gpu")
            if g:
                add(f"G1 post-projection defect [{q['name']}, {w}, GPU]", g["max_post_cancellation_defect"],
                    gates["post_defect"], g["max_post_cancellation_defect"] < gates["post_defect"])
                add(f"G0 GPU==numpy projected Q [{q['name']}, {w}] (rel Linf)", g["max_abs_diff_vs_numpy_rel_Linf"],
                    gates["gpu_numpy_agreement"], g["max_abs_diff_vs_numpy_rel_Linf"] < gates["gpu_numpy_agreement"])
        for w in ("euclidean", "fplus"):
            v_all = s["projected"][w]["max_rel_correction_l2_all"]
            v_int = s["projected"][w]["max_rel_correction_l2_interior"]
            add(f"G3 ||dQ||/||Q|| < {gates['rel_correction']:g} [{q['name']}, {w}, interior |x|<{INTERIOR_ABS_X}]",
                v_int, gates["rel_correction"], v_int < gates["rel_correction"])
            add(f"G3 ||dQ||/||Q|| < {gates['rel_correction']:g} [{q['name']}, {w}, all points]",
                v_all, gates["rel_correction"], v_all < gates["rel_correction"],
                "near-equilibrium boundary points inflate any Q-relative measure")
            nr = s["projected"][w]["max_negative_mass_ratio_vs_raw"]
            nn = s["projected"][w]["max_newly_negative_vs_raw"]
            mr = s["projected"][w]["max_min_ratio_vs_raw"]
            add(f"G4 neg-mass(f+dt Qc)/neg-mass(f+dt Q) <= {gates['negmass_ratio']} [{q['name']}, {w}]",
                nr, gates["negmass_ratio"], nr <= gates["negmass_ratio"])
            add(f"G4 newly negative nodes <= {gates['newly_negative']} [{q['name']}, {w}]",
                nn, gates["newly_negative"], nn <= gates["newly_negative"])
            add(f"G5 min(f+dt Qc)/min(f+dt Q) <= {gates['min_ratio']} [{q['name']}, {w}]",
                mr, gates["min_ratio"], (not math.isnan(mr)) and mr <= gates["min_ratio"])
    if prop is not None:
        s = prop["summary"]["raw"]
        vs = max(s["max_state_defect_all"].values())
        add(f"G2 raw defect [{prop['name']}] state-normalised, all points", vs, gates["raw_proposed"],
            vs < gates["raw_proposed"])
    if base is not None and prop is not None:
        b = base["summary"]["raw"]["max_abs_uz_source_rate"]
        p = prop["summary"]["raw"]["max_abs_uz_source_rate"]
        ratio = b / p if p > 0 else float("inf")
        add(f"G6 artificial u_z source reduced x{gates['z_reduction']:g} [{base['name']} -> {prop['name']}]",
            ratio, gates["z_reduction"], ratio >= gates["z_reduction"])
        bz = base["summary"]["raw"]["max_abs_momentum_z_state_defect"]
        pz = prop["summary"]["raw"]["max_abs_momentum_z_state_defect"]
        ratio2 = bz / pz if pz > 0 else float("inf")
        add(f"G6 momentum_z state defect reduced x{gates['z_reduction']:g} [{base['name']} -> {prop['name']}]",
            ratio2, gates["z_reduction"], ratio2 >= gates["z_reduction"])
    for q in quads:
        t = q["summary"]["timing"]
        if "median_collision_ms_per_point" in t:
            for w in ("euclidean", "fplus"):
                g = q["summary"]["projected"][w].get("gpu")
                if g and not math.isnan(g["median_projection_ms"]):
                    frac = g["median_projection_ms"] / t["median_collision_ms_per_point"]
                    add(f"G7 projection cost <= {gates['projection_cost_fraction']:g} x collision [{q['name']}, {w}]",
                        frac, gates["projection_cost_fraction"], frac <= gates["projection_cost_fraction"])
    return results


DEFAULT_GATES = {
    "post_defect": 1.0e-12, "raw_proposed": 5.0e-5, "rel_correction": 5.0e-3,
    "negmass_ratio": 1.05, "newly_negative": 33, "min_ratio": 1.10, "z_reduction": 100.0,
    "projection_cost_fraction": 0.05, "gpu_numpy_agreement": 1.0e-10,
}


def write_points_csv(path: Path, quads: list[dict]) -> None:
    fields = ["quadrature", "Nrho", "M", "label", "x_nondim", "density", "ux", "uz", "T",
              "min_f", "negative_mass_fraction_f", "collision_median_ms",
              "raw_canc_mass", "raw_canc_momentum_x", "raw_canc_momentum_z", "raw_canc_energy",
              "raw_state_mass", "raw_state_momentum_x", "raw_state_momentum_z", "raw_state_energy",
              "raw_uz_source_rate", "raw_L2", "raw_negmass_update"]
    for w in WEIGHTINGS:
        fields += [f"{w}_post_canc_max", f"{w}_relL2", f"{w}_negmass_ratio", f"{w}_newly_neg",
                   f"{w}_min_ratio", f"{w}_dt_maxBl", f"{w}_gpu_ms", f"{w}_gpu_post_canc_max",
                   f"{w}_gpu_vs_numpy"]
    with Path(path).open("w", newline="") as stream:
        wr = csv.DictWriter(stream, fieldnames=fields)
        wr.writeheader()
        for q in quads:
            for r in q["records"]:
                op = r["operators"]
                row = {"quadrature": q["name"], "Nrho": q["Nrho"], "M": q["M"], "label": r["label"],
                       "x_nondim": r["x_nondim"], "density": r["input"]["density"],
                       "ux": r["input"]["velocity"][0], "uz": r["input"]["velocity"][2],
                       "T": r["input"]["temperature"], "min_f": r["input"]["min"],
                       "negative_mass_fraction_f": r["input"]["negative_mass_fraction"],
                       "collision_median_ms": (r["collision_timing_ms"] or {}).get("median"),
                       "raw_canc_mass": op["raw"]["cancellation_defect"]["mass"],
                       "raw_canc_momentum_x": op["raw"]["cancellation_defect"]["momentum_x"],
                       "raw_canc_momentum_z": op["raw"]["cancellation_defect"]["momentum_z"],
                       "raw_canc_energy": op["raw"]["cancellation_defect"]["energy"],
                       "raw_state_mass": op["raw"]["state_defect"]["mass"],
                       "raw_state_momentum_x": op["raw"]["state_defect"]["momentum_x"],
                       "raw_state_momentum_z": op["raw"]["state_defect"]["momentum_z"],
                       "raw_state_energy": op["raw"]["state_defect"]["energy"],
                       "raw_uz_source_rate": op["raw"]["uz_source_rate"], "raw_L2": op["raw"]["L2"],
                       "raw_negmass_update": op["raw"]["euler_update"]["negative_mass_fraction"]}
                for w in WEIGHTINGS:
                    b = op.get(w, {})
                    if "error" in b or not b:
                        continue
                    eu = b["euler_update"]
                    row.update({f"{w}_post_canc_max": max(b["cancellation_defect"].values()),
                                f"{w}_relL2": b["rel_correction_l2"],
                                f"{w}_negmass_ratio": eu["negative_mass_ratio_vs_reference"],
                                f"{w}_newly_neg": eu["newly_negative_vs_reference"],
                                f"{w}_min_ratio": eu["min_ratio_vs_reference"],
                                f"{w}_dt_maxBl": b["dt_max_abs_basis_lam"]})
                    if "gpu" in b:
                        g = b["gpu"]
                        row.update({f"{w}_gpu_ms": (g["timing_ms"] or {}).get("median"),
                                    f"{w}_gpu_post_canc_max": max(g["cancellation_defect"].values()),
                                    f"{w}_gpu_vs_numpy": g["max_abs_diff_vs_numpy"]})
                wr.writerow(row)


def print_summary_table(quads: list[dict], gate_results: list[dict]) -> None:
    print("\n=== per-quadrature summary ===")
    hdr = f"{'quadrature':14s} {'rawCanc(all)':>12s} {'rawCanc(int)':>12s} {'rawState(all)':>13s} {'|du_z/dt|max':>12s} {'coll ms/pt':>10s}"
    print(hdr)
    for q in quads:
        s = q["summary"]; r = s["raw"]
        t = s["timing"].get("median_collision_ms_per_point", float("nan"))
        print(f"{q['name']:14s} {max(r['max_cancellation_defect_all'].values()):12.3e} "
              f"{max(r['max_cancellation_defect_interior'].values()):12.3e} "
              f"{max(r['max_state_defect_all'].values()):13.3e} {r['max_abs_uz_source_rate']:12.3e} {t:10.3f}")
    print(f"\n{'quadrature':14s} {'weighting':10s} {'post(np)':>9s} {'post(gpu)':>9s} {'relL2 int':>9s} {'relL2 all':>9s} "
          f"{'negratio':>8s} {'newneg':>6s} {'minratio':>8s} {'gpu ms':>7s} {'gpu-np':>8s}")
    for q in quads:
        for w in WEIGHTINGS:
            b = q["summary"]["projected"][w]
            g = b.get("gpu", {})
            print(f"{q['name']:14s} {w:10s} {b['max_post_cancellation_defect']:9.1e} "
                  f"{g.get('max_post_cancellation_defect', float('nan')):9.1e} "
                  f"{b['max_rel_correction_l2_interior']:9.2e} {b['max_rel_correction_l2_all']:9.2e} "
                  f"{b['max_negative_mass_ratio_vs_raw']:8.3f} {b['max_newly_negative_vs_raw']:6.0f} "
                  f"{b['max_min_ratio_vs_raw']:8.3f} {g.get('median_projection_ms', float('nan')):7.3f} "
                  f"{g.get('max_abs_diff_vs_numpy', float('nan')):8.1e}")
    print("\n=== gates ===")
    for g in gate_results:
        flag = "PASS" if g["pass"] else "FAIL"
        print(f"[{flag}] {g['gate']}: value={g['value']:.3e} threshold={g['threshold']:.3e} {g['note']}")
    npass = sum(1 for g in gate_results if g["pass"])
    print(f"\nGATES_PASSED={npass}/{len(gate_results)}")
