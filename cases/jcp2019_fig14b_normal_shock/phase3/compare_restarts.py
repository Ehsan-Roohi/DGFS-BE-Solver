"""Compare DGFS restart snapshots against a supplied reference distribution.

The script computes per-point physical moments from the full distributions,
domain inventories, overshoot and negativity diagnostics, final residuals, and
wall times.  Reference and restart times are read from file metadata; no
specific start time is assumed.
"""Compare the phase-3b short restarts (t = 30 -> 30.1): per-point moments computed
directly from the full distributions (rho, u_x, u_z, T, viscous normal stress
P_xx - p, heat flux q_x, negativity), domain inventories (mass, momentum, energy,
H) relative to the t=30 reference, final residuals, and wall time.

    python compare_restarts.py --config dgfs_fig14b.ini --mesh mesh.frfsm \
        --reference dist_dgfs_fig14b-30.0.frfss --runs run_M6_raw run_M6_fw run_M16_raw run_M16_fw
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

R_UNIVERSAL = 8.3144598
GLL2_WEIGHTS = np.array([1.0, 4.0, 1.0]) / 6.0


def read_ini(path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    cfg.optionxform = str
    with path.open() as stream:
        cfg.read_file(stream)
    return cfg


def velocity_mesh(cfg):
    nv = cfg.getint("constants", "Nv")
    t0 = cfg.getfloat("non-dim", "T0")
    mm = cfg.getfloat("non-dim", "molarMass0")
    rho0 = cfg.getfloat("non-dim", "rho0")
    u0 = math.sqrt(2.0 * R_UNIVERSAL * t0 / mm)
    cmax = cfg.getfloat("velocity-mesh", "cmax") / u0
    tmax = cfg.getfloat("velocity-mesh", "Tmax") / t0
    dev = cfg.getfloat("velocity-mesh", "dev")
    L = cmax + dev * math.sqrt(tmax)
    c0 = np.linspace(-L + L / nv, L - L / nv, nv)
    cx, cy, cz = np.meshgrid(c0, c0, c0, indexing="ij")
    cv = np.vstack((cx.ravel(), cy.ravel(), cz.ravel()))
    cw = (2.0 * L / nv) ** 3
    return cv, cw, {"u0": u0, "T0": t0, "rho0": rho0, "L": L}


def load_mesh(path: Path):
    import h5py

    with h5py.File(path, "r") as h5:
        mesh = h5["spt_line_p0"][()]
    left = mesh[:, :, 0].min(axis=0)
    right = mesh[:, :, 0].max(axis=0)
    order = np.argsort(left)
    left, right = left[order], right[order]
    x = np.vstack((left, 0.5 * (left + right), right))   # (3, nelem)
    return x, order, left, right


def read_dist(path: Path, order):
    import h5py

    with h5py.File(path, "r") as h5:
        soln = h5["soln_line_p0"][()]
        stats = h5["stats"][()].decode() if "stats" in h5 else ""
    return soln[:, :, order], stats


def stats_value(stats: str, key: str) -> float | None:
    m = re.search(rf"^{key}\s*=\s*(\S+)", stats, flags=re.M)
    return float(m.group(1)) if m else None


def point_moments(f: np.ndarray, cv: np.ndarray, cw: float) -> dict:
    rho = cw * np.sum(f)
    u = cw * (cv @ f) / rho
    c = cv - u[:, None]
    c2 = np.sum(c * c, axis=0)
    T = (2.0 / 3.0) * cw * np.dot(c2, f) / rho
    pxx = cw * np.dot(c[0] * c[0], f)          # P_xx (nondim, = rho T/2 in equilibrium)
    p = rho * T / 2.0
    qx = 0.5 * cw * np.dot(c2 * c[0], f)
    neg = np.maximum(-f, 0.0)
    return {"rho": float(rho), "ux": float(u[0]), "uy": float(u[1]), "uz": float(u[2]),
            "T": float(T), "Pxx_minus_p": float(pxx - p), "qx": float(qx),
            "min_f": float(np.min(f)), "negative_count": int(np.count_nonzero(f < 0)),
            "negative_mass_fraction": float(cw * np.sum(neg) / max(cw * np.sum(np.maximum(f, 0)), 1e-300))}


def inventories(soln: np.ndarray, xw: np.ndarray, cv: np.ndarray, cw: float) -> dict:
    wx = xw[:, None, :]
    mass_by_v = np.sum(soln * wx, axis=(0, 2))
    mass = cw * np.sum(mass_by_v)
    mom = cw * (cv @ mass_by_v)
    energy = 0.5 * cw * np.dot(np.sum(cv * cv, axis=0), mass_by_v)
    mask = soln > 0
    hint = np.zeros_like(soln)
    hint[mask] = soln[mask] * np.log(soln[mask])
    H = cw * np.sum(hint * wx)
    return {"mass": float(mass), "momentum": [float(v) for v in mom], "energy": float(energy), "H": float(H)}


def shock_position(x: np.ndarray, rho: np.ndarray, mid: float) -> float | None:
    xs = x.T.ravel(); rs = rho.T.ravel()      # element-major ordering
    for i in range(len(xs) - 1):
        if (rs[i] - mid) * (rs[i + 1] - mid) <= 0 and rs[i + 1] != rs[i]:
            return float(xs[i] + (mid - rs[i]) * (xs[i + 1] - xs[i]) / (rs[i + 1] - rs[i]))
    return None


def last_regular_residual(path: Path, dt: float, tstart: float):
    if not path.is_file():
        return None
    prev = None; best = None
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                t = float(row["t"]); raw = float(row["f"]); nz = float(row.get("f_normalized", "nan"))
            except (KeyError, TypeError, ValueError):
                continue
            step = t - prev if prev is not None else math.nan
            prev = t
            if t > tstart and 0.5 * dt <= step <= 1.5 * dt and math.isfinite(raw):
                best = {"t": t, "raw": raw, "normalized": nz}
    return best


def analyse_snapshot(path: Path, cv, cw, x, xw, order, nd):
    soln, stats = read_dist(path, order)
    pts = [[point_moments(soln[u, :, e], cv, cw) for e in range(soln.shape[2])] for u in range(soln.shape[0])]
    rho = np.array([[pts[u][e]["rho"] for e in range(soln.shape[2])] for u in range(3)])
    keys = ("rho", "ux", "uy", "uz", "T", "Pxx_minus_p", "qx", "min_f")
    profiles = {k: np.array([[p[k] for p in row] for row in pts]) for k in keys}
    extrema = {k: {"minimum": float(np.min(v)), "maximum": float(np.max(v))}
               for k, v in profiles.items()}
    monotone_overshoot = {}
    for k in ("rho", "ux", "T"):
        v = profiles[k]
        left, right = float(v[0, 0]), float(v[-1, -1])
        lo, hi = min(left, right), max(left, right)
        jump = max(hi - lo, np.finfo(float).tiny)
        below, above = max(lo - float(np.min(v)), 0.0), max(float(np.max(v)) - hi, 0.0)
        monotone_overshoot[k] = {
            "left": left, "right": right, "below": below, "above": above,
            "fraction_of_jump": max(below, above) / jump,
        }
    cell_averages = {k: np.sum(GLL2_WEIGHTS[:, None] * v, axis=0).tolist()
                     for k, v in profiles.items()}
    return {"file": str(path), "points": pts, "inventories": inventories(soln, xw, cv, cw),
            "tcurr": stats_value(stats, "tcurr"), "wall_time": stats_value(stats, "wall-time"),
            "nsteps": stats_value(stats, "nsteps"),
            "extrema": extrema, "monotone_overshoot": monotone_overshoot,
            "cell_averages": cell_averages,
            "max_abs_uz_nondim": float(max(abs(p["uz"]) for row in pts for p in row)),
            "max_abs_uz_m_per_s": float(nd["u0"] * max(abs(p["uz"]) for row in pts for p in row)),
            "min_f": float(min(p["min_f"] for row in pts for p in row)),
            "max_negative_mass_fraction": float(max(p["negative_mass_fraction"] for row in pts for p in row)),
            "shock_position_nondim": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True, help="reference full distribution")
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--tend", type=float, default=30.1)
    ap.add_argument("--tstart", type=float, default=30.0)
    ap.add_argument("--output-json", type=Path, default=Path("p3b_comparison.json"))
    ap.add_argument("--output-csv", type=Path, default=Path("p3b_comparison.csv"))
    ap.add_argument("--output-md", type=Path, default=Path("p3b_comparison.md"))
    ap.add_argument("--output-figure", type=Path, default=Path("p3b_comparison.png"))
    ap.add_argument("--output-cell-figure", type=Path, default=Path("p3b_cell_averages.png"))
    args = ap.parse_args()

    cfg = read_ini(args.config)
    cv, cw, nd = velocity_mesh(cfg)
    dt = cfg.getfloat("solver-time-integrator", "dt")
    rho_mid = 0.5 * (cfg.getfloat("soln-bcs-left", "rho") + cfg.getfloat("soln-bcs-right", "rho")) / nd["rho0"]
    x, order, left, right = load_mesh(args.mesh)
    xw = GLL2_WEIGHTS[:, None] * (right - left)[None, :]

    def analyse(path):
        res = analyse_snapshot(path, cv, cw, x, xw, order, nd)
        rho = np.array([[p["rho"] for p in row] for row in res["points"]])
        res["shock_position_nondim"] = shock_position(x, rho, rho_mid)
        return res

    ref = analyse(args.reference)
    runs = []
    for d in args.runs:
        cands = sorted(d.glob("dist_p3b_*.frfss"),
                       key=lambda p: float(re.search(r"-([0-9.]+)\.frfss$", p.name).group(1)))
        if not cands:
            print(f"[{d}] no dist_p3b_*.frfss found, skipping"); continue
        final = cands[-1]
        inis = sorted(d.glob("p3b_*.ini"))
        rc = read_ini(inis[0]) if inis else None
        res = analyse(final)
        res["run"] = d.name
        res["Nrho"] = rc.getint("constants", "Nrho") if rc else None
        res["M"] = rc.getint("spherical-design-rule", "M") if rc else None
        res["projection"] = rc.get("scattering-model", "projection", fallback="none") if rc else None
        res["residual_final"] = last_regular_residual(d / "kinetic_residual_p3b.csv", dt, args.tstart)
        wt = d / "WALLTIME.txt"
        res["wall_seconds_job"] = (float(wt.read_text().split("=")[1]) if wt.is_file() else None)
        inv, rinv = res["inventories"], ref["inventories"]
        res["inventory_change_vs_reference"] = {
            "mass": inv["mass"] - rinv["mass"],
            "momentum": [a - b for a, b in zip(inv["momentum"], rinv["momentum"])],
            "energy": inv["energy"] - rinv["energy"], "H": inv["H"] - rinv["H"],
            "mass_relative": (inv["mass"] - rinv["mass"]) / rinv["mass"],
            "energy_relative": (inv["energy"] - rinv["energy"]) / rinv["energy"]}
        runs.append(res)
    if not runs:
        raise SystemExit("no runs analysed")

    # Profile differences versus the legacy M6 run and the M24 raw benchmark.
    keys = ("rho", "ux", "uz", "T", "Pxx_minus_p", "qx")
    def profile(res, k):
        return np.array([[p[k] for p in row] for row in res["points"]])
    base = runs[0]
    benchmark = next((r for r in runs if r.get("M") == 24 and r.get("projection") == "none"), runs[-1])
    for r in runs:
        r["profile_rms_diff_vs_first_run"] = {k: float(np.sqrt(np.mean((profile(r, k) - profile(base, k)) ** 2)))
                                              for k in keys}
        r["profile_rms_change_vs_reference"] = {k: float(np.sqrt(np.mean((profile(r, k) - profile(ref, k)) ** 2)))
                                                for k in keys}
        r["profile_rms_diff_vs_M24_raw"] = {k: float(np.sqrt(np.mean((profile(r, k) - profile(benchmark, k)) ** 2)))
                                            for k in keys}

    # outputs
    report = {"reference": ref, "runs": runs, "benchmark_run": benchmark["run"],
              "nondim": nd, "dt": dt, "tend": args.tend}
    args.output_json.write_text(json.dumps(report, indent=1) + "\n")
    fields = ["run", "Nrho", "M", "projection", "tcurr", "nsteps", "wall_time_stats", "wall_seconds_job",
              "resid_raw", "resid_normalized", "d_mass_rel", "d_mom_x", "d_mom_z", "d_energy_rel", "d_H",
              "max_abs_uz_m_per_s", "min_f", "max_negmass_frac", "shock_position_mm",
              "rho_overshoot", "ux_overshoot", "T_overshoot",
              "rms_rho_vs_M24", "rms_ux_vs_M24", "rms_T_vs_M24",
              "rms_qx_vs_M24", "rms_stress_vs_M24", "rms_uz_vs_M24"]
    H0 = cfg.getfloat("non-dim", "H0")
    with args.output_csv.open("w", newline="") as stream:
        wr = csv.DictWriter(stream, fieldnames=fields); wr.writeheader()
        for r in runs:
            rf = r["residual_final"] or {}
            ch = r["inventory_change_vs_reference"]
            wr.writerow({"run": r["run"], "Nrho": r["Nrho"], "M": r["M"], "projection": r["projection"],
                         "tcurr": r["tcurr"], "nsteps": r["nsteps"], "wall_time_stats": r["wall_time"],
                         "wall_seconds_job": r["wall_seconds_job"], "resid_raw": rf.get("raw"),
                         "resid_normalized": rf.get("normalized"), "d_mass_rel": ch["mass_relative"],
                         "d_mom_x": ch["momentum"][0], "d_mom_z": ch["momentum"][2],
                         "d_energy_rel": ch["energy_relative"], "d_H": ch["H"],
                         "max_abs_uz_m_per_s": r["max_abs_uz_m_per_s"], "min_f": r["min_f"],
                         "max_negmass_frac": r["max_negative_mass_fraction"],
                         "shock_position_mm": (r["shock_position_nondim"] * H0 * 1e3
                                               if r["shock_position_nondim"] is not None else None),
                         "rho_overshoot": r["monotone_overshoot"]["rho"]["fraction_of_jump"],
                         "ux_overshoot": r["monotone_overshoot"]["ux"]["fraction_of_jump"],
                         "T_overshoot": r["monotone_overshoot"]["T"]["fraction_of_jump"],
                         "rms_rho_vs_M24": r["profile_rms_diff_vs_M24_raw"]["rho"],
                         "rms_ux_vs_M24": r["profile_rms_diff_vs_M24_raw"]["ux"],
                         "rms_T_vs_M24": r["profile_rms_diff_vs_M24_raw"]["T"],
                         "rms_qx_vs_M24": r["profile_rms_diff_vs_M24_raw"]["qx"],
                         "rms_stress_vs_M24": r["profile_rms_diff_vs_M24_raw"]["Pxx_minus_p"],
                         "rms_uz_vs_M24": r["profile_rms_diff_vs_M24_raw"]["uz"]})
    lines = ["| run | Nrho | M | proj | steps | wall s | resid raw | resid norm | dM/M | dP_x | dP_z | dE/E | dH | max|u_z| m/s | min f | negmass | x_shock mm |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ref_time = ref.get("tcurr")
    ref_label = f"reference t={ref_time:g}" if ref_time is not None else "reference"
    lines.append(f"| {ref_label} | | | | | | | | | | | | | {ref['max_abs_uz_m_per_s']:.3e} | {ref['min_f']:.2e} | "
                 f"{ref['max_negative_mass_fraction']:.2e} | {ref['shock_position_nondim']*H0*1e3 if ref['shock_position_nondim'] is not None else float('nan'):.4f} |")
    for r in runs:
        rf = r["residual_final"] or {}; ch = r["inventory_change_vs_reference"]
        sp = r["shock_position_nondim"]
        lines.append(f"| {r['run']} | {r['Nrho']} | {r['M']} | {r['projection']} | {r['nsteps']} | "
                     f"{r['wall_seconds_job'] if r['wall_seconds_job'] is not None else r['wall_time']} | "
                     f"{rf.get('raw', float('nan')):.3e} | {rf.get('normalized', float('nan')):.3e} | "
                     f"{ch['mass_relative']:+.2e} | {ch['momentum'][0]:+.2e} | {ch['momentum'][2]:+.2e} | "
                     f"{ch['energy_relative']:+.2e} | {ch['H']:+.2e} | {r['max_abs_uz_m_per_s']:.3e} | "
                     f"{r['min_f']:.2e} | {r['max_negative_mass_fraction']:.2e} | "
                     f"{sp*H0*1e3 if sp is not None else float('nan'):.4f} |")
    args.output_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        xs = (x.T.ravel()) * H0 * 1e3
        panels = [("uz", "u_z [m/s]", nd["u0"]), ("rho", "rho [kg/m^3]", nd["rho0"]), ("T", "T [K]", nd["T0"]),
                  ("qx", "q_x [nondim]", 1.0), ("Pxx_minus_p", "P_xx - p [nondim]", 1.0), ("min_f", "min f", 1.0)]
        for ax, (k, lab, scale) in zip(axes.flat, panels):
            ax.plot(xs, profile(ref, k).T.ravel() * scale, "k--", lw=1, label="t=30 reference")
            for r in runs:
                ax.plot(xs, profile(r, k).T.ravel() * scale, marker="o", ms=3, lw=1, label=r["run"])
            ax.set(xlabel="x [mm]", ylabel=lab); ax.grid(alpha=0.3)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"Phase-3b restarts t={args.tstart} -> {args.tend}")
        fig.savefig(args.output_figure, dpi=160)
        print(f"figure: {args.output_figure}")

        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        xc = 0.5 * (left + right) * H0 * 1e3
        cell_panels = [("rho", "cell-averaged density", nd["rho0"]),
                       ("ux", "cell-averaged u_x [m/s]", nd["u0"]),
                       ("T", "cell-averaged T [K]", nd["T0"]),
                       ("qx", "cell-averaged q_x [nondim]", 1.0),
                       ("Pxx_minus_p", "cell-averaged P_xx-p", 1.0),
                       ("uz", "cell-averaged u_z [m/s]", nd["u0"])]
        for ax, (k, lab, scale) in zip(axes2.flat, cell_panels):
            ax.plot(xc, np.asarray(ref["cell_averages"][k]) * scale, "k--", lw=1.5,
                    label="t=30 reference")
            for r in runs:
                ax.plot(xc, np.asarray(r["cell_averages"][k]) * scale,
                        marker="o", ms=3, lw=1.2, label=r["run"])
            ax.set(xlabel="x [mm]", ylabel=lab); ax.grid(alpha=0.3)
        axes2.flat[0].legend(fontsize=8, loc="best")
        fig2.savefig(args.output_cell_figure, dpi=180)
        print(f"cell-average figure: {args.output_cell_figure}")
    except Exception as exc:  # matplotlib missing on the cluster env is fine
        print(f"figure skipped: {exc}")


if __name__ == "__main__":
    main()
