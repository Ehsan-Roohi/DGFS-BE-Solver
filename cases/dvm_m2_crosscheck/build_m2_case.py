#!/usr/bin/env python3
"""Build a DGFS Mach-2 case from the certified BGK-DVM reference contract."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

EXPECTED = {
    "rho1": 1.0, "u1": 2.581988897471611, "T1": 1.0,
    "rho2": 2.285714285714286, "u2": 1.1296201426438297,
    "T2": 2.0781249999999996, "gamma": 5.0/3.0, "M1": 2.0,
}


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=2e-7, abs_tol=2e-8)


def mesh_text(nelem):
    xs = np.linspace(-0.5, 0.5, nelem + 1)
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "3",
             '0 1 "left"', '0 2 "right"', '1 3 "fluid"', "$EndPhysicalNames",
             "$Nodes", str(nelem + 1)]
    lines += [f"{i + 1} {x:.17g} 0 0" for i, x in enumerate(xs)]
    lines += ["$EndNodes", "$Elements", str(nelem + 2),
              "1 15 2 1 1 1", f"2 15 2 2 2 {nelem + 1}"]
    lines += [f"{i + 3} 1 2 3 3 {i + 1} {i + 2}" for i in range(nelem)]
    lines += ["$EndElements", ""]
    return "\n".join(lines)


def ini_text(s, h0, rho1, t1, u1, rho2, t2, u2):
    return f"""[backend]
precision = double
rank-allocator = linear

[constants]
Nv = 32
NvBatchSize = 64
Nrho = 24

[non-dim]
T0 = {t1:.17g}
H0 = {h0:.17g}
rho0 = {rho1:.17g}
molarMass0 = 4.0047236985e-3

[velocity-mesh]
dev = 12
cmax = 0
Tmax = {t1:.17g}

[spherical-design-rule]
ssrule = womersley
M = 24

[scattering-model]
type = vhs-gll
omega = 0.5
dRef = 2.17e-10
Tref = 273
projection = none
projection-solve = device

[solver]
system = dgfs
order = 2

[solver-time-integrator]
scheme = dgfs-tvd-rk2
controller = none
tstart = 0
tend = 0.25
dt = 0.001
dt-min = 1e-15

[solver-interfaces]
riemann-solver = rusanov
ldg-beta = 0.5
ldg-tau = 0.1

[solver-interfaces-line]
flux-pts = gauss-legendre-lobatto

[solver-interfaces-point]
flux-pts = gauss-legendre

[solver-elements-line]
soln-pts = gauss-legendre-lobatto

[soln-plugin-nancheck]
nsteps = 25

[soln-plugin-dgfsresidualstd]
nsteps = 1
output-file = 1
file = kinetic_residual_m2.csv
normalise = true

[soln-plugin-dgfsdistwriterstd]
dt-out = 0.25
basedir = .
basename = dist_dgfs_M2-{{t:.2f}}

[soln-plugin-dgfsmomwriterstd]
dt-out = 0.25
basedir = .
basename = bulksol_dgfs_M2-{{t:.2f}}

[soln-ics]
type = maxwellianExpr
rho = {rho1:.17g} + ({rho2:.17g}-{rho1:.17g})*(x + 0.5)
T = {t1:.17g} + ({t2:.17g}-{t1:.17g})*(x + 0.5)
ux = {u1:.17g} + ({u2:.17g}-{u1:.17g})*(x + 0.5)
uy = 0
uz = 0

[soln-bcs-left]
type = dgfs-inlet-normalshock
rho = {rho1:.17g}
T = {t1:.17g}
ux = {u1:.17g}
uy = 0
uz = 0

[soln-bcs-right]
type = dgfs-inlet-normalshock
rho = {rho2:.17g}
T = {t2:.17g}
ux = {u2:.17g}
uy = 0
uz = 0
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dvm", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--elements", type=int, default=32)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.dvm, allow_pickle=True)
    required = {"x_mfp", "f", "v", "w", "rho", "ux", "T", "qx", "sig", "states"}
    missing = sorted(required - set(z.files))
    if missing: raise SystemExit("P6_DVM_KEYS_MISSING=" + ",".join(missing))
    s = dict(z["states"].item())
    bad = {k: (s.get(k), v) for k, v in EXPECTED.items() if k not in s or not close(s[k], v)}
    if bad: raise SystemExit("P6_DVM_STATE_MISMATCH=" + repr(bad))

    molar = 4.0047236985e-3
    rgas = 8.31446261815324/molar
    rho1, t1 = 1.916e-5, 223.0
    vscale = math.sqrt(rgas*t1)
    rho2, t2 = rho1*s["rho2"], t1*s["T2"]
    u1, u2 = vscale*s["u1"], vscale*s["u2"]
    old_kn, old_h0 = 0.055299416348671006, 0.03
    lambda1 = old_kn*old_h0
    h0 = 80.0*lambda1

    (out/"mesh_M2_32elem.msh").write_text(mesh_text(args.elements))
    (out/"dgfs_M2.ini").write_text(ini_text(s, h0, rho1, t1, u1, rho2, t2, u2))

    x = np.asarray(z["x_mfp"], dtype=float)*lambda1*1e3
    scales = {"rho": rho1, "ux": vscale, "T": t1,
              "qx": rho1*vscale**3, "sig": rho1*vscale**2}
    with open(out/"dvm_M2_reference.csv", "w", newline="") as stream:
        w = csv.writer(stream); w.writerow(["x_mm","rho_kg_m3","ux_m_s","T_K","qx_W_m2","Pxx_minus_p_Pa"])
        for vals in zip(x, z["rho"]*scales["rho"], z["ux"]*scales["ux"],
                        z["T"]*scales["T"], z["qx"]*scales["qx"], z["sig"]*scales["sig"]):
            w.writerow([f"{float(v):.17g}" for v in vals])
    digest = hashlib.sha256(Path(args.dvm).read_bytes()).hexdigest()
    contract = {"case":"steady_normal_shock", "mach":2.0, "gamma":5/3,
                "dvm_collision":"BGK", "dgfs_collision":"full Boltzmann VHS",
                "comparison_class":"cross-model, not same-operator validation",
                "dvm_path":str(Path(args.dvm).resolve()), "dvm_sha256":digest,
                "dvm_grid":{"nx":len(z["x"]),"nv":len(z["v"]),"x_mfp":[float(x) for x in (np.min(z["x_mfp"]),np.max(z["x_mfp"]))]},
                "dgfs_grid":{"elements":args.elements,"order":2,"Nrho":24,"Momega":24,"dev":12},
                "physical":{"rho1":rho1,"T1":t1,"u1":u1,"rho2":rho2,"T2":t2,"u2":u2,"lambda1_m":lambda1,"H0_m":h0}}
    (out/"P6_CASE_CONTRACT.json").write_text(json.dumps(contract, indent=2)+"\n")
    (out/"P6_CASE_VERIFIED").touch()
    print("P6_DVM_M2_CONTRACT_VERIFIED")
    print(f"P6_H0_M={h0:.17g}")
    print(f"P6_DGFS_U1={u1:.12g}")
    print(f"P6_DGFS_U2={u2:.12g}")

if __name__ == "__main__": main()
