#!/usr/bin/env python3
"""Five-moment conservative projection for the DGFS fast-spectral collision term.

Notation (uniform velocity grid, weight ``cw`` per node, ``N = Nv**3`` nodes):

    B   = [1, vx, vy, vz, 0.5|v|^2]            (5 x N)   collision invariants
    m   = cw * B @ Q                            (5,)      discrete moments of Q
    G_w = cw * (B * w) @ B.T                    (5 x 5)   weighted Gram matrix
    lam = G_w^{-1} m
    Q_c = Q - w * (B.T @ lam)                   minimises sum_j cw delta_j^2 / w_j
                                                subject to cw * B @ Q_c = 0
Weightings
    euclidean   w = 1                 (plain minimum-L2 correction)
    f           w = f                 (signed-f diagnostic; not safe when f < 0)
    fplus       w = max(f, 0)         (non-negative weighted correction)
    maxwellian  w = M[f]              (local Maxwellian with the moments of f)

Every solve uses the diagonally scaled Gram matrix D G_w D (unit diagonal); on the
Nv=32, [-7,7]^3 grid this brings cond2 of the unweighted Gram matrix from ~3.6e3
to ~17.

The numpy part is the reference.  ``GPUConservativeProjector`` implements
``euclidean``, ``f`` and ``fplus`` with pycuda kernels that use exactly the AoSoA
addressing (``upt*ldim + SOA_IX(elem, idx, ncola)``) of the solver's ``vhs-gll``
kernels, so it can be called right after ``scattering.fs(...)`` on the same
device matrices (this is what the solver hook in ``system.py`` does).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

INVARIANT_NAMES = ("mass", "momentum_x", "momentum_y", "momentum_z", "energy")
WEIGHTINGS = ("euclidean", "f", "fplus", "maxwellian")
GPU_WEIGHTINGS = ("euclidean", "f", "fplus")


# --------------------------------------------------------------------------- #
# numpy reference
# --------------------------------------------------------------------------- #
def invariant_basis(cv: np.ndarray) -> np.ndarray:
    cv = np.asarray(cv, dtype=float)
    if cv.ndim != 2 or cv.shape[0] != 3:
        raise ValueError("cv must have shape (3, N)")
    return np.vstack((np.ones(cv.shape[1]), cv, 0.5 * np.sum(cv * cv, axis=0)))


def bulk_moments(f: np.ndarray, cv: np.ndarray, cw: float) -> tuple[float, np.ndarray, float]:
    """(rho, u, T) in the solver's nondimensional units (Maxwellian ~ exp(-c^2/T))."""
    f = np.asarray(f, dtype=float)
    rho = cw * float(np.sum(f))
    u = cw * (cv @ f) / rho
    c = cv - u[:, None]
    T = (2.0 / 3.0) * cw * float(np.dot(np.sum(c * c, axis=0), f)) / rho
    return rho, u, T


def maxwellian(cv: np.ndarray, rho: float, u, T: float) -> np.ndarray:
    c = cv - np.asarray(u, dtype=float)[:, None]
    return rho / (math.pi * T) ** 1.5 * np.exp(-np.sum(c * c, axis=0) / T)


def kinetic_energy_scale(rho: float, u, T: float) -> float:
    u = np.asarray(u, dtype=float)
    return 0.5 * rho * float(np.dot(u, u)) + 0.75 * rho * T


def state_scales(f: np.ndarray, cv: np.ndarray, cw: float) -> np.ndarray:
    """Scales for the *state-normalised* source rate of each invariant:
    mass -> rho; momentum_x -> rho*max(|u_x|, sqrt(T)); transverse momentum ->
    rho*sqrt(T) (thermal momentum); energy -> 0.5 rho u^2 + 0.75 rho T."""
    rho, u, T = bulk_moments(f, cv, cw)
    st = math.sqrt(max(T, 0.0))
    return np.array([rho, rho * max(abs(u[0]), st), rho * st, rho * st,
                     kinetic_energy_scale(rho, u, T)])


@dataclass
class ProjectionResult:
    weighting: str
    Qc: np.ndarray
    lam: np.ndarray
    delta: np.ndarray
    moments_before: np.ndarray
    moments_after: np.ndarray
    gram: np.ndarray
    scaled_condition: float
    max_abs_basis_lam: float          # max_j |(B^T lam)_j|  (positivity indicator)
    weight_min: float
    weight_max: float
    extras: dict = field(default_factory=dict)

    @property
    def rel_correction_l2(self) -> float:
        num = math.sqrt(float(np.dot(self.delta, self.delta)))
        q0 = self.Qc - self.delta
        den = math.sqrt(float(np.dot(q0, q0)))
        return num / max(den, np.finfo(float).tiny)


class ConservativeProjector:
    """numpy implementation (double precision)."""

    def __init__(self, cv: np.ndarray, cw: float):
        self.cv = np.asarray(cv, dtype=float)
        self.cw = float(cw)
        self.N = self.cv.shape[1]
        self.B = invariant_basis(self.cv)
        self.absB = np.abs(self.B)
        self.G_euclid = self.cw * (self.B @ self.B.T)
        self.D_euclid = 1.0 / np.sqrt(np.diag(self.G_euclid))
        self.cond_unscaled = float(np.linalg.cond(self.G_euclid))
        Gs = self.D_euclid[:, None] * self.G_euclid * self.D_euclid[None, :]
        self.cond_scaled = float(np.linalg.cond(Gs))
        self._chol_euclid = np.linalg.cholesky(Gs)

    def moments(self, Q: np.ndarray) -> np.ndarray:
        return self.cw * (self.B @ np.asarray(Q, dtype=float))

    def cancellation_scales(self, Q: np.ndarray) -> np.ndarray:
        return self.cw * (self.absB @ np.abs(np.asarray(Q, dtype=float)))

    def cancellation_defects(self, Q: np.ndarray) -> np.ndarray:
        return np.abs(self.moments(Q)) / np.maximum(self.cancellation_scales(Q),
                                                    np.finfo(float).tiny)

    def state_defects(self, Q: np.ndarray, f: np.ndarray) -> np.ndarray:
        return np.abs(self.moments(Q)) / np.maximum(state_scales(f, self.cv, self.cw),
                                                    np.finfo(float).tiny)

    def weight(self, weighting: str, f: np.ndarray | None) -> np.ndarray:
        if weighting == "euclidean":
            return np.ones(self.N)
        if f is None:
            raise ValueError(f"weighting '{weighting}' needs f")
        f = np.asarray(f, dtype=float)
        if weighting == "f":
            return f.copy()
        if weighting == "fplus":
            return np.maximum(f, 0.0)
        if weighting == "maxwellian":
            rho, u, T = bulk_moments(f, self.cv, self.cw)
            return maxwellian(self.cv, rho, u, T)
        raise ValueError(f"unknown weighting '{weighting}'")

    def project(self, Q: np.ndarray, f: np.ndarray | None = None,
                weighting: str = "euclidean") -> ProjectionResult:
        Q = np.asarray(Q, dtype=float)
        m = self.moments(Q)
        if weighting == "euclidean":
            G, D = self.G_euclid, self.D_euclid
            y = np.linalg.solve(self._chol_euclid, D * m)
            y = np.linalg.solve(self._chol_euclid.T, y)
            lam = D * y
            Bt_lam = self.B.T @ lam
            delta = -Bt_lam
            cond, wmin, wmax = self.cond_scaled, 1.0, 1.0
        else:
            w = self.weight(weighting, f)
            G = self.cw * ((self.B * w[None, :]) @ self.B.T)
            diag = np.diag(G)
            if np.any(diag <= 0.0) or not np.all(np.isfinite(diag)):
                raise FloatingPointError("weighted Gram matrix has a non-positive diagonal")
            D = 1.0 / np.sqrt(diag)
            Gs = D[:, None] * G * D[None, :]
            cond = float(np.linalg.cond(Gs))
            lam = D * np.linalg.solve(Gs, D * m)
            Bt_lam = self.B.T @ lam
            delta = -w * Bt_lam
            wmin, wmax = float(np.min(w)), float(np.max(w))
        Qc = Q + delta
        return ProjectionResult(
            weighting=weighting, Qc=Qc, lam=lam, delta=delta, moments_before=m,
            moments_after=self.moments(Qc), gram=G, scaled_condition=cond,
            max_abs_basis_lam=float(np.max(np.abs(Bt_lam))), weight_min=wmin,
            weight_max=wmax)


# --------------------------------------------------------------------------- #
# positivity / entropy diagnostics
# --------------------------------------------------------------------------- #
def negativity(f: np.ndarray, cw: float) -> dict:
    f = np.asarray(f, dtype=float)
    negmass = cw * float(np.sum(np.maximum(-f, 0.0)))
    posmass = cw * float(np.sum(np.maximum(f, 0.0)))
    return {"min": float(np.min(f)), "max": float(np.max(f)),
            "negative_count": int(np.count_nonzero(f < 0.0)),
            "negative_mass": negmass,
            "negative_mass_fraction": negmass / max(posmass, np.finfo(float).tiny)}


def euler_update_negativity(f: np.ndarray, Q: np.ndarray, dt: float, cw: float,
                            reference_update: np.ndarray | None = None) -> dict:
    """Negativity of f + dt*Q; optionally compared with a reference update."""
    fn = np.asarray(f, dtype=float) + dt * np.asarray(Q, dtype=float)
    out = negativity(fn, cw)
    if reference_update is not None:
        ref = negativity(reference_update, cw)
        out["newly_negative_vs_reference"] = int(
            np.count_nonzero((fn < 0.0) & (reference_update >= 0.0)))
        out["negative_mass_ratio_vs_reference"] = float(
            out["negative_mass"] / max(ref["negative_mass"], np.finfo(float).tiny))
        out["min_ratio_vs_reference"] = (float(out["min"] / ref["min"])
                                         if ref["min"] < 0.0 else float("nan"))
    return out


def entropy_indicator(Q: np.ndarray, f: np.ndarray, cw: float) -> dict:
    """cw*sum_{f>0} Q log f.  Only an indicator (negative tails, spectral
    truncation): not a rigorous discrete H-theorem statement."""
    f = np.asarray(f, dtype=float)
    Q = np.asarray(Q, dtype=float)
    mask = f > 0.0
    return {"Q_log_f_positive_nodes": cw * float(np.dot(Q[mask], np.log(f[mask]))),
            "positive_node_fraction": float(np.mean(mask))}


# --------------------------------------------------------------------------- #
# GPU implementation (pycuda), same AoSoA addressing as the vhs-gll kernels
# --------------------------------------------------------------------------- #
_GPU_SRC = r"""
#define SOA_SZ @SOASZ@
#define SOA_IX(a, v, nv) ((((a) / SOA_SZ)*(nv) + (v))*SOA_SZ + (a) % SOA_SZ)
#define NQ 20
#define VSIZE @VSIZE@
#define NTHREADS @NTHREADS@
#define WMODE @WMODE@   /* 0: euclidean (w=1), 1: w=f, 2: w=max(f,0) */

__device__ __forceinline__ double weight_of(const double fj)
{
#if WMODE == 0
    return 1.0;
#elif WMODE == 1
    return fj;
#else
    return fj > 0.0 ? fj : 0.0;
#endif
}

/* Pass 1: per-block partial sums of
     s[0..4]  = sum_j B_i(v_j) Q_j
     s[5..19] = sum_j w_j B_a B_b, (a<=b) row-major upper triangle:
        (0,0)(0,1)(0,2)(0,3)(0,4)(1,1)(1,2)(1,3)(1,4)(2,2)(2,3)(2,4)(3,3)(3,4)(4,4) */
__global__ void proj_partial
(
    const int nrow, const int ldim, const int ncola, const int ncolb,
    const int elem, const int upt,
    const double* __restrict__ cvx, const double* __restrict__ cvy,
    const double* __restrict__ cvz,
    const double* __restrict__ f, const double* __restrict__ Q,
    double* __restrict__ partial
)
{
    __shared__ double sh[NQ*NTHREADS];
    double s[NQ];
    #pragma unroll
    for (int k = 0; k < NQ; ++k) s[k] = 0.0;

    for (int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < VSIZE;
         idx += gridDim.x*blockDim.x)
    {
        const int idx_s = upt*ldim + SOA_IX(elem, idx, ncola);
        const double vx = cvx[idx], vy = cvy[idx], vz = cvz[idx];
        const double e = 0.5*(vx*vx + vy*vy + vz*vz);
        const double q = Q[idx_s];
        const double w = weight_of(f[idx_s]);
        s[0] += q;      s[1] += vx*q;   s[2] += vy*q;   s[3] += vz*q;   s[4] += e*q;
        s[5]  += w;        s[6]  += w*vx;     s[7]  += w*vy;    s[8]  += w*vz;   s[9]  += w*e;
        s[10] += w*vx*vx;  s[11] += w*vx*vy;  s[12] += w*vx*vz; s[13] += w*vx*e;
        s[14] += w*vy*vy;  s[15] += w*vy*vz;  s[16] += w*vy*e;
        s[17] += w*vz*vz;  s[18] += w*vz*e;
        s[19] += w*e*e;
    }
    #pragma unroll
    for (int k = 0; k < NQ; ++k) sh[k*NTHREADS + threadIdx.x] = s[k];
    __syncthreads();
    for (int stride = NTHREADS/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            #pragma unroll
            for (int k = 0; k < NQ; ++k)
                sh[k*NTHREADS + threadIdx.x] += sh[k*NTHREADS + threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int k = 0; k < NQ; ++k) partial[blockIdx.x*NQ + k] = sh[k*NTHREADS];
    }
}

/* Pass 2 (one block): reduce partials, assemble G (5x5) and m (5), scale by
   D = 1/sqrt(diag G), Gaussian elimination with partial pivoting, write lam[5]
   and the reduced sums[NQ] (already multiplied by cw).                        */
__global__ void proj_solve
(
    const int nblocks, const double cw,
    const double* __restrict__ partial,
    double* __restrict__ lam, double* __restrict__ sums
)
{
    __shared__ double sh[NQ*NTHREADS];
    for (int k = 0; k < NQ; ++k) {
        double acc = 0.0;
        for (int b = threadIdx.x; b < nblocks; b += NTHREADS) acc += partial[b*NQ + k];
        sh[k*NTHREADS + threadIdx.x] = acc;
    }
    __syncthreads();
    for (int stride = NTHREADS/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            for (int k = 0; k < NQ; ++k)
                sh[k*NTHREADS + threadIdx.x] += sh[k*NTHREADS + threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double S[NQ];
        for (int k = 0; k < NQ; ++k) { S[k] = cw*sh[k*NTHREADS]; sums[k] = S[k]; }
        const int ut[5][5] = { {5,6,7,8,9}, {6,10,11,12,13}, {7,11,14,15,16},
                               {8,12,15,17,18}, {9,13,16,18,19} };
        double A[5][6], D[5], y[5];
        for (int i = 0; i < 5; ++i) D[i] = rsqrt(S[ut[i][i]]);
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) A[i][j] = D[i]*S[ut[i][j]]*D[j];
            A[i][5] = D[i]*S[i];
        }
        for (int c = 0; c < 5; ++c) {
            int piv = c; double best = fabs(A[c][c]);
            for (int r = c+1; r < 5; ++r) { const double v = fabs(A[r][c]); if (v > best) { best = v; piv = r; } }
            if (piv != c) { for (int j = 0; j < 6; ++j) { const double t = A[c][j]; A[c][j] = A[piv][j]; A[piv][j] = t; } }
            const double inv = 1.0/A[c][c];
            for (int r = c+1; r < 5; ++r) {
                const double fct = A[r][c]*inv;
                if (fct != 0.0) for (int j = c; j < 6; ++j) A[r][j] -= fct*A[c][j];
            }
        }
        for (int r = 4; r >= 0; --r) {
            double acc = A[r][5];
            for (int j = r+1; j < 5; ++j) acc -= A[r][j]*y[j];
            y[r] = acc/A[r][r];
        }
        for (int i = 0; i < 5; ++i) lam[i] = D[i]*y[i];
    }
}

/* Pass 3: Q_j <- Q_j - w_j (lam0 + lam1 vx + lam2 vy + lam3 vz + lam4 e_j) */
__global__ void proj_apply
(
    const int nrow, const int ldim, const int ncola, const int ncolb,
    const int elem, const int upt,
    const double* __restrict__ cvx, const double* __restrict__ cvy,
    const double* __restrict__ cvz,
    const double* __restrict__ f, double* __restrict__ Q,
    const double* __restrict__ lam
)
{
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < VSIZE) {
        const int idx_s = upt*ldim + SOA_IX(elem, idx, ncola);
        const double vx = cvx[idx], vy = cvy[idx], vz = cvz[idx];
        const double e = 0.5*(vx*vx + vy*vy + vz*vz);
        const double bl = lam[0] + lam[1]*vx + lam[2]*vy + lam[3]*vz + lam[4]*e;
        Q[idx_s] -= weight_of(f[idx_s])*bl;
    }
}
"""

_UT = [[5, 6, 7, 8, 9], [6, 10, 11, 12, 13], [7, 11, 14, 15, 16],
       [8, 12, 15, 17, 18], [9, 13, 16, 18, 19]]


class GPUConservativeProjector:
    """pycuda projector operating in place on the solver's device matrices.

    ``apply(d_arr_in, d_arr_out, elem, upt)`` mirrors ``scattering.fs`` and
    replaces the collision term stored in ``d_arr_out`` at (elem, upt) by its
    five-moment conservative projection.  ``d_arr_in`` holds f (needed for the
    weighted variants; for ``euclidean`` it is only used for addressing).
    """

    def __init__(self, backend, vm, weighting: str = "euclidean", solve: str = "device",
                 nblocks: int = 128, nthreads: int = 256):
        if weighting not in GPU_WEIGHTINGS:
            raise ValueError(f"GPU weighting must be one of {GPU_WEIGHTINGS}")
        if solve not in ("device", "host"):
            raise ValueError("solve must be 'device' or 'host'")
        if np.dtype(backend.fpdtype) != np.dtype(np.float64):
            raise TypeError("phase-3 GPU projection kernels require precision = double")
        if nblocks < 1 or nthreads < 1 or nthreads & (nthreads - 1):
            raise ValueError("nblocks must be positive and nthreads must be a power of two")
        from pycuda import compiler, gpuarray

        self.gpuarray = gpuarray
        self.backend, self.vm = backend, vm
        self.weighting, self.solve = weighting, solve
        self.nblocks, self.nthreads = int(nblocks), int(nthreads)
        self.vsize = int(vm.vsize())
        self.cw = float(vm.cw())
        cv = np.asarray(vm.cv(), dtype=np.float64)
        self.d_cvx = gpuarray.to_gpu(np.ascontiguousarray(cv[0]))
        self.d_cvy = gpuarray.to_gpu(np.ascontiguousarray(cv[1]))
        self.d_cvz = gpuarray.to_gpu(np.ascontiguousarray(cv[2]))
        self.d_partial = gpuarray.zeros(self.nblocks * 20, dtype=np.float64)
        self.d_lam = gpuarray.zeros(5, dtype=np.float64)
        self.d_sums = gpuarray.zeros(20, dtype=np.float64)

        wmode = {"euclidean": 0, "f": 1, "fplus": 2}[weighting]
        src = (_GPU_SRC.replace("@SOASZ@", str(backend.soasz))
               .replace("@VSIZE@", str(self.vsize))
               .replace("@NTHREADS@", str(self.nthreads))
               .replace("@WMODE@", str(wmode)))
        module = compiler.SourceModule(src)
        self.k_partial = module.get_function("proj_partial")
        self.k_partial.prepare("iiiiiiPPPPPP")
        self.k_solve = module.get_function("proj_solve")
        self.k_solve.prepare("idPPP")
        self.k_apply = module.get_function("proj_apply")
        self.k_apply.prepare("iiiiiiPPPPPP")
        self.block = (self.nthreads, 1, 1)
        self.grid_partial = (self.nblocks, 1)
        self.grid_solve = (1, 1)
        self.grid_apply = ((self.vsize + self.nthreads - 1) // self.nthreads, 1)
        self.last_lambda = None
        self.last_sums = None

    def _solve_host(self):
        partial = self.d_partial.get().reshape(self.nblocks, 20)
        S = self.cw * partial.sum(axis=0)
        G = np.array([[S[_UT[i][j]] for j in range(5)] for i in range(5)])
        m = S[:5]
        D = 1.0 / np.sqrt(np.diag(G))
        lam = D * np.linalg.solve(D[:, None] * G * D[None, :], D * m)
        self.d_lam.set(np.ascontiguousarray(lam, dtype=np.float64))
        self.last_lambda, self.last_sums = lam, S

    def apply(self, d_arr_in, d_arr_out, elem: int, upt: int) -> None:
        nrow, ldim, _ = d_arr_in.traits
        ncola, ncolb = d_arr_in.ioshape[1:]
        f_ptr = d_arr_in._as_parameter_
        q_ptr = d_arr_out._as_parameter_
        self.k_partial.prepared_call(
            self.grid_partial, self.block, int(nrow), int(ldim), int(ncola), int(ncolb),
            int(elem), int(upt), self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
            f_ptr, q_ptr, self.d_partial.ptr)
        if self.solve == "device":
            self.k_solve.prepared_call(
                self.grid_solve, self.block, int(self.nblocks), float(self.cw),
                self.d_partial.ptr, self.d_lam.ptr, self.d_sums.ptr)
        else:
            self._solve_host()
        self.k_apply.prepared_call(
            self.grid_apply, self.block, int(nrow), int(ldim), int(ncola), int(ncolb),
            int(elem), int(upt), self.d_cvx.ptr, self.d_cvy.ptr, self.d_cvz.ptr,
            f_ptr, q_ptr, self.d_lam.ptr)

    def fetch_lambda(self) -> np.ndarray:
        """Device -> host copy of the last multipliers (synchronises)."""
        if self.solve == "device":
            self.last_lambda = self.d_lam.get()
            self.last_sums = self.d_sums.get()
        return self.last_lambda
