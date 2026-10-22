"""Surrogate and reduced-order models of the lap simulation, and global sensitivity (Sobol) on the simulator itself.

Inputs x = (CdA, ClA, mu, P_kW, mass) over the ranges in RANGES. Outputs: lap time (scalar) and the speed trace v(s).
  * Gaussian process with ARD Matern-5/2 kernel on lap time.
  * Polynomial chaos (Legendre, total degree p, least squares); Sobol indices follow analytically from the coefficients.
  * POD reduced-order model of the speed trace: v(s) ~ mean + sum_k a_k(x) phi_k(s), with a_k regressed by polynomial chaos.
  * Sobol indices by Saltelli sampling on the simulator (SALib), with bootstrap intervals.
Honest note: the simulator takes ~0.3 ms per lap, so the surrogates give no practical speed-up here. They are built and scored to
show the workflow, and the accuracy numbers are what carry over to a model that is expensive.
"""
from __future__ import annotations

import itertools
import time

import jax.numpy as jnp
import numpy as np
from numpy.polynomial import legendre
from scipy.stats import qmc

from . import physics as P
from .circuits import Circuit

NAMES = ["CdA", "ClA", "mu", "P_kW", "mass"]
RANGES = np.array([[0.9, 1.9], [2.0, 7.0], [1.2, 2.6], [400.0, 750.0], [800.0, 860.0]])


def to_unit(x):          # physical -> [-1, 1]
    return 2 * (x - RANGES[:, 0]) / (RANGES[:, 1] - RANGES[:, 0]) - 1


def to_phys(u):
    return RANGES[:, 0] + (u + 1) / 2 * (RANGES[:, 1] - RANGES[:, 0])


def lhs(n, seed=0):
    return to_phys(2 * qmc.LatinHypercube(d=5, seed=seed).random(n) - 1)


def simulate_many(c: Circuit, X: np.ndarray):
    """Direct simulation. Returns speed traces (n, N) and lap times (n,)."""
    dr = jnp.asarray((c.drs_frac[0] > 0.5).astype(float))
    kap = jnp.asarray(c.track.kappa)
    f = P.jax.jit(P.jax.vmap(lambda th: P._simulate(kap, c.track.ds, th[0], th[1], th[2], th[3], th[4], dr)))
    v, t = f(jnp.asarray(X))
    return np.asarray(v), np.asarray(t)


# ---- polynomial chaos -------------------------------------------------------------------------------------------------
def multi_indices(d, p):
    return np.array([a for a in itertools.product(range(p + 1), repeat=d) if sum(a) <= p])


class PCE:
    def __init__(self, p=4, ridge=1e-8):
        self.p, self.ridge = p, ridge
        self.alpha = multi_indices(5, p)

    def _basis(self, U):
        # orthonormal Legendre on [-1,1] under the uniform measure: sqrt(2k+1) P_k
        V = [np.stack([np.sqrt(2 * k + 1) * legendre.legval(U[:, j], [0] * k + [1]) for k in range(self.p + 1)], 1) for j in range(5)]
        return np.prod([V[j][:, self.alpha[:, j]] for j in range(5)], axis=0)

    def fit(self, X, Y):
        B = self._basis(to_unit(X))
        A = B.T @ B + self.ridge * np.eye(B.shape[1])
        self.coef = np.linalg.solve(A, B.T @ Y)
        return self

    def predict(self, X):
        return self._basis(to_unit(X)) @ self.coef

    def sobol(self):
        """First-order and total Sobol indices of a scalar-output PCE (coef shape (m,) or (m,1))."""
        c = np.asarray(self.coef).reshape(len(self.alpha), -1)[:, 0]
        var = np.sum(c[1:] ** 2)
        S1, ST = np.zeros(5), np.zeros(5)
        for a, ck in zip(self.alpha[1:], c[1:]):
            nz = np.nonzero(a)[0]
            if len(nz) == 1:
                S1[nz[0]] += ck**2
            ST[nz] += ck**2
        return S1 / var, ST / var


# ---- POD ROM ----------------------------------------------------------------------------------------------------------
class POD:
    def __init__(self, n_modes=8, p=4):
        self.k, self.p = n_modes, p

    def fit(self, X, V):
        self.mean = V.mean(0)
        U, S, Vt = np.linalg.svd(V - self.mean, full_matrices=False)
        self.phi = Vt[: self.k]
        self.energy = (S**2 / np.sum(S**2))[: self.k]
        A = (V - self.mean) @ self.phi.T
        self.pce = PCE(self.p).fit(X, A)
        return self

    def predict(self, X):
        return self.mean + self.pce.predict(X) @ self.phi


def fit_gp(X, y, seed=0):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(5), length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(1e-6, (1e-10, 1e-2))
    gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=seed)
    gp.fit(to_unit(X), y)
    return gp


def sobol_direct(c: Circuit, n=2048, seed=0, boot=200):
    """Saltelli sampling on the simulator; first-order and total indices with 95% bootstrap intervals (SALib)."""
    from SALib.analyze import sobol as sa
    from SALib.sample import sobol as ssamp
    prob = dict(num_vars=5, names=NAMES, bounds=RANGES.tolist())
    X = ssamp.sample(prob, n, calc_second_order=False, seed=seed)
    _, t = simulate_many(c, X)
    r = sa.analyze(prob, t, calc_second_order=False, num_resamples=boot, conf_level=0.95, seed=seed)
    return r, len(X)


def timing(c: Circuit, gp, pce, n=2000):
    X = lhs(n, 7)
    simulate_many(c, X[:10]); t0 = time.perf_counter(); simulate_many(c, X); td = (time.perf_counter() - t0) / n
    t0 = time.perf_counter(); gp.predict(to_unit(X)); tg = (time.perf_counter() - t0) / n
    t0 = time.perf_counter(); pce.predict(X); tp = (time.perf_counter() - t0) / n
    return dict(direct_ms=td * 1e3, gp_ms=tg * 1e3, pce_ms=tp * 1e3)
