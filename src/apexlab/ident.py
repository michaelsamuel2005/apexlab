"""Identify effective vehicle parameters from measured speed traces, and test them on laps and circuits the fit never saw.

theta = (CdA, ClA, mu, P_kW) with the mass fixed (assumption). Fits use log-parameters and bounded least squares on
the speed residual (m/s, thinned to about 50 m spacing) plus the lap-time residual.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from . import physics as P
from . import circuits as _circuits
from .circuits import Circuit

LO = np.array([0.6, 1.0, 0.8, 300.0])
HI = np.array([1.8, 7.0, 3.0, 1000.0])
SIGMA_V = 2.0       # m/s, scale of speed residuals (thinned), tuned only by the order of magnitude of the first fit
SIGMA_T = 0.15      # s, scale of lap-time residual
USE_DRS = True
THIN = 10           # grid points between residual samples (50 m)


def _full(theta):
    return jnp.concatenate([theta, jnp.array([_circuits.MASS_QUALI])])


def make_residual(c: Circuit, lap_idx: int):
    kappa, ds = jnp.asarray(c.track.kappa), c.track.ds
    vm = jnp.asarray(c.v_meas[lap_idx][::THIN])
    tm = float(c.times[lap_idx])
    drs = jnp.asarray((c.drs_frac[lap_idx] > 0.5).astype(float)) if USE_DRS else None

    def sim(theta):
        v, t = P.simulate(kappa, ds, _full(theta), drs)
        return v, t

    @jax.jit
    def res(theta):
        v, t = sim(theta)
        return jnp.concatenate([(v[::THIN] - vm) / SIGMA_V, jnp.array([(t - tm) / SIGMA_T])])

    return sim, res


def fit_circuit(c: Circuit, lap_idx: int = 0, x0=(1.15, 4.0, 1.9, 650.0)) -> dict:
    """Free fit of (CdA, ClA, mu, P) to one lap at one circuit."""
    _, res = make_residual(c, lap_idx)
    jac = jax.jit(jax.jacfwd(res))
    f = lambda x: np.asarray(res(jnp.asarray(np.exp(x))))
    J = lambda x: np.asarray(jac(jnp.asarray(np.exp(x)))) * np.exp(x)[None, :]
    r = least_squares(f, np.log(x0), jac=J, bounds=(np.log(LO), np.log(HI)), x_scale=1.0)
    th = np.exp(r.x)
    return {"theta": th, "cost": float(r.cost), "n": len(r.fun)}


def predict(c: Circuit, theta, lap_idx: int):
    """Speed rmse (m/s) and lap-time error (s, simulated minus measured) of theta on lap lap_idx of circuit c."""
    v, t = P.simulate(jnp.asarray(c.track.kappa), c.track.ds, _full(jnp.asarray(theta)), jnp.asarray((c.drs_frac[lap_idx] > 0.5).astype(float)) if USE_DRS else None)
    v = np.asarray(v)
    return float(np.sqrt(np.mean((v - c.v_meas[lap_idx]) ** 2))), float(t) - float(c.times[lap_idx]), v


def fit_shared(circs: dict[str, Circuit], lap_idx: int = 0, x0=None, fixed_aero: dict | None = None) -> dict:
    """Shared (mu, P) and per-circuit (CdA, ClA) fitted jointly on one lap per circuit."""
    keys = list(circs)
    ress = {k: make_residual(circs[k], lap_idx)[1] for k in keys}
    n = len(keys)

    def unpack(z):
        e = jnp.exp(z)
        mu, pw = e[0], e[1]
        aero = e[2:].reshape(n, 2)
        return [jnp.array([aero[i, 0], aero[i, 1], mu, pw]) for i in range(n)]

    @jax.jit
    def allres(z):
        ths = unpack(z)
        return jnp.concatenate([ress[k](ths[i]) for i, k in enumerate(keys)])

    jac = jax.jit(jax.jacfwd(allres))
    z0 = np.log(np.r_[1.9, 650.0, np.tile([1.15, 4.0], n)]) if x0 is None else np.log(x0)
    lo = np.log(np.r_[LO[2], LO[3], np.tile(LO[:2], n)])
    hi = np.log(np.r_[HI[2], HI[3], np.tile(HI[:2], n)])
    f = lambda z: np.asarray(allres(jnp.asarray(z)))
    Jf = lambda z: np.asarray(jac(jnp.asarray(z))) * np.exp(z)[None, :]
    r = least_squares(f, z0, jac=Jf, bounds=(lo, hi))
    e = np.exp(r.x)
    return {"mu": e[0], "power_kw": e[1], "aero": {k: e[2 + 2 * i: 4 + 2 * i] for i, k in enumerate(keys)}, "cost": float(r.cost)}
