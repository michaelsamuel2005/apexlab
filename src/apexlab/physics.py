"""Quasi-steady-state point-mass lap-time simulation (JAX, differentiable, jit-compiled).

State: speed v(s) along a closed line of curvature kappa(s). At each point the car is on the limit of a friction ellipse:

    normal load       N   = m g + 0.5 rho ClA v^2
    lateral force     Fy  = m v^2 kappa            (needed to follow the line)
    tyre capacity         mu N  in any direction  (friction ellipse: Fx_avail = sqrt((mu N)^2 - Fy^2))
    drive force       Fd  = min(P / v, k_trac * Fx_avail)
    braking force     Fb  = Fx_avail
    resistance        R   = 0.5 rho CdA v^2 + c_rr m g

Cornering-speed limit (Fy = mu N):  v_c^2 = mu m g / (m kappa - mu 0.5 rho ClA)   (infinite if the denominator <= 0).
Speed profile: a backward braking pass and a forward traction pass over two concatenated laps (closed loop), min() with v_c.

Not modelled: load transfer, tyre temperature and wear, suspension and kerbs, gear shifts and ERS deployment (power is one
effective constant), DRS only as a fixed fractional drag cut on the measured open zones, wind, slope, and the driver. Parameters are therefore *effective*.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

RHO = 1.20      # air density, kg/m3 (assumption; the 2025 weather channels could refine it)
G = 9.81
CRR = 0.010     # rolling resistance coefficient (assumption)
K_TRAC = 0.55   # share of the friction capacity available to the driven rear axle (assumption)
V_MIN, V_CAP = 8.0, 120.0
PARAMS = ("cda", "cla", "mu", "power_kw", "mass")


def _fx_avail(v, kappa, m, cla, mu):
    n = m * G + 0.5 * RHO * cla * v**2
    fy = m * v**2 * kappa
    return jnp.sqrt(jnp.maximum((mu * n) ** 2 - fy**2, 1e-6))


def corner_limit(kappa, m, cla, mu):
    den = m * kappa - mu * 0.5 * RHO * cla
    v2 = mu * m * G / jnp.where(den > 1e-9, den, 1.0)
    return jnp.where(den > 1e-9, jnp.minimum(jnp.sqrt(v2), V_CAP), V_CAP)


DRS_CUT = 0.25   # fractional drag (CdA) reduction with the wing open (assumption; stated in the model card)


def _simulate(kappa, ds, cda, cla, mu, power_kw, mass, drs=None):
    m, P = mass, power_kw * 1e3
    drs2 = jnp.zeros(2 * kappa.shape[0]) if drs is None else jnp.concatenate([drs, drs])
    n = kappa.shape[0]
    kap2 = jnp.concatenate([kappa, kappa])
    vlim = corner_limit(kap2, m, cla, mu)

    def resist(v, d):
        return 0.5 * RHO * cda * (1.0 - DRS_CUT * d) * v**2 + CRR * m * G

    def back(v_next, xs):                        # braking: speed allowed at i given speed at i+1
        vl, kp, d = xs
        fb = _fx_avail(v_next, kp, m, cla, mu)
        a = (fb + resist(v_next, d)) / m
        v = jnp.minimum(vl, jnp.sqrt(v_next**2 + 2 * a * ds))
        return v, v

    _, vb = jax.lax.scan(back, vlim[-1], (vlim[::-1], kap2[::-1], drs2[::-1]))
    vb = vb[::-1]

    def fwd(v_prev, xs):                         # traction: speed reachable at i from speed at i-1
        vl, kp, d = xs
        fx = _fx_avail(v_prev, kp, m, cla, mu)
        fd = jnp.minimum(P / jnp.maximum(v_prev, V_MIN), K_TRAC * fx)
        a = (fd - resist(v_prev, d)) / m
        v = jnp.minimum(vl, jnp.sqrt(jnp.maximum(v_prev**2 + 2 * a * ds, V_MIN**2)))
        return v, v

    _, vf = jax.lax.scan(fwd, vb[0], (vb, kap2, drs2))
    v = vf[n:]                                    # second lap of the forward pass
    v = jnp.maximum(v, V_MIN)
    vn = jnp.roll(v, -1)
    t = jnp.sum(2 * ds / (v + vn))
    return v, t


@partial(jax.jit, static_argnums=())
def simulate(kappa, ds, theta, drs=None):
    """theta = (cda, cla, mu, power_kw, mass); drs = per-point open fraction in [0,1] or None. Returns (speed m/s, lap time s)."""
    return _simulate(kappa, ds, theta[0], theta[1], theta[2], theta[3], theta[4], drs)


simulate_batch = jax.jit(jax.vmap(lambda k, d, th, dr=None: _simulate(k, d, th[0], th[1], th[2], th[3], th[4], dr), in_axes=(None, None, 0)))


def lap_time(kappa: np.ndarray, ds: float, theta) -> float:
    return float(simulate(jnp.asarray(kappa), ds, jnp.asarray(theta, float))[1])


# ---- closed-form references used by the verification tests -------------------------------------------------------------
def terminal_speed(cda, power_kw, mass, mu=10.0):
    """Straight-line top speed where drive power balances drag and rolling resistance: P/v = 0.5 rho CdA v^2 + c_rr m g."""
    from scipy.optimize import brentq
    P = power_kw * 1e3
    return brentq(lambda v: P / v - 0.5 * RHO * cda * v**2 - CRR * mass * G, 5.0, 200.0)


def circular_corner_speed(radius, cla, mu, mass):
    """Steady-state cornering speed on a circle of radius R with downforce: v^2 = mu m g / (m / R - mu 0.5 rho ClA)."""
    den = mass / radius - mu * 0.5 * RHO * cla
    return np.sqrt(mu * mass * G / den) if den > 0 else np.inf


def braking_distance_no_aero(v0, v1, mu, mass, cda=0.0):
    """Straight-line braking distance from v0 to v1 with constant tyre-limited deceleration and no downforce or drag (cla = cda = 0)."""
    a = mu * G
    return (v0**2 - v1**2) / (2 * a)
