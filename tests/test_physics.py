"""Verification of the point-mass simulation against closed-form answers (no telemetry needed)."""
import numpy as np
import jax.numpy as jnp
import pytest

from apexlab import physics as P
from apexlab.geometry import Track, build_track
from apexlab.laps import Lap


def run(kappa, ds, cda, cla, mu, power_kw, mass=800.0):
    v, t = P.simulate(jnp.asarray(kappa), ds, jnp.asarray([cda, cla, mu, power_kw, mass]))
    return np.asarray(v), float(t)


def test_terminal_speed_matches_power_balance():
    kappa = np.zeros(4000)
    v, _ = run(kappa, 5.0, 1.15, 0.0, 2.0, 600.0)
    assert v[-1] == pytest.approx(P.terminal_speed(1.15, 600.0, 800.0), rel=0.01)


def test_circle_lap_time_matches_closed_form():
    for radius, cla in [(60.0, 0.0), (120.0, 4.0), (200.0, 2.5)]:
        n = 400
        ds = 2 * np.pi * radius / n
        v, t = run(np.full(n, 1 / radius), ds, 0.0, cla, 1.8, 900.0)      # no drag: the closed form is exact up to c_rr
        vc = P.circular_corner_speed(radius, cla, 1.8, 800.0)
        assert np.all(np.abs(v - vc) < 0.01 * vc)
        assert t == pytest.approx(2 * np.pi * radius / vc, rel=0.01)
        # with drag the tyres must also supply thrust, so the steady speed is a little below the pure-cornering limit
        v2, _ = run(np.full(n, 1 / radius), ds, 1.1, cla, 1.8, 900.0)
        assert 0.9 * vc < v2.mean() <= vc * 1.001


def test_braking_distance_matches_constant_deceleration():
    # straight into a tight corner whose cornering limit is v1; no aero, so deceleration is (mu + c_rr) g exactly
    ds, mu = 1.0, 1.5
    kappa = np.zeros(3000)
    kappa[2900:] = 1 / 30.0
    v, _ = run(kappa, ds, 0.0, 0.0, mu, 2000.0)
    v1 = np.sqrt(mu * G * 30.0) if (G := P.G) else 0
    a = (mu + P.CRR) * P.G
    # find the braking point: the last grid point where the speed starts to fall before the corner
    v0 = v[2000]
    d_model = None
    i = 2899
    while v[i] < v[i - 1]:
        i -= 1
    d_model = (2900 - i) * ds
    assert d_model == pytest.approx((v[i] ** 2 - v1**2) / (2 * a), rel=0.03)


def test_traction_limited_acceleration():
    # from the corner exit, with huge power the car accelerates at k_trac * mu * g - c_rr g (no aero)
    ds, mu = 0.5, 1.6
    kappa = np.zeros(2000)
    kappa[:20] = 1 / 20.0
    v, _ = run(kappa, ds, 0.0, 0.0, mu, 1e5)
    i0 = 40
    a_meas = (v[i0 + 100] ** 2 - v[i0] ** 2) / (2 * 100 * ds)
    assert a_meas == pytest.approx(P.K_TRAC * mu * P.G - P.CRR * P.G, rel=0.03)


def test_monotone_in_power_and_downforce():
    # a synthetic circuit with straights and corners
    s = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    kappa = 0.004 + 0.02 * np.maximum(0, np.sin(3 * s)) ** 4
    ds = 5000.0 / 600
    t = lambda **k: run(kappa, ds, **{"cda": 1.1, "cla": 4.0, "mu": 1.8, "power_kw": 650.0, **k})[1]
    assert t(power_kw=750.0) < t(power_kw=650.0) < t(power_kw=550.0)
    assert t(cla=4.5) < t(cla=4.0) < t(cla=3.5)       # in corner-dominated laps downforce wins at fixed drag
    assert t(cda=1.0) < t(cda=1.2)
    assert t(mu=2.0) < t(mu=1.6)


def test_lateral_acceleration_never_exceeds_tyre_capacity():
    s = np.linspace(0, 2 * np.pi, 800, endpoint=False)
    kappa = 0.003 + 0.03 * np.maximum(0, np.sin(4 * s)) ** 2
    v, _ = run(kappa, 4000.0 / 800, 1.1, 4.0, 1.8, 650.0)
    n = 800.0 * P.G + 0.5 * P.RHO * 4.0 * v**2
    assert np.all(800.0 * v**2 * kappa <= 1.8 * n * 1.001)


def test_geometry_recovers_circle_curvature():
    """Position noise of 0.3 m (the spline smoothing level). Noise of 0.5 m or more breaks the fit: curvature blows up, so the geometry setting assumes five-lap averaging."""
    R = 150.0
    th = np.linspace(0, 2 * np.pi, 600)
    rng = np.random.default_rng(0)
    lap = Lap("SYN", "XXX", 1, 60.0, "SOFT", th, th * R, np.full_like(th, 50.0),
              R * np.cos(th) + rng.normal(0, 0.3, th.size), R * np.sin(th) + rng.normal(0, 0.3, th.size),
              th, th, th, th)
    t = build_track(lap)
    assert np.median(t.kappa) == pytest.approx(1 / R, rel=0.15)
    assert t.length == pytest.approx(2 * np.pi * R, rel=0.02)
