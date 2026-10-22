"""Tests for the DRS term, the surrogate tools (against functions with known answers) and the tyre-data cleaning."""
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from apexlab import physics as P, surrogate as S, tyres


def test_drs_open_makes_a_straight_faster_and_changes_nothing_when_closed():
    kappa = np.full(400, 1e-5); th = jnp.array([1.2, 4.0, 1.8, 600.0, 830.0])
    _, t0 = P.simulate(jnp.asarray(kappa), 5.0, th)
    _, tz = P.simulate(jnp.asarray(kappa), 5.0, th, jnp.zeros(400))
    _, t1 = P.simulate(jnp.asarray(kappa), 5.0, th, jnp.ones(400))
    assert float(t0) == pytest.approx(float(tz), abs=1e-12)
    assert float(t1) < float(t0)


def test_lhs_is_inside_ranges_and_stratified():
    X = S.lhs(100, seed=3)
    assert np.all(X >= S.RANGES[:, 0]) and np.all(X <= S.RANGES[:, 1])
    for j in range(5):   # exactly one point in each of the 100 strata
        u = (X[:, j] - S.RANGES[j, 0]) / (S.RANGES[j, 1] - S.RANGES[j, 0])
        assert len(np.unique(np.floor(u * 100))) == 100


def test_pce_recovers_a_polynomial_and_its_sobol_indices():
    X = S.lhs(400, seed=1); U = S.to_unit(X)
    y = U[:, 0] + 2 * U[:, 1] + 0.5 * U[:, 0] * U[:, 1]
    pce = S.PCE(3).fit(X, y)
    Xt = S.lhs(200, seed=2); Ut = S.to_unit(Xt)
    assert np.max(np.abs(pce.predict(Xt).ravel() - (Ut[:, 0] + 2 * Ut[:, 1] + 0.5 * Ut[:, 0] * Ut[:, 1]))) < 1e-6
    S1, ST = pce.sobol()
    var = (1 / 3) + 4 / 3 + 0.25 / 9            # variances of x1, 2 x2, 0.5 x1 x2 for uniform(-1,1)
    assert S1[0] == pytest.approx((1 / 3) / var, abs=1e-6) and S1[1] == pytest.approx((4 / 3) / var, abs=1e-6)
    assert ST[0] == pytest.approx((1 / 3 + 0.25 / 9) / var, abs=1e-6)
    assert np.all(S1[2:] < 1e-9)


def test_pod_reconstructs_a_low_rank_field():
    rng = np.random.default_rng(0); X = S.lhs(300, seed=4); U = S.to_unit(X)
    s = np.linspace(0, 1, 50)
    V = 50 + U[:, [0]] * np.sin(2 * np.pi * s) + U[:, [1]] ** 2 * np.cos(2 * np.pi * s)
    pod = S.POD(n_modes=3, p=3).fit(X, V)
    assert np.max(np.abs(pod.predict(X) - V)) < 1e-6


def test_tyre_cleaning_drops_pit_laps_first_lap_and_slow_laps():
    n = 12
    d = pd.DataFrame(dict(time=[90.0] * n, lap=range(1, n + 1), compound=["SOFT"] * n, stint=1, life=range(1, n + 1), status="1", pin="None", pout="None",
                          iacc=True, **{"del": False}, wTT=30.0, wAT=20.0, drv="AAA", event="XXX", order=0))
    d.loc[4, "pin"] = "123.0"; d.loc[7, "time"] = 200.0
    c = tyres.clean(d)
    assert 1 not in c.lap.values and 5 not in c.lap.values and 8 not in c.lap.values
    assert len(c) == n - 3
