"""Empirical-Bayes identification of (CdA, ClA, mu, P) across circuits (NumPyro, NUTS).

Per circuit, log theta ~ Normal(m, s) with (m, s) the mean and spread of the log point-fits of the *other* circuits
(empirical Bayes; a full hierarchical NUTS run cost 0.1 s per gradient and was not affordable here).
Likelihood: speed on the pole lap (thinned to 50 m, residual scale sigma_v learned) and lap times of the training laps
(scale sigma_t learned). Neighbouring speed residuals are strongly autocorrelated, so sigma_v absorbs model error and the
intervals are *checked on held-out laps* rather than trusted.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from . import physics as P
from .circuits import MASS_QUALI, Circuit

PRIOR_MU = np.log([1.4, 4.0, 1.8, 550.0])   # CdA m2, ClA m2, mu, kW: broad, weakly informative
THIN = 10
NAMES = ["CdA", "ClA", "mu", "P_kW"]


def _data(circs: dict[str, Circuit], train_laps=(0, 1, 2)):
    keys = list(circs)
    kap = jnp.stack([jnp.asarray(circs[k].track.kappa) for k in keys]) if len({len(circs[k].track.kappa) for k in keys}) == 1 else None
    return keys, kap


def prior_from(fits: dict, exclude: str | None = None, floor=0.2):
    L = np.log([v for k, v in fits.items() if k != exclude])
    return L.mean(0), np.maximum(L.std(0, ddof=1), floor)


def make_model(c: Circuit, m, s, train_laps=(0, 1, 2)):
    kap, ds = jnp.asarray(c.track.kappa), c.track.ds
    dr = jnp.asarray((c.drs_frac[0] > 0.5).astype(float))
    vm = jnp.asarray(c.v_meas[0][::THIN])
    tm = jnp.asarray(c.times[list(train_laps)])

    def model():
        lt = numpyro.sample("log_theta", dist.Normal(jnp.asarray(m), jnp.asarray(s)).to_event(1))
        th = jnp.exp(lt)
        sv = numpyro.sample("sigma_v", dist.HalfNormal(5.0))
        st = numpyro.sample("sigma_t", dist.HalfNormal(1.0))
        v, t = P._simulate(kap, ds, th[0], th[1], th[2], th[3], MASS_QUALI, dr)
        numpyro.sample("v", dist.Normal(v[::THIN], sv).to_event(1), obs=vm)
        numpyro.sample("t", dist.Normal(t, st).expand([len(train_laps)]).to_event(1), obs=tm)
    return model


def run(c: Circuit, m, s, train_laps=(0, 1, 2), warmup=500, samples=500, chains=2, seed=0, max_tree_depth=8):
    mc = MCMC(NUTS(make_model(c, m, s, train_laps), target_accept_prob=0.9, max_tree_depth=max_tree_depth), num_warmup=warmup, num_samples=samples,
              num_chains=chains, chain_method="sequential", progress_bar=False)
    mc.run(jax.random.PRNGKey(seed))
    return mc


def predictive_lap_time(c: Circuit, log_theta, sigma_t, n_laps, rng):
    """Posterior-predictive lap times (parameter uncertainty plus sigma_t) for n_laps further laps: array (draws, n_laps)."""
    kap, dr = jnp.asarray(c.track.kappa), jnp.asarray((c.drs_frac[0] > 0.5).astype(float))
    th = jnp.exp(jnp.asarray(log_theta))
    _, t = jax.vmap(lambda x: P._simulate(kap, c.track.ds, x[0], x[1], x[2], x[3], MASS_QUALI, dr))(th)
    t = np.asarray(t)
    return t[:, None] + rng.normal(0, 1, (len(t), n_laps)) * np.asarray(sigma_t)[:, None]
