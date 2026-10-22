"""Tyre-degradation mixed model, chronological split, and the conversion of pace loss into an equivalent grip loss."""
import json, pickle
import jax, jax.numpy as jnp, numpy as np, pandas as pd
from apexlab import circuits, physics as P, tyres
from apexlab.fetch import ROOT

OUT = ROOT / "outputs" / "tables"
d = tyres.clean(tyres.load_race_laps())
res, ev, te = tyres.chronological_eval(d, 16)
full = tyres.fit(d)
ci = full.conf_int()
fe = pd.DataFrame({"coef": full.fe_params, "lo95": ci.loc[full.fe_params.index, 0], "hi95": ci.loc[full.fe_params.index, 1]})
fe.to_csv(OUT / "tyre_fixed_effects.csv")
# degradation rate by compound, in %/lap and s/lap on a 90 s lap
rates = {}
for comp in ("SOFT", "MEDIUM", "HARD"):
    k = f"life:C(compound, Treatment('MEDIUM'))[{comp}]"
    rates[comp] = dict(pct_per_lap=float(fe.loc[k, "coef"]), lo95=float(fe.loc[k, "lo95"]), hi95=float(fe.loc[k, "hi95"]), s_per_lap_90s=float(fe.loc[k, "coef"]) * 0.9)
# equivalent grip loss: d(lap time)/d(mu) from the simulator at fitted parameters (median of point fits) on three circuits
fits = pickle.loads((ROOT / "outputs/cache/fits.pkl").read_bytes())
th = np.median(np.array(list(fits.values())), axis=0)
cs = circuits.load_all(); sens = {}
for k in ("ITA", "GBR", "HUN"):
    c = cs[k]; kap, dr = jnp.asarray(c.track.kappa), jnp.asarray((c.drs_frac[0] > 0.5).astype(float))
    f = lambda mu: P._simulate(kap, c.track.ds, th[0], th[1], mu, th[3], circuits.MASS_QUALI, dr)[1]
    t0 = float(f(th[2])); g = float(jax.grad(f)(th[2]))
    sens[k] = dict(t_lap=t0, dt_dmu=g, pct_time_per_pct_mu=g * th[2] / t0)
for comp, r in rates.items():
    r["equiv_grip_loss_pct_per_lap"] = {k: r["pct_per_lap"] / (-s["pct_time_per_pct_mu"]) for k, s in sens.items()}
json.dump(dict(eval=ev, rates=rates, sens=sens, n_laps=len(d), n_events=int(d.event.nunique()), n_groups=int(d.grp.nunique()),
               random_effects=dict(group_var=float(res.cov_re.iloc[0, 0]), slope_var=float(res.cov_re.iloc[1, 1]), resid_var=float(res.scale))),
          open(OUT / "tyres.json", "w"), indent=1)
te.to_pickle(ROOT / "outputs/cache/tyre_test.pkl"); d.to_pickle(ROOT / "outputs/cache/tyre_laps.pkl")
print(json.dumps(dict(eval=ev, rates=rates, sens=sens), indent=1))
