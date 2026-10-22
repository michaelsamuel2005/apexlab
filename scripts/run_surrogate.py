"""Surrogates, POD reduced-order model and Sobol sensitivity on three circuits. Writes outputs/tables/surrogate_*.csv."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from apexlab import circuits, surrogate as S
from apexlab.fetch import ROOT

OUT = ROOT / "outputs" / "tables"; OUT.mkdir(parents=True, exist_ok=True)
cs = circuits.load_all(); keys = ["ITA", "GBR", "HUN"]
Xte = S.lhs(1000, seed=99)
conv, sob, tim, rom = [], [], [], []
for k in keys:
    c = cs[k]; vte, tte = S.simulate_many(c, Xte)
    for n in (25, 50, 100, 200, 400):
        X = S.lhs(n, seed=n); v, t = S.simulate_many(c, X)
        gp = S.fit_gp(X, t); pce = S.PCE(3 if n < 100 else 4).fit(X, t)
        for name, pred in (("GP-ARD", gp.predict(S.to_unit(Xte))), ("PCE", pce.predict(Xte).ravel())):
            conv.append(dict(circuit=k, n_train=n, model=name, rmse_s=float(np.sqrt(np.mean((pred - tte) ** 2))), r2=float(r2_score(tte, pred)), max_abs_s=float(np.abs(pred - tte).max())))
    # final models at n=400
    X = S.lhs(400, seed=400); v, t = S.simulate_many(c, X)
    gp = S.fit_gp(X, t); pce = S.PCE(4).fit(X, t)
    tim.append(dict(circuit=k, **S.timing(c, gp, pce)))
    ls = gp.kernel_.k1.k2.length_scale
    sob.append(dict(circuit=k, method="PCE-analytic", **{f"S1_{n}": a for n, a in zip(S.NAMES, pce.sobol()[0])}, **{f"ST_{n}": a for n, a in zip(S.NAMES, pce.sobol()[1])}))
    r, nsim = S.sobol_direct(c, n=2048)
    sob.append(dict(circuit=k, method=f"Saltelli-direct ({nsim} sims)", **{f"S1_{n}": a for n, a in zip(S.NAMES, r["S1"])}, **{f"ST_{n}": a for n, a in zip(S.NAMES, r["ST"])},
                    **{f"S1ci_{n}": a for n, a in zip(S.NAMES, r["S1_conf"])}, **{f"STci_{n}": a for n, a in zip(S.NAMES, r["ST_conf"])}))
    sob.append(dict(circuit=k, method="GP ARD length-scales (unit cube)", **{f"S1_{n}": float(a) for n, a in zip(S.NAMES, ls)}))
    # POD ROM of the speed trace
    for n_modes in (2, 4, 8, 12):
        pod = S.POD(n_modes, 4).fit(X, v); vp = pod.predict(Xte)
        rom.append(dict(circuit=k, modes=n_modes, energy_captured=float(pod.energy.sum()) if n_modes <= 12 else np.nan,
                        speed_rmse_ms=float(np.sqrt(np.mean((vp - vte) ** 2))), speed_max_abs_ms=float(np.abs(vp - vte).max()),
                        laptime_rmse_s=float(np.sqrt(np.mean((2 * c.track.ds / (vp + np.roll(vp, -1, axis=1))).sum(1) - tte) ** 2))))
pd.DataFrame(conv).to_csv(OUT / "surrogate_convergence.csv", index=False)
pd.DataFrame(sob).to_csv(OUT / "sobol.csv", index=False)
pd.DataFrame(tim).to_csv(OUT / "surrogate_timing.csv", index=False)
pd.DataFrame(rom).to_csv(OUT / "pod_rom.csv", index=False)
print(pd.DataFrame(conv).query("n_train in (50,400)").round(3).to_string()); print(pd.DataFrame(tim).round(4)); print(pd.DataFrame(rom).round(3))
print(pd.DataFrame(sob).round(3).to_string())
