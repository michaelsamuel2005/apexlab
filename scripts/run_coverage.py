"""Posterior-predictive lap-time intervals checked on laps the posterior never saw (laps 4 and 5 of the five fastest; laps 1 to 3 were used)."""
import glob, pickle
import numpy as np, pandas as pd
from apexlab import bayes, circuits
from apexlab.fetch import ROOT
cs = circuits.load_all(); res = {}
for f in glob.glob(str(ROOT / "outputs/cache/bayes*.pkl")): res.update(pickle.load(open(f, "rb")))
rng = np.random.default_rng(0); rows = []
for k, r in res.items():
    c = cs[k]; pred = bayes.predictive_lap_time(c, r["log_theta"], r["sigma_t"], 2, rng); obs = c.times[3:5]
    for j in range(2):
        lo5, lo25, md, hi75, hi95 = np.percentile(pred[:, j], [5, 25, 50, 75, 95])
        rows.append(dict(circuit=k, lap=j + 4, observed=obs[j], median=md, lo90=lo5, hi90=hi95, in50=lo25 <= obs[j] <= hi75, in90=lo5 <= obs[j] <= hi95, width90=hi95 - lo5,
                         param_only_width90=np.subtract(*np.percentile(bayes.predictive_lap_time(c, r["log_theta"], np.zeros_like(r["sigma_t"]), 1, rng)[:, 0], [95, 5]))))
d = pd.DataFrame(rows); d.to_csv(ROOT / "outputs/tables/bayes_coverage.csv", index=False)
post = []
for k, r in res.items():
    th = np.exp(r["log_theta"]); post.append(dict(circuit=k, **{f"{n}_med": np.median(th[:, i]) for i, n in enumerate(bayes.NAMES)}, **{f"{n}_lo": np.percentile(th[:, i], 5) for i, n in enumerate(bayes.NAMES)},
        **{f"{n}_hi": np.percentile(th[:, i], 95) for i, n in enumerate(bayes.NAMES)}, corr_ClA_mu=np.corrcoef(np.log(th[:, 1]), np.log(th[:, 2]))[0, 1], sigma_v=r["sigma_v"].mean(), sigma_t=r["sigma_t"].mean(),
        rhat_max=r["rhat"].max(), ess_min=r["ess"].min(), divergences=r["div"]))
pd.DataFrame(post).to_csv(ROOT / "outputs/tables/bayes_posteriors.csv", index=False)
print(d[["in50", "in90", "width90", "param_only_width90"]].mean().round(2)); print(len(d), "held-out laps")
