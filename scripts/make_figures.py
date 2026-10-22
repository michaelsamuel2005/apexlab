"""All report figures from the cached results and tables."""
import json, pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from apexlab import circuits, ident, validate, physics as P
from apexlab.fetch import ROOT

FIG = ROOT / "outputs" / "figures"; FIG.mkdir(parents=True, exist_ok=True); T = ROOT / "outputs" / "tables"
BLUE, ORANGE, GREY, TEAL, RED = "#2a6fb0", "#d97a12", "#6b7280", "#1b8a7a", "#b3402f"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 150, "savefig.bbox": "tight"})
cs = circuits.load_all(); fits = pickle.loads((ROOT / "outputs/cache/fits.pkl").read_bytes())

# 1 measured vs simulated speed (fit lap) and a held-out driver's lap
fig, ax = plt.subplots(3, 1, figsize=(7.2, 7.4), gridspec_kw={"hspace": .42})
for a, k in zip(ax, ("ITA", "GBR", "HUN")):
    c = cs[k]; rm, dt, v = ident.predict(c, fits[k], 0)
    a.plot(c.track.s, c.v_meas[0] * 3.6, color=GREY, lw=1.2, label=f"measured, pole lap ({c.lap_info[0][0]})")
    a.plot(c.track.s, v * 3.6, color=BLUE, lw=1.2, label="simulated (fitted)")
    d = c.drs_frac[0] > .5; a.fill_between(c.track.s, 0, 1, where=d, color=ORANGE, alpha=.15, transform=a.get_xaxis_transform(), label="DRS open")
    a.set_title(f"{k}: speed RMSE {rm*3.6:.1f} km/h, lap time {dt:+.2f} s", loc="left", fontsize=9); a.set_ylabel("speed, km/h")
ax[0].legend(ncol=3, fontsize=7, loc="lower right"); ax[-1].set_xlabel("distance along lap, m")
fig.savefig(FIG / "fig1_speed_traces.png"); plt.close(fig)

# 2 leave-one-circuit-out
B = validate.leave_one_circuit_out(cs, fits); B.to_csv(T / "loco.csv", index=False)
fig, a = plt.subplots(figsize=(7.2, 3.2)); x = np.arange(len(B)); w = .38
a.bar(x - w/2, B.baseline_abs_dt, w, color=GREY, label="pace baseline (length x training pace)"); a.bar(x + w/2, B.model_abs_dt, w, color=BLUE, label="apexlab, no data from held-out circuit")
a.set_xticks(x, B.circuit); a.set_ylabel("absolute lap-time error, s"); a.legend(fontsize=7)
a.set_title(f"Leave-one-circuit-out: mean error {B.model_abs_dt.mean():.1f} s vs {B.baseline_abs_dt.mean():.1f} s", loc="left", fontsize=9)
fig.savefig(FIG / "fig2_loco.png"); plt.close(fig)

# 3 posterior forest + ClA-mu correlation
import glob
res = {}
for f in glob.glob(str(ROOT / "outputs/cache/bayes*.pkl")): res.update(pickle.load(open(f, "rb")))
if res:
    names = ["CdA, m²", "ClA, m²", "μ", "P, kW"]
    fig, ax = plt.subplots(1, 4, figsize=(7.4, 3.2), sharey=True)
    ks = [k for k in cs if k in res]
    for j, a in enumerate(ax):
        for i, k in enumerate(ks):
            s = np.exp(res[k]["log_theta"][:, j]); lo, md, hi = np.percentile(s, [5, 50, 95])
            a.plot([lo, hi], [i, i], color=BLUE, lw=2); a.plot(md, i, "o", color=BLUE, ms=3)
        a.set_xlabel(names[j]); a.set_yticks(range(len(ks)), ks)
    ax[0].invert_yaxis(); fig.suptitle("Posterior medians and 90% intervals by circuit (prior: other circuits' point fits)", fontsize=9, x=.02, ha="left")
    fig.savefig(FIG / "fig3_posteriors.png"); plt.close(fig)
    k = "GBR" if "GBR" in res else ks[0]; s = np.exp(res[k]["log_theta"])
    fig, a = plt.subplots(1, 2, figsize=(7.2, 3))
    a[0].scatter(s[:, 1], s[:, 2], s=6, alpha=.4, color=BLUE); a[0].set_xlabel("ClA, m²"); a[0].set_ylabel("μ"); a[0].set_title(f"{k}: downforce and grip trade off (r = {np.corrcoef(np.log(s[:,1]), np.log(s[:,2]))[0,1]:.2f})", loc="left", fontsize=8)
    a[1].scatter(s[:, 0], s[:, 3], s=6, alpha=.4, color=TEAL); a[1].set_xlabel("CdA, m²"); a[1].set_ylabel("P, kW"); a[1].set_title(f"drag and power (r = {np.corrcoef(np.log(s[:,0]), np.log(s[:,3]))[0,1]:.2f})", loc="left", fontsize=8)
    fig.savefig(FIG / "fig4_identifiability.png"); plt.close(fig)

# 5 surrogate convergence + Sobol
cv = pd.read_csv(T / "surrogate_convergence.csv") if (T / "surrogate_convergence.csv").exists() else None
if cv is not None:
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3))
    for m, col in (("GP-ARD", BLUE), ("PCE", ORANGE)):
        g = cv[cv.model == m].groupby("n_train").rmse_s.mean(); ax[0].loglog(g.index, g.values, "o-", color=col, label=m)
    ax[0].set_xlabel("training simulations"); ax[0].set_ylabel("lap-time RMSE on 1000 test points, s"); ax[0].legend(fontsize=7)
    so = pd.read_csv(T / "sobol.csv"); so = so[so.method.str.startswith("Saltelli")]
    st = so[[f"ST_{n}" for n in ("CdA", "ClA", "mu", "P_kW", "mass")]].mean(); s1 = so[[f"S1_{n}" for n in ("CdA", "ClA", "mu", "P_kW", "mass")]].mean()
    xx = np.arange(5); ax[1].bar(xx - .2, s1.values, .4, color=BLUE, label="first order"); ax[1].bar(xx + .2, st.values, .4, color=ORANGE, label="total")
    ax[1].set_xticks(xx, ["CdA", "ClA", "μ", "P", "mass"]); ax[1].set_ylabel("Sobol index of lap time"); ax[1].legend(fontsize=7)
    fig.savefig(FIG / "fig5_surrogate_sobol.png"); plt.close(fig)

# 6 tyres
if (ROOT / "outputs/cache/tyre_laps.pkl").exists():
    d = pd.read_pickle(ROOT / "outputs/cache/tyre_laps.pkl"); te = pd.read_pickle(ROOT / "outputs/cache/tyre_test.pkl"); tj = json.load(open(T / "tyres.json"))
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3))
    for comp, col in (("SOFT", RED), ("MEDIUM", ORANGE), ("HARD", GREY)):
        r = tj["rates"][comp]; l = np.arange(0, 40); ax[0].plot(l, r["pct_per_lap"] * l, color=col, label=f"{comp}: {r['pct_per_lap']:.3f} %/lap")
        ax[0].fill_between(l, r["lo95"] * l, r["hi95"] * l, color=col, alpha=.15)
    ax[0].set_xlabel("tyre age, laps"); ax[0].set_ylabel("pace loss, % of lap time"); ax[0].legend(fontsize=7)
    ax[1].scatter(te.pred, te.r, s=1, alpha=.15, color=BLUE); lim = [-5, 6]; ax[1].plot(lim, lim, color=GREY, lw=1)
    ax[1].set_xlabel("predicted, % (fixed effects, races 17 to 24)"); ax[1].set_ylabel("observed, %"); ax[1].set_title(f"RMSE {tj['eval']['rmse_model']:.2f} vs {tj['eval']['rmse_baseline']:.2f} (mean-only)", loc="left", fontsize=8)
    fig.savefig(FIG / "fig6_tyres.png"); plt.close(fig)

# 7 assumptions
if (T / "assumption_sensitivity.csv").exists():
    s = pd.read_csv(T / "assumption_sensitivity.csv"); fig, a = plt.subplots(figsize=(7.2, 3))
    a.barh(s.variant[::-1], s.loco_mae_s[::-1], color=BLUE); a.set_xlabel("leave-one-circuit-out lap-time MAE, s"); fig.savefig(FIG / "fig7_assumptions.png"); plt.close(fig)
print("figures done")
