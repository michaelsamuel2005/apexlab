"""Tyre degradation from 2025 race laps: a mixed-effects model with an honest chronological train/test split.

Response r = lap time / (event median of clean laps) - 1.  Fixed effects: race progress (fuel burn proxy), tyre age by compound,
track temperature. Random effects: intercept and tyre-age slope per event-driver stint group.
Clean laps: timing-accurate, not deleted, green status, no in/out lap, not lap 1, tyre age >= 1, within 107% of the event median.
Limitations: no traffic or gap data in these files (dirty air inflates lap times and is not modelled); drivers manage pace.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .fetch import EVENTS, ROOT


def load_race_laps() -> pd.DataFrame:
    rows = []
    for order, (key, folder) in enumerate(EVENTS.items()):
        for f in sorted((ROOT / "data" / "raw" / folder / "Race").glob("*/laptimes.json")):
            d = json.loads(f.read_text())
            df = pd.DataFrame({k: d[k] for k in ["time", "lap", "compound", "stint", "life", "status", "pin", "pout", "iacc", "del", "wTT", "wAT", "drv"] if k in d})
            df["event"], df["order"] = key, order
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col in ("time", "lap", "life", "stint"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["lap", "life", "stint"])
    ok = (d.iacc == True) & (d["del"] == False) & (d.status.astype(str) == "1") & (d.pin.astype(str) == "None") & (d.pout.astype(str) == "None")
    ok &= (d.lap > 1) & (d.life >= 1) & d.time.notna() & d.compound.isin(["SOFT", "MEDIUM", "HARD"])
    d = d[ok].copy()
    med = d.groupby("event").time.transform("median")
    d = d[d.time < 1.07 * med].copy()
    d["ref"] = d.groupby("event").time.transform("median")
    d["r"] = 100 * (d.time / d.ref - 1)                       # per cent
    d["prog"] = d.lap / d.groupby("event").lap.transform("max")
    d["tt"] = pd.to_numeric(d.wTT, errors="coerce")
    d["tt"] = d.tt - d.groupby("event").tt.transform("mean")   # deviation from the event mean track temperature
    d["tt"] = d.tt.fillna(0.0)
    d["grp"] = d.event + "_" + d.drv + "_" + d.stint.astype(str)
    return d.reset_index(drop=True)


FORMULA = "r ~ prog + C(compound, Treatment('MEDIUM')) + life:C(compound, Treatment('MEDIUM')) + tt"


def fit(d: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm(FORMULA, d, groups=d["grp"], re_formula="~life")
        return m.fit(method="lbfgs", maxiter=400)


def predict_fixed(res, d: pd.DataFrame) -> np.ndarray:
    return np.asarray(res.predict(d))


def chronological_eval(d: pd.DataFrame, n_train_events: int = 16):
    """Train on the first events, test on the later ones. Population-level prediction (stint random effects unknown for a new race)."""
    tr, te = d[d.order < n_train_events], d[d.order >= n_train_events]
    res = fit(tr)
    pred = predict_fixed(res, te)
    # predictive sd of a lap in a new stint: residual + random-effect variance at that tyre age
    cov = np.asarray(res.cov_re)
    sd = np.sqrt(res.scale + cov[0, 0] + 2 * cov[0, 1] * te.life.values + cov[1, 1] * te.life.values ** 2)
    e = te.r.values - pred
    base = te.r.values - tr.r.mean()
    out = dict(n_train=len(tr), n_test=len(te), rmse_model=float(np.sqrt(np.mean(e**2))), rmse_baseline=float(np.sqrt(np.mean(base**2))),
               cover90=float(np.mean(np.abs(e) < 1.645 * sd)), cover50=float(np.mean(np.abs(e) < 0.6745 * sd)))
    return res, out, te.assign(pred=pred, sd=sd)
