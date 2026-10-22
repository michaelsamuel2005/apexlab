"""Validation of the identified model: (A) other drivers' laps at the same circuit, (B) leave-one-circuit-out with an aero prior,
compared with a model-free baseline (circuit length times the pace of the training circuits)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ident
from .circuits import Circuit


def per_lap_scores(c: Circuit, theta) -> pd.DataFrame:
    rows = []
    for i, (drv, lap, t, comp) in enumerate(c.lap_info):
        rm, dt, _ = ident.predict(c, theta, i)
        rows.append(dict(circuit=c.key, lap_rank=i, driver=drv, compound=comp, speed_rmse=rm, dt=dt, t_meas=t))
    return pd.DataFrame(rows)


def one_lap_calibration(circs: dict[str, Circuit]):
    """Fit each circuit's four parameters to its pole lap; score laps 2 to 5 (other drivers) that the fit never saw."""
    fits, rows = {}, []
    for k, c in circs.items():
        r = ident.fit_circuit(c, 0)
        fits[k] = r["theta"]
        d = per_lap_scores(c, r["theta"]); d["fit_lap"] = d.lap_rank == 0
        rows.append(d)
    return fits, pd.concat(rows, ignore_index=True)


def leave_one_circuit_out(circs: dict[str, Circuit], fits: dict[str, np.ndarray]):
    """Predict each circuit with the median parameters of the *other* circuits (no data from the held-out circuit),
    against the pace baseline. Returns one row per held-out circuit (scored on all five laps)."""
    rows = []
    for k, c in circs.items():
        others = [kk for kk in circs if kk != k]
        th = np.median([fits[o] for o in others], axis=0)
        pace = np.median([circs[o].times.mean() / circs[o].track.length for o in others])
        base = pace * c.track.length
        sc = per_lap_scores(c, th)
        rows.append(dict(circuit=k, model_dt=sc.dt.mean(), model_abs_dt=sc.dt.abs().mean(), model_speed_rmse=sc.speed_rmse.mean(),
                         baseline_dt=base - c.times.mean(), t_mean=c.times.mean()))
    d = pd.DataFrame(rows)
    d["baseline_abs_dt"] = d.baseline_dt.abs()
    return d
