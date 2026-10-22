"""Build the per-circuit working set: mean driven line, grid, and the measured speed of the five fastest qualifying laps."""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np

from . import fetch as F
from . import geometry, laps
from .fetch import ROOT

MASS_QUALI = 830.0      # kg, assumption: 800 kg minimum car and driver plus a light qualifying fuel load


@dataclass
class Circuit:
    key: str
    track: geometry.Track
    lap_info: list[tuple[str, int, float, str]]   # driver, lap, official time, compound
    v_meas: np.ndarray                             # (n_laps, N) measured speed on the grid, m/s
    drs_frac: np.ndarray                           # (n_laps, N) share of laps with DRS open at each grid point
    throttle: np.ndarray                           # (n_laps, N)
    brake: np.ndarray                              # (n_laps, N)

    @property
    def times(self) -> np.ndarray:
        return np.array([x[2] for x in self.lap_info])


def build(key: str, n_laps: int = 5) -> Circuit:
    Ls = laps.best_laps(key, n_laps)
    track = geometry.build_track(geometry.mean_line(Ls))

    def on_grid(L, a):
        d = L.dist * (track.length / L.dist[-1])
        return np.interp(track.s, d, a)

    return Circuit(key, track, [(L.driver, L.lap, L.time, L.compound) for L in Ls],
                   np.array([on_grid(L, L.v) for L in Ls]), np.array([on_grid(L, L.drs) for L in Ls]),
                   np.array([on_grid(L, L.throttle) for L in Ls]), np.array([on_grid(L, L.brake) for L in Ls]))


def load_all(keys=None, refresh=False) -> dict[str, Circuit]:
    keys = keys or F.IDENT
    cache = ROOT / "outputs" / "cache" / "circuits.pkl"
    if cache.exists() and not refresh:
        d = pickle.loads(cache.read_bytes())
        if all(k in d for k in keys):
            return {k: d[k] for k in keys}
    d = {k: build(k) for k in keys}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(d))
    return d
