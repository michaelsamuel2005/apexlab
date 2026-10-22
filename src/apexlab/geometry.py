"""Circuit geometry (arc length, curvature) from the driven line of a telemetry lap.

The line is the measured x, y of a fast lap, smoothed with a periodic spline. Curvature therefore belongs to the line that
car actually drove (not the centreline), which is what a point-mass simulation needs. Position noise is about 1 to 2 m.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.ndimage import gaussian_filter1d

from .laps import Lap


@dataclass
class Track:
    circuit: str
    s: np.ndarray        # m, uniform grid, length N
    ds: float
    kappa: np.ndarray    # 1/m, absolute curvature on the grid
    x: np.ndarray
    y: np.ndarray
    length: float


def mean_line(laps: list[Lap], n: int = 2000) -> Lap:
    """Average the driven line of several fast laps on normalised distance; returns a Lap carrying the mean x, y of the first lap's other channels.

    Averaging n laps cuts the white position noise by sqrt(n); the lap-to-lap spread of the line itself is about 1 to 2 m.
    """
    g = np.linspace(0, 1, n + 1)
    xs = np.mean([np.interp(g, l.dist / l.dist[-1], l.x) for l in laps], axis=0)
    ys = np.mean([np.interp(g, l.dist / l.dist[-1], l.y) for l in laps], axis=0)
    ref = laps[0]
    out = Lap(**{**ref.__dict__})
    out.x, out.y = xs, ys
    out.dist = g * np.mean([l.dist[-1] for l in laps])
    return out


def build_track(lap: Lap, ds: float = 5.0, sigma: float = 0.3, median_window: int = 7, smooth_m: float = 10.0) -> Track:
    """Smooth the driven line with a periodic cubic spline and sample curvature on a uniform arc-length grid."""
    x, y = lap.x.copy(), lap.y.copy()
    # close the loop: replace the last point by the first
    x[-1], y[-1] = x[0], y[0]
    keep = np.r_[True, np.hypot(np.diff(x), np.diff(y)) > 1e-6]
    x, y = x[keep], y[keep]
    seg = np.hypot(np.diff(x), np.diff(y))
    u = np.r_[0, np.cumsum(seg)]
    tck, _ = splprep([x, y], u=u / u[-1], s=len(x) * sigma**2, per=True)
    uu = np.linspace(0, 1, 20001)
    xs, ys = splev(uu, tck)
    L = np.r_[0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]
    length = L[-1]
    n = int(round(length / ds))
    s = np.arange(n) * (length / n)
    ug = np.interp(s, L, uu)
    dx, dy = splev(ug, tck, der=1)
    ddx, ddy = splev(ug, tck, der=2)
    kappa = (dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-12)
    xg, yg = splev(ug, tck)
    if median_window > 1:  # suppress isolated curvature spikes caused by position noise (periodic median filter)
        k = np.abs(kappa)
        h = median_window // 2
        pad = np.r_[k[-h:], k, k[:h]]
        kappa = np.median(np.lib.stride_tricks.sliding_window_view(pad, median_window), axis=1)
    if smooth_m > 0:  # Gaussian smoothing of curvature along arc length (periodic), scale about one corner feature
        kappa = gaussian_filter1d(np.abs(kappa), smooth_m / (length / n), mode="wrap")
    return Track(lap.circuit, s, length / n, np.abs(kappa), np.asarray(xg), np.asarray(yg), length)


def measured_speed_on_grid(lap: Lap, track: Track) -> np.ndarray:
    """Measured speed (m/s) resampled onto the track grid; lap distance is rescaled to the spline arc length."""
    d = lap.dist * (track.length / lap.dist[-1])
    return np.interp(track.s, d, lap.v)
