"""Pick and load qualifying laps, and turn a telemetry lap into arrays.

Telemetry positions are in tenths of a metre (the path length of a lap is 10.0 times its integrated distance); speed is km/h.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import fetch as F

NONE = (None, "None")


@dataclass
class Lap:
    circuit: str
    driver: str
    lap: int
    time: float            # official lap time, s
    compound: str
    t: np.ndarray          # s
    dist: np.ndarray       # m, integrated from speed (feed channel)
    v: np.ndarray          # m/s
    x: np.ndarray          # m
    y: np.ndarray          # m
    throttle: np.ndarray
    brake: np.ndarray
    drs: np.ndarray
    gear: np.ndarray


def fastest_valid(circuit: str, session: str = "Qualifying", n: int = 5) -> list[tuple[float, str, int, str]]:
    """The n fastest valid laps, at most one per driver: (time, driver, lap, compound).

    Valid = timing accurate, not deleted for track limits, not an in or out lap.
    """
    ev = F.EVENTS[circuit]
    best: dict[str, tuple[float, int, str]] = {}
    for d in F.drivers(ev, session):
        l = F.laptimes(ev, session, d)
        if not l:
            continue
        for lap, t, acc, dl, pin, pout, comp in zip(l["lap"], l["time"], l["iacc"], l["del"], l["pin"], l["pout"], l["compound"]):
            if t in NONE or not acc or dl or pin not in NONE or pout not in NONE:
                continue
            t = float(t)
            if d not in best or t < best[d][0]:
                best[d] = (t, lap, comp)
    out = sorted((t, d, lap, comp) for d, (t, lap, comp) in best.items())
    return out[:n]


def load_lap(circuit: str, driver: str, lap: int, time: float, compound: str, session: str = "Qualifying") -> Lap:
    tel = F.telemetry(F.EVENTS[circuit], session, driver, lap)["tel"]
    a = lambda k: np.asarray(tel[k], float)
    return Lap(circuit, driver, lap, time, compound, a("time"), a("distance"), a("speed") / 3.6,
               a("x") / 10.0, a("y") / 10.0, a("throttle"), a("brake"), a("drs"), a("gear"))


def best_laps(circuit: str, n: int = 5) -> list[Lap]:
    return [load_lap(circuit, d, lap, t, c) for t, d, lap, c in fastest_valid(circuit, n=n)]
