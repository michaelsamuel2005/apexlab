"""Download the public data used by apexlab and write a checksum manifest.

Source: github.com/TracingInsights/2025 (per-lap car telemetry and lap data exported from the F1 timing feed through FastF1;
repository Apache-2.0; the underlying F1 data are for personal, non-commercial use; TracingInsights is unofficial and not
associated with the Formula 1 companies). Files are fetched once and cached under data/raw.

    python -m apexlab.fetch            # circuits for identification + all 2025 race lap files
"""
from __future__ import annotations

import hashlib
import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
BASE = "https://raw.githubusercontent.com/TracingInsights/2025/main"

# 2025 calendar in order; short key -> event folder name
EVENTS = {
    "AUS": "Australian Grand Prix", "CHN": "Chinese Grand Prix", "JPN": "Japanese Grand Prix", "BHR": "Bahrain Grand Prix",
    "SAU": "Saudi Arabian Grand Prix", "MIA": "Miami Grand Prix", "EMI": "Emilia Romagna Grand Prix", "MON": "Monaco Grand Prix",
    "ESP": "Spanish Grand Prix", "CAN": "Canadian Grand Prix", "AUT": "Austrian Grand Prix", "GBR": "British Grand Prix",
    "BEL": "Belgian Grand Prix", "HUN": "Hungarian Grand Prix", "NED": "Dutch Grand Prix", "ITA": "Italian Grand Prix",
    "AZE": "Azerbaijan Grand Prix", "SGP": "Singapore Grand Prix", "USA": "United States Grand Prix", "MEX": "Mexico City Grand Prix",
    "BRA": "São Paulo Grand Prix", "LVG": "Las Vegas Grand Prix", "QAT": "Qatar Grand Prix", "ABU": "Abu Dhabi Grand Prix",
}
# permanent circuits used to identify and test the physics model (street circuits excluded: walls, bumps, low grip variation)
IDENT = ["AUS", "BHR", "JPN", "ESP", "AUT", "GBR", "BEL", "HUN", "NED", "ITA", "USA", "ABU"]
_S = requests.Session()


def _url(event: str, *parts: str) -> str:
    return "/".join([BASE, urllib.parse.quote(event)] + [urllib.parse.quote(p) for p in parts])


def get(event: str, *parts: str, binary=False):
    """Fetch (or read from cache) one file of the archive."""
    path = RAW.joinpath(event, *parts)
    if not path.exists():
        r = _S.get(_url(event, *parts), timeout=60)
        if r.status_code != 200:
            raise FileNotFoundError(f"{r.status_code}: {event}/{'/'.join(parts)}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    return path.read_bytes() if binary else json.loads(path.read_text())


def drivers(event: str, session: str) -> list[str]:
    return [d["driver"] for d in get(event, session, "drivers.json")["drivers"]]


def laptimes(event: str, session: str, drv: str) -> dict | None:
    try:
        return get(event, session, drv, "laptimes.json")
    except FileNotFoundError:
        return None


def telemetry(event: str, session: str, drv: str, lap: int) -> dict:
    return get(event, session, drv, f"{lap}_tel.json")


def fetch_all(workers: int = 8) -> None:
    jobs = []
    for ev in EVENTS.values():
        for ses in ("Race", "Qualifying"):
            try:
                for d in drivers(ev, ses):
                    jobs.append((ev, ses, d))
            except FileNotFoundError:
                pass
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda j: laptimes(*j), jobs))
    manifest = {}
    for p in sorted(RAW.rglob("*.json")):
        manifest[str(p.relative_to(RAW))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT / "data" / "MANIFEST.json").write_text(json.dumps(manifest, indent=0))
    print(f"{len(manifest)} files; manifest written")


if __name__ == "__main__":
    fetch_all()
