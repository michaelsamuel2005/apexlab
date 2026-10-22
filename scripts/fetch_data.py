"""Download the 2025 per-lap telemetry used (laptimes for every race and qualifying session, telemetry for the laps used) and write data/MANIFEST.json."""
from apexlab import circuits, fetch
fetch.fetch_all()
circuits.load_all(refresh=True)
