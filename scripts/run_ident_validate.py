"""Point fits per circuit (pole lap), scoring on the other four laps, and leave-one-circuit-out validation."""
import pickle
import pandas as pd
from apexlab import circuits, validate
from apexlab.fetch import ROOT
cs = circuits.load_all(); fits, A = validate.one_lap_calibration(cs)
pickle.dump(fits, open(ROOT / "outputs/cache/fits.pkl", "wb"))
A.to_csv(ROOT / "outputs/tables/lap_scores.csv", index=False)
pd.DataFrame({k: v for k, v in fits.items()}, index=["CdA", "ClA", "mu", "P_kW"]).T.to_csv(ROOT / "outputs/tables/point_fits.csv")
validate.leave_one_circuit_out(cs, fits).to_csv(ROOT / "outputs/tables/loco.csv", index=False)
print(A.groupby("fit_lap")[["speed_rmse", "dt"]].mean())
