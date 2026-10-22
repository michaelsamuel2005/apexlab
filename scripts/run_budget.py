"""Assumption sensitivity ("error budget"): refit all circuits under each alternative fixed assumption and report how the
leave-one-circuit-out lap-time error and the fitted parameters move. Writes outputs/tables/assumption_sensitivity.csv."""
import jax, numpy as np, pandas as pd
from apexlab import circuits, ident, physics as P, validate
from apexlab.fetch import ROOT

cs = circuits.load_all()
base = dict(MASS=circuits.MASS_QUALI, RHO=P.RHO, DRS=P.DRS_CUT, KT=P.K_TRAC)
variants = {"base": {}, "mass 815 kg": {"MASS": 815.0}, "mass 845 kg": {"MASS": 845.0}, "air density 1.15": {"RHO": 1.15}, "air density 1.25": {"RHO": 1.25},
            "DRS drag cut 0.15": {"DRS": 0.15}, "DRS drag cut 0.35": {"DRS": 0.35}, "rear traction share 0.45": {"KT": 0.45}, "rear traction share 0.65": {"KT": 0.65}}
rows = []
for name, v in variants.items():
    circuits.MASS_QUALI = v.get("MASS", base["MASS"]); P.RHO = v.get("RHO", base["RHO"]); P.DRS_CUT = v.get("DRS", base["DRS"]); P.K_TRAC = v.get("KT", base["KT"])
    jax.clear_caches()
    fits, A = validate.one_lap_calibration(cs)
    B = validate.leave_one_circuit_out(cs, fits)
    th = np.array(list(fits.values()))
    rows.append(dict(variant=name, fit_speed_rmse=A[A.fit_lap].speed_rmse.mean(), other_laps_dt_mean=A[~A.fit_lap].dt.mean(), loco_mae_s=B.model_abs_dt.mean(),
                     CdA=th[:, 0].mean(), ClA=th[:, 1].mean(), mu=th[:, 2].mean(), P_kW=th[:, 3].mean()))
    print(rows[-1], flush=True)
circuits.MASS_QUALI = base["MASS"]; P.RHO = base["RHO"]; P.DRS_CUT = base["DRS"]; P.K_TRAC = base["KT"]
pd.DataFrame(rows).to_csv(ROOT / "outputs" / "tables" / "assumption_sensitivity.csv", index=False)
