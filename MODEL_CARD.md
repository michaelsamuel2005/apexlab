# Model card: apexlab point-mass lap-time model

**Intended use.** Teaching and portfolio demonstration of model identification, uncertainty quantification, surrogate modelling and validation on public data; exploratory what-if comparisons of the *direction* of parameter effects (more power, more downforce, more grip) at circuits in the 2025 calendar.

**Not for.** Predicting the performance of any real car or team, set-up decisions, car development, race strategy, or any use that relies on the parameters being physical coefficients.

**Model.** Quasi-steady-state point mass on the driven line; friction ellipse; load from downforce; drive limited by power and a fixed share of grip; DRS as a fixed 25 % drag cut on the measured open zones. Four fitted parameters per circuit (CdA, ClA, μ, P); mass 830 kg fixed.

**Data.** 2025 per-lap telemetry for 12 circuits (five fastest qualifying laps each) and clean race laps from all 24 races; see `docs/REPORT.md` section 2.

**Performance** (full detail in the report). Fitted lap: 3.4 m/s speed RMS. Unseen circuit, uncalibrated: 2.1 s mean absolute lap-time error, range 0.2 to 6.0 s. Surrogates: GP 0.06 s RMSE at 400 simulations. Bayesian posteriors: 8 of 12 circuits, one of them not converged.

**Known failure modes.** (1) Slow, tight corners if curvature is over-smoothed or position noise exceeds about 0.3 m after averaging. (2) Bahrain: 6.0 s error when predicted from other circuits. (3) ClA/μ trade-off and ClA hitting its bound at six of twelve circuits. (4) POD reconstruction of the speed trace has kinks at constraint switches (up to 44 m/s pointwise) though lap time is accurate. (5) Polynomial chaos is unusable below about 126 simulations at degree 4.

**Assumptions to challenge.** Mass 830 kg, air density 1.20 kg/m³, rolling resistance 0.010, traction share 0.55, DRS cut 25 %. Varying them moves the headline error between 1.8 and 2.2 s.

**Ethics and licence.** Data are public, unofficial and for personal, non-commercial use under the terms of the source and the underlying timing data. The code is the author's. Built with an AI coding assistant; no affiliation with, or data from, any Formula 1 team.
