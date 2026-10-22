# apexlab

Physics-based, reduced-order and data-driven modelling of F1 car performance from **public 2025 telemetry**. A differentiable point-mass lap-time simulator, parameter identification with Bayesian posteriors, surrogate and reduced-order models with Sobol sensitivity, and a mixed-effects tyre-degradation model. Everything is tested, validated on data the fit did not see, and reported with its failures.

Read `docs/REPORT.pdf` for the results and `MODEL_CARD.md` for what the model can and cannot be used for.

## Headline results (2025 season, 12 circuits, 24 races)

| Question | Result |
|---|---|
| Simulated vs measured speed on the fitted lap | 3.4 m/s RMS (mean over 12 circuits) |
| Lap time at an unseen circuit, no calibration | 2.1 s mean absolute error (0.2 to 6.0 s) vs 6.4 s for a pace baseline |
| Sensitivity of that error to the fixed assumptions | 1.8 to 2.2 s over eight alternatives |
| GP surrogate of lap time | 0.06 s RMSE from 400 simulations (0.25 s from 50) |
| Sobol indices: PCE-analytic vs 14,336-run Saltelli | agree within 0.003 |
| Bayesian posteriors | 8 of 12 circuits; ClA and μ correlation about -0.98; nominal 90% interval covered 88% of 16 held-out laps (50% interval: 19%) |
| Tyre degradation (soft / medium / hard) | 0.080 / 0.039 / 0.037 % of lap time per lap |
| Tyre model on the last 8 races (trained on first 16) | 0.97 % RMSE vs 1.31 % mean-only |

## What is honest about it

- **Public data only.** No wind-tunnel, CFD, rig or team data, and none is claimed. Source: [TracingInsights/2025](https://github.com/TracingInsights/2025) (Apache-2.0 code; underlying timing data for personal, non-commercial use; unofficial). Raw files are not redistributed; `scripts/fetch_data.py` downloads them and checks them against `data/MANIFEST.json`.
- **Parameters are effective.** A point mass has no load transfer, tyre thermal model or ERS strategy, so the fitted CdA, ClA, μ and power absorb all of that. Six of twelve ClA fits sit on their bound (a seventh just below it) and ClA and μ trade against each other.
- **Verified against closed-form solutions**, not against another lap-time simulator.
- **One tuning step on fitted data:** the curvature-smoothing setting was chosen on three circuits (ITA, GBR, HUN) and applied to all twelve.
- **Built with an AI coding assistant (Claude).** No Formula 1 experience is claimed.

## Layout

```
src/apexlab/  fetch, laps, geometry, physics, circuits, ident, validate, bayes, surrogate, tyres
scripts/      run_all.py and one script per experiment
tests/        12 tests (closed-form checks, DRS, surrogate tools, tyre-data filter)
outputs/      tables/ (CSV, JSON) and figures/
docs/         REPORT.pdf, REPORT.md
```

## Run it

```
pip install -e ".[dev]"
pytest -q                       # 12 tests, a few seconds, no data needed
python scripts/run_all.py       # about 1.5 h on 2 CPUs; the Bayesian fits dominate. The delivered results include posteriors for 8 of the 12 circuits
```

Python 3.10+, JAX (x64), NumPyro, SALib, scikit-learn, statsmodels.
