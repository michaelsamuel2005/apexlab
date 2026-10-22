"""Reproduce every result. Order matters (cache files feed later steps). Total run time on 2 CPUs: about 1.5 hours,
almost all of it the Bayesian fits (scripts/run_bayes.py)."""
import subprocess, sys
from pathlib import Path
here = Path(__file__).parent
for s in ["fetch_data.py", "run_ident_validate.py", "run_bayes.py", "run_surrogate.py", "run_tyres.py", "run_budget.py", "make_figures.py"]:
    print("==", s, flush=True)
    subprocess.run([sys.executable, str(here / s)], check=True)
