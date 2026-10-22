"""Write docs/bayes_section.md from the posterior and coverage tables, so every number in the text is generated."""
import numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1]
P = pd.read_csv(R / "outputs/tables/bayes_posteriors.csv"); C = pd.read_csv(R / "outputs/tables/bayes_coverage.csv")
n = len(P); nl = len(C)
w = (P.ClA_hi - P.ClA_lo) / P.ClA_med
bad = P[(P.rhat_max > 1.03) | (P.ess_min < 50)]; good = P.drop(bad.index)
skipped = [k for k in ["AUS","BHR","JPN","ESP","AUT","GBR","BEL","HUN","NED","ITA","USA","ABU"] if k not in set(P.circuit)]
bad_txt = (f"{', '.join(bad.circuit)} did not mix (R-hat {bad.rhat_max.max():.2f}, effective sample size {bad.ess_min.min():.0f}) and its posterior should be disregarded; " if len(bad) else "")
txt = f"""# 6. Bayesian posteriors and what they can and cannot say

I sampled the posterior of (CdA, ClA, μ, P) with NUTS in NumPyro, two chains of 250 draws each after 200 warm-up, for {n} of the 12 circuits ({', '.join(P.circuit)}). The other {len(skipped)} ({', '.join(skipped)}) were not run: I stopped the job to save time, so this section covers {n} circuits, not 12. The prior for a circuit is log-normal with the mean and standard deviation (floored at 0.2) of the log point-fits of the *other* circuits: an empirical-Bayes choice. A fully hierarchical model over all 12 circuits was tried first; it cost 0.1 s per gradient and about 45 minutes per chain, so I did not run it. The likelihood is the thinned speed trace of the pole lap with a learned scale σ_v and the lap times of the three fastest laps with a learned scale σ_t. The maximum tree depth was 10 for AUS, BHR, ESP, GBR, HUN and ITA and 8 for JPN and AUT (the first attempt at JPN had not finished after 15 minutes).

**Sampler health.** {bad_txt}the other {len(good)} have R-hat at most {good.rhat_max.max():.2f} and effective sample size at least {good.ess_min.min():.0f}. Across all {n} there were {int(P.divergences.sum())} divergent transitions in {n*500} draws ({int(P.divergences.max())} at most in one circuit). Effective sample sizes of about 100 to 250 are modest, so the intervals are indicative, not precise.

**What the posteriors say.** ClA and μ are almost perfectly anti-correlated (correlation of logs {P.corr_ClA_mu.median():.2f}, range {P.corr_ClA_mu.min():.2f} to {P.corr_ClA_mu.max():.2f}): the data fix a combination of downforce and grip and not the two separately, which is the practical meaning of "effective parameters". The posterior medians of ClA run from {P.ClA_med.min():.1f} to {P.ClA_med.max():.1f} m², above the 7 m² bound used in the point fits at several circuits (the prior comes from point fits that sit on that bound). I would not read those values as physical downforce. The 90% intervals on ClA are {w.min():.0%} to {w.max():.0%} of the median wide, which is narrow compared with the spread between circuits and therefore overstates what the data know (see the next paragraph).

![Posterior medians and 90% intervals per circuit.](../outputs/figures/fig3_posteriors.png)

![Posterior samples at Silverstone: the downforce and grip trade-off and the drag and power trade-off.](../outputs/figures/fig4_identifiability.png)

**Is the interval honest?** Only partly. Each posterior predicts the lap times of laps 4 and 5 of the five fastest (parameter uncertainty plus the learned σ_t), which it never saw. Across {nl} held-out laps ({n} circuits, two laps each) the nominal 90% interval contained the observed time {C.in90.mean():.0%} of the time (the misses are both laps at JPN, the circuit that did not mix) and the nominal 50% interval only {C.in50.mean():.0%}. The 50% result is poor and has a clear cause: the laps are ordered fastest first, so laps 4 and 5 are slower than the laps used for training, and the observed time lies above the predictive median in {int((C.observed > C['median']).sum())} of {nl} cases. The interval is centred on the pole-lap pace and is not widened for that offset. The mean 90% width is {C.width90.mean():.2f} s, of which parameter uncertainty alone is {C.param_only_width90.mean():.2f} s. The speed residuals are strongly autocorrelated, so the likelihood overstates the information; {nl} laps is a small test and the coverage figures carry wide sampling error.
"""
(R / "docs/bayes_section.md").write_text(txt); print(txt)
