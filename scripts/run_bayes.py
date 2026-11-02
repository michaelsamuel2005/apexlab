import sys, time, numpy as np, pickle
from pathlib import Path
from apexlab import circuits, bayes, validate
from numpyro.diagnostics import summary
part=int(sys.argv[1]) if len(sys.argv)>1 else 0
CACHE=Path(__file__).resolve().parents[1]/"outputs"/"cache"; CACHE.mkdir(parents=True,exist_ok=True)
cs=circuits.load_all(); fits,_=validate.one_lap_calibration(cs)
pickle.dump(fits,open(CACHE/"fits.pkl","wb"))
keys=list(cs)[part::2] if len(sys.argv)>1 else list(cs); res={}
for k in keys:
    c=cs[k]; t=time.time(); m,s=bayes.prior_from(fits,exclude=k)
    mc=bayes.run(c,m,s,warmup=200,samples=250,chains=2)
    sm=mc.get_samples(group_by_chain=True)
    sd=summary({"lt":sm["log_theta"]},group_by_chain=True)["lt"]
    g=mc.get_samples()
    res[k]=dict(m=m,s=s,log_theta=np.asarray(g["log_theta"]),sigma_t=np.asarray(g["sigma_t"]),sigma_v=np.asarray(g["sigma_v"]),rhat=np.asarray(sd["r_hat"]),ess=np.asarray(sd["n_eff"]),div=int(mc.get_extra_fields()["diverging"].sum()))
    print(k,f"{time.time()-t:.0f}s rhat {res[k]['rhat'].max():.3f} ess {res[k]['ess'].min():.0f} div {res[k]['div']} theta {np.round(np.exp(res[k]['log_theta']).mean(0),2)}",flush=True)
    pickle.dump(res,open(CACHE/f"bayes{part}.pkl","wb"))
