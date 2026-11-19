
import os
import numpy as np
from dynesty import NestedSampler
from dynesty import utils as dyfunc
import config as cfg
import forward_model as fm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "..", "data", "synthetic_spectrum.npz")
OUT_FILE = os.path.join(HERE, "..", "results", "posterior.npz")
SUMMARY_FILE = os.path.join(HERE, "..", "results", "summary.txt")


d = np.load(DATA_FILE)
wl, flux, sigma = d["wavelength"], d["flux"], float(d["sigma"])


low = np.array([cfg.PRIORS[k][0] for k in cfg.PARAM_NAMES])
high = np.array([cfg.PRIORS[k][1] for k in cfg.PARAM_NAMES])
ndim = len(cfg.PARAM_NAMES)

def prior_transform(u):
    """dynesty works in a 'unit cube': every parameter is a number u in [0, 1].
    This function maps u onto the physical prior range. Because the abundance
    parameters are log10(mixing ratio) with a UNIFORM prior, the priors on the
    mixing ratios themselves are LOG-UNIFORM (equal probability per decade)."""
    return low + (high - low) * u



def log_likelihood(theta):
    model = fm.forward_flux(theta, wl, h2o_mode="constant")
    # Gaussian likelihood: -1/2 * sum[ (data - model)^2 / sigma^2 + log(2 pi sigma^2) ]
    return -0.5 * np.sum(((flux - model) / sigma) ** 2 + np.log(2.0 * np.pi * sigma ** 2))


print("Running dynesty: %d parameters, %d live points ..." % (ndim, cfg.NLIVE))
sampler = NestedSampler(log_likelihood, prior_transform, ndim, nlive=cfg.NLIVE)
sampler.run_nested(dlogz=cfg.DLOGZ, print_progress=True)
res = sampler.results


weights = np.exp(res.logwt - res.logz[-1])
samples = dyfunc.resample_equal(res.samples, weights)
logz, logzerr = float(res.logz[-1]), float(res.logzerr[-1])
np.savez(OUT_FILE, samples=samples, logz=logz, logzerr=logzerr,
         param_names=np.array(cfg.PARAM_NAMES))


lines = []
lines.append("log evidence ln Z = %.2f +/- %.2f   (%d posterior samples)" % (logz, logzerr, len(samples)))
lines.append("")
lines.append("%-9s %9s %9s %9s %9s   %s" % ("param", "truth", "median", "-1sig", "+1sig", "truth inside"))
for k, name in enumerate(cfg.PARAM_NAMES):
    p2, p16, p50, p84, p97 = np.percentile(samples[:, k], [2.5, 16, 50, 84, 97.5])
    truth = cfg.TRUTH_VECTOR[k]
    if p16 <= truth <= p84:
        flag = "68% interval"
    elif p2 <= truth <= p97:
        flag = "95% interval only"
    else:
        flag = "OUTSIDE 95%"
    lines.append("%-9s %9.3f %9.3f %9.3f %9.3f   %s" % (name, truth, p50, p50 - p16, p84 - p50, flag))
lines.append("")
lines.append("Note: with 6 parameters, one or two truths falling outside the 68% interval is")
lines.append("normal. Only a truth outside the 95% interval deserves investigation.")
text = "\n".join(lines)
print("\n" + text)
with open(SUMMARY_FILE, "w") as f:
    f.write(text + "\n")
