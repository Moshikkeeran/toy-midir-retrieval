
import os
import numpy as np
from dynesty import NestedSampler
from dynesty import utils as dyfunc
import config as cfg
import forward_model as fm

CH4_TRUTHS = [-7.0, -6.0, -5.0]         

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(HERE, "..", "results", "detection_test.txt")
wl = cfg.make_wavelength_grid()
i_ref = np.argmin(np.abs(wl - cfg.LAMBDA_REF))
names = cfg.PARAM_NAMES
low = np.array([cfg.PRIORS[k][0] for k in names])
high = np.array([cfg.PRIORS[k][1] for k in names])
CH4_INDEX = names.index("log_CH4")
NO_CH4 = -30.0                           


def run(flux, sigma, with_ch4):
    """One nested-sampling run. Returns ln Z and its error."""
    keep = np.array([i for i in range(len(names)) if with_ch4 or i != CH4_INDEX])
    lo, hi = low[keep], high[keep]

    def prior_transform(u):
        return lo + (hi - lo) * u

    def log_like(x):
        theta = np.empty(len(names))
        theta[keep] = x
        if not with_ch4:
            theta[CH4_INDEX] = NO_CH4
        model = fm.forward_flux(theta, wl, h2o_mode="constant")
        return -0.5 * np.sum(((flux - model) / sigma) ** 2 + np.log(2.0 * np.pi * sigma ** 2))

    sampler = NestedSampler(log_like, prior_transform, len(keep), nlive=cfg.NLIVE)
    sampler.run_nested(dlogz=cfg.DLOGZ, print_progress=False)
    r = sampler.results
    return float(r.logz[-1]), float(r.logzerr[-1])


def jeffreys(log10k):
    """Jeffreys scale as in Konrad et al. (2024), Table 1 (support for the model WITH CH4)."""
    if log10k < 0:
        return "support for model without CH4"
    for upper, label in ((0.5, "very weak"), (1.0, "substantial"), (2.0, "strong")):
        if log10k < upper:
            return label
    return "decisive"

lines = ["%-14s %12s %12s %9s %9s  %s" % ("true log CH4", "lnZ with", "lnZ without", "ln B", "log10 K", "Jeffreys")]
for log_ch4 in CH4_TRUTHS:
    truth = cfg.TRUTH_VECTOR.copy()
    truth[CH4_INDEX] = log_ch4
    rng = np.random.default_rng(cfg.RNG_SEED)
    model = fm.forward_flux(truth, wl, h2o_mode="constant")
    sigma = model[i_ref] / cfg.SNR_REF
    flux = model + rng.normal(0.0, sigma, wl.size)
    print("true log CH4 = %.1f: running both models ..." % log_ch4)
    z1, e1 = run(flux, sigma, True)
    z0, e0 = run(flux, sigma, False)
    ln_b = z1 - z0
    lines.append("%-14.1f %7.2f+-%.2f %7.2f+-%.2f %9.2f %9.2f  %s"
                 % (log_ch4, z1, e1, z0, e0, ln_b, ln_b / np.log(10.0), jeffreys(ln_b / np.log(10.0))))

lines += ["", "ln B = ln Z(with CH4) - ln Z(without CH4). Toy opacities: a demonstration of the",
          "method, not a statement about LIFE sensitivity."]
text = "\n".join(lines)
print("\n" + text)
with open(OUT_FILE, "w") as f:
    f.write(text + "\n")
