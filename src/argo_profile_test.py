
import os
import sys
import numpy as np
from dynesty import NestedSampler
from dynesty import utils as dyfunc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg
import forward_model as fm

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_FILE = os.path.join(HERE, "..", "figures", "fig4_profile_vs_retrieval.png")
OUT_FILE = os.path.join(HERE, "..", "results", "argo_profile_test.txt")

if len(sys.argv) != 2:
    sys.exit("Usage: python src/argo_profile_test.py path/to/profile.csv")
tab = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
needed = ["pressure_bar", "temperature_K", "H2O", "CO2", "CH4"]
missing = [c for c in needed if c not in tab.dtype.names]
if missing:
    sys.exit("CSV is missing columns: %s (found: %s)" % (missing, tab.dtype.names))
order = np.argsort(tab["pressure_bar"])
P = tab["pressure_bar"][order]
if P[0] > cfg.P_TOP * 3 or P[-1] < cfg.P_BOTTOM / 3:
    print("WARNING: the profile covers %.2g-%.2g bar; the model uses %.2g-%.2g bar. Values "
          "outside the range are held constant at the edge." % (P[0], P[-1], cfg.P_TOP, cfg.P_BOTTOM))


p_mid, dp = fm.pressure_grid()
logp_layers = np.log10(p_mid / 1e5)                       # layer pressures in log10(bar)
logp_in = np.log10(P)
temp = np.interp(logp_layers, logp_in, tab["temperature_K"][order])
vmr = {}
for sp in ("H2O", "CO2", "CH4"):                          # interpolate mixing ratios in log space
    x = np.clip(tab[sp][order], 1e-30, None)
    vmr[sp] = 10.0 ** np.interp(logp_layers, logp_in, np.log10(x))
t_surface = float(np.interp(np.log10(cfg.P_BOTTOM), logp_in, tab["temperature_K"][order]))


wl = cfg.make_wavelength_grid()
truth_flux = fm.emergent_flux(wl, temp, vmr, t_surface, 1.0)
rng = np.random.default_rng(cfg.RNG_SEED)
sigma = truth_flux[np.argmin(np.abs(wl - cfg.LAMBDA_REF))] / cfg.SNR_REF
flux = truth_flux + rng.normal(0.0, sigma, wl.size)


low = np.array([cfg.PRIORS[k][0] for k in cfg.PARAM_NAMES])
high = np.array([cfg.PRIORS[k][1] for k in cfg.PARAM_NAMES])

def prior_transform(u):
    return low + (high - low) * u

def log_like(theta):
    model = fm.forward_flux(theta, wl, h2o_mode="constant")
    return -0.5 * np.sum(((flux - model) / sigma) ** 2 + np.log(2.0 * np.pi * sigma ** 2))

print("Running nested sampling on the profile-based spectrum ...")
sampler = NestedSampler(log_like, prior_transform, len(low), nlive=cfg.NLIVE)
sampler.run_nested(dlogz=cfg.DLOGZ, print_progress=True)
res = sampler.results
samples = dyfunc.resample_equal(res.samples, np.exp(res.logwt - res.logz[-1]))


def column_mean(x):                                       # pressure-weighted mean mixing ratio
    return np.sum(x * dp) / np.sum(dp)

lines = ["profile file: %s" % os.path.basename(sys.argv[1]), "surface T taken from profile: %.1f K" % t_surface, ""]
lines.append("%-8s %-28s %-28s" % ("param", "retrieved median [16,84]", "reference value(s) in the profile"))
q = lambda k: np.percentile(samples[:, k], [16, 50, 84])
lo_, md_, hi_ = q(0); lines.append("%-8s %-28s %-28s" % ("T0", "%.1f [%.1f, %.1f]" % (md_, lo_, hi_), "surface T = %.1f" % t_surface))
for k, sp in ((2, "H2O"), (3, "CO2"), (4, "CH4")):
    lo_, md_, hi_ = q(k)
    ref = "column mean %.2f ; surface-layer %.2f" % (np.log10(column_mean(vmr[sp])), np.log10(vmr[sp][-1]))
    lines.append("%-8s %-28s %-28s" % ("log " + sp, "%.2f [%.2f, %.2f]" % (md_, lo_, hi_), ref))
lo_, md_, hi_ = q(5); lines.append("%-8s %-28s %-28s" % ("R_p", "%.3f [%.3f, %.3f]" % (md_, lo_, hi_), "1.000"))
# fit quality of the constant-abundance model at the posterior median: Konrad et al. (2024) found
# that biased parameters can still come with an acceptable fit, so this number must be read WITH the table
med_model = fm.forward_flux(np.median(samples, axis=0), wl, h2o_mode="constant")
chi2_pp = np.sum(((flux - med_model) / sigma) ** 2) / wl.size
lines += ["", "chi2 per data point of the retrieved constant-abundance model: %.2f (about 1 = statistically acceptable fit)" % chi2_pp]
lines += ["", "A retrieved constant abundance can differ from both numbers: it is the value that",
          "best mimics the spectrum, not a column mean. Differences are the point of this test."]
text = "\n".join(lines)
print("\n" + text)
with open(OUT_FILE, "w") as f:
    f.write(text + "\n")


med = np.median(samples, axis=0)
fig, axes = plt.subplots(1, 4, figsize=(13, 4.5), sharey=True)
t_ret = fm.temperature_profile(p_mid, med[0], med[1])
axes[0].plot(temp, p_mid / 1e5, "tab:red", label="profile (truth)")
axes[0].plot(t_ret, p_mid / 1e5, "tab:blue", label="retrieved T(P)")
axes[0].set_xlabel("temperature [K]"); axes[0].legend(fontsize=8)
for ax, sp, k in zip(axes[1:], ("H2O", "CO2", "CH4"), (2, 3, 4)):
    lo_, md_, hi_ = q(k)
    ax.plot(vmr[sp], p_mid / 1e5, "tab:red", label="profile (truth)")
    ax.axvline(10 ** md_, color="tab:blue", label="retrieved constant")
    ax.axvspan(10 ** lo_, 10 ** hi_, color="tab:blue", alpha=0.25)
    ax.set_xscale("log"); ax.set_xlabel("%s mixing ratio" % sp)
axes[0].set_yscale("log"); axes[0].invert_yaxis(); axes[0].set_ylabel("pressure [bar]")
axes[1].legend(fontsize=8)
fig.savefig(FIG_FILE, dpi=150, bbox_inches="tight")
print("Figure written to", os.path.normpath(FIG_FILE))
