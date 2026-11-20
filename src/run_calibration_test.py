"""
run_calibration_test.py  --  are the posteriors CALIBRATED?  (a small version of Gebhard et al. 2025, Fig. 5)

Usage (from the project folder):
    python src/run_calibration_test.py                          # N=20 noise seeds, correct noise assumption
    python src/run_calibration_test.py --sigma-factor 0.5       # retrieval ASSUMES half the true noise
    python src/run_calibration_test.py --n 30 --nlive 250 --truth-from-prior

"""
import argparse
import os
import numpy as np
from dynesty import NestedSampler
from dynesty import utils as dyfunc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg
import forward_model as fm

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20, help="number of repetitions")
ap.add_argument("--nlive", type=int, default=250, help="live points per retrieval")
ap.add_argument("--sigma-factor", type=float, default=1.0, help="assumed noise = factor x true noise")
ap.add_argument("--truth-from-prior", action="store_true", help="draw a new truth from the prior each time")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
tag = "sf%.2f_%s" % (args.sigma_factor, "prior" if args.truth_from_prior else "fixed")
OUT_NPZ = os.path.join(HERE, "..", "results", "calibration_%s.npz" % tag)
OUT_FIG = os.path.join(HERE, "..", "figures", "fig5_calibration_%s.png" % tag)

wl = cfg.make_wavelength_grid()
i_ref = np.argmin(np.abs(wl - cfg.LAMBDA_REF))
low = np.array([cfg.PRIORS[k][0] for k in cfg.PARAM_NAMES])
high = np.array([cfg.PRIORS[k][1] for k in cfg.PARAM_NAMES])
ndim = len(low)
rng_master = np.random.default_rng(cfg.RNG_SEED + 1000)

ranks, truths = [], []
for i in range(args.n):
    rng = np.random.default_rng(rng_master.integers(1 << 31))
    truth = low + (high - low) * rng.random(ndim) if args.truth_from_prior else cfg.TRUTH_VECTOR.copy()
    model = fm.forward_flux(truth, wl, h2o_mode="constant")
    sigma_true = model[i_ref] / cfg.SNR_REF
    flux = model + rng.normal(0.0, sigma_true, wl.size)
    sigma = args.sigma_factor * sigma_true                     # the noise level the retrieval ASSUMES

    def prior_transform(u):
        return low + (high - low) * u

    def log_like(theta):
        m = fm.forward_flux(theta, wl, h2o_mode="constant")
        return -0.5 * np.sum(((flux - m) / sigma) ** 2 + np.log(2.0 * np.pi * sigma ** 2))

    sampler = NestedSampler(log_like, prior_transform, ndim, nlive=args.nlive)
    sampler.run_nested(dlogz=cfg.DLOGZ, print_progress=False)
    res = sampler.results
    samples = dyfunc.resample_equal(res.samples, np.exp(res.logwt - res.logz[-1]))
    ranks.append((samples < truth[None, :]).mean(axis=0))      # rank of the truth, per parameter
    truths.append(truth)
    print("run %d/%d done" % (i + 1, args.n), flush=True)

ranks = np.array(ranks)
np.savez(OUT_NPZ, ranks=ranks, truths=np.array(truths), sigma_factor=args.sigma_factor)

# ---- summary: how often does the truth fall inside the central 68% / 95% interval? -------------
inside68 = ((ranks > 0.16) & (ranks < 0.84)).mean(axis=0)
inside95 = ((ranks > 0.025) & (ranks < 0.975)).mean(axis=0)
lines = ["N = %d runs, assumed noise = %.2f x true noise, truth: %s" % (args.n, args.sigma_factor, "from prior" if args.truth_from_prior else "fixed"),
         "", "%-9s %12s %12s" % ("param", "in 68% int.", "in 95% int."),]
for k, name in enumerate(cfg.PARAM_NAMES):
    lines.append("%-9s %11.0f%% %11.0f%%" % (name, 100 * inside68[k], 100 * inside95[k]))
err = 100 * np.sqrt(0.68 * 0.32 / args.n)
lines += ["", "Calibrated posteriors: about 68% and 95%. With N = {} the counting uncertainty on a".format(args.n),
          "fraction near 68% is roughly +/- {:.0f}%, so only large departures mean anything.".format(err)]
text = "\n".join(lines)
print("\n" + text)
with open(os.path.join(HERE, "..", "results", "calibration_%s.txt" % tag), "w") as f:
    f.write(text + "\n")

# ---- P-P plot ---------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), sharex=True, sharey=True)
x = np.linspace(0, 1, 200)
for k, ax in enumerate(axes.ravel()):
    r = np.sort(ranks[:, k])
    ax.step(np.concatenate([[0], r, [1]]), np.concatenate([[0], np.arange(1, len(r) + 1) / len(r), [1]]), where="post")
    ax.plot(x, x, "k--", lw=0.8)
    ax.set_title(cfg.PARAM_NAMES[k], fontsize=9)
for ax in axes[-1]:
    ax.set_xlabel("uniform CDF")
for ax in axes[:, 0]:
    ax.set_ylabel("ECDF of truth ranks")
fig.suptitle("P-P plot (N=%d, assumed noise = %.2f x true)" % (args.n, args.sigma_factor), fontsize=10)
fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print("Figure written to", os.path.normpath(OUT_FIG))
