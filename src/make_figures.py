
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg
import forward_model as fm

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")
DATA_FILE = os.path.join(HERE, "..", "data", "synthetic_spectrum.npz")
POST_FILE = os.path.join(HERE, "..", "results", "posterior.npz")


def pair_plot(samples, labels, truth):
    """Minimal fallback corner plot, used only if the 'corner' package is missing."""
    n = samples.shape[1]
    fig, axes = plt.subplots(n, n, figsize=(2.0 * n, 2.0 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off"); continue
            if i == j:
                ax.hist(samples[:, i], bins=30, color="0.4")
                ax.axvline(truth[i], color="tab:red")
                ax.set_yticks([])
            else:
                ax.hist2d(samples[:, j], samples[:, i], bins=30, cmap="Greys")
                ax.axvline(truth[j], color="tab:red", lw=0.8)
                ax.axhline(truth[i], color="tab:red", lw=0.8)
            if i == n - 1: ax.set_xlabel(labels[j], fontsize=8)
            else: ax.set_xticklabels([])
            if j == 0 and i > 0: ax.set_ylabel(labels[i], fontsize=8)
            elif j > 0: ax.set_yticklabels([])
            ax.tick_params(labelsize=6)
    return fig


def fig_corner(samples):
    try:
        import corner
        fig = corner.corner(samples, labels=cfg.PARAM_LABELS, truths=cfg.TRUTH_VECTOR,
                            quantiles=[0.16, 0.5, 0.84], show_titles=True,
                            title_kwargs={"fontsize": 9}, truth_color="tab:red")
    except ImportError:
        print("  (package 'corner' not found, using the simple fallback plot)")
        fig = pair_plot(samples, cfg.PARAM_LABELS, cfg.TRUTH_VECTOR)
    fig.savefig(os.path.join(FIG_DIR, "fig1_corner.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_spectrum(samples, d):
    wl, flux, sigma, truth_flux = d["wavelength"], d["flux"], float(d["sigma"]), d["truth_flux"]
    rng = np.random.default_rng(0)
    pick = samples[rng.choice(len(samples), size=300, replace=False)]
    # posterior-predictive: run the forward model for 300 posterior samples
    fluxes, tbs = [], []
    for t in pick:
        f, rad = fm.forward_flux(t, wl, h2o_mode="constant", return_brightness=True)
        fluxes.append(f); tbs.append(fm.brightness_temperature(wl, rad))
    fluxes, tbs = np.array(fluxes), np.array(tbs)
    lo, med, hi = np.percentile(fluxes, [16, 50, 84], axis=0)
    tlo, tmed, thi = np.percentile(tbs, [16, 50, 84], axis=0)
    # brightness temperature of the true (noiseless) model
    tf, trad = fm.forward_flux(cfg.TRUTH_VECTOR, wl, h2o_mode=cfg.TRUTH_H2O_PROFILE, return_brightness=True)
    ttruth = fm.brightness_temperature(wl, trad)

    fig, (ax, axt, axr) = plt.subplots(3, 1, figsize=(9, 9), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 2, 1]})
    ax.errorbar(wl, flux, yerr=sigma, fmt=".", color="0.5", ms=3, lw=0.6,
                label="synthetic data (S/N=%d at %.1f um)" % (cfg.SNR_REF, cfg.LAMBDA_REF))
    ax.plot(wl, truth_flux, color="tab:red", lw=1.2, label="truth (noiseless)")
    ax.plot(wl, med, color="tab:blue", lw=1.2, label="posterior median")
    ax.fill_between(wl, lo, hi, color="tab:blue", alpha=0.3, label="posterior 16-84%")
    ax.set_ylabel(r"flux at 10 pc [$\mu$Jy]"); ax.legend(fontsize=8)

    # brightness temperature = how bright the planet looks, expressed as a temperature.
    # The open window (~8.5-12.5 um) shows the warm surface; saturated bands show the cold upper air.
    axt.plot(wl, ttruth, color="tab:red", lw=1.2, label="truth")
    axt.plot(wl, tmed, color="tab:blue", lw=1.2, label="posterior median")
    axt.fill_between(wl, tlo, thi, color="tab:blue", alpha=0.3)
    axt.set_ylabel("brightness temp. [K]"); axt.legend(fontsize=8)

    axr.axhline(0, color="k", lw=0.6)
    axr.plot(wl, (flux - med) / sigma, ".", color="0.4", ms=3)
    axr.set_ylabel("(data-model)/sigma"); axr.set_xlabel("wavelength [micron]")
    fig.savefig(os.path.join(FIG_DIR, "fig2_spectrum.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_why_midir():
    """Planet (288 K) vs Sun-like star (5772 K), both as blackbodies, in the mid-IR.
    Thermal emission only: at visible wavelengths a planet is seen in REFLECTED light,
    which this simple plot does not model."""
    wl = np.linspace(3.0, 25.0, 600)
    r_sun = 6.957e8
    d = 10 * fm.PARSEC
    planet = np.pi * fm.planck_nu(wl, 288.0) * (fm.R_EARTH / d) ** 2 / fm.JY * 1e6
    star = np.pi * fm.planck_nu(wl, 5772.0) * (r_sun / d) ** 2 / fm.JY * 1e6
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    a1.semilogy(wl, star, color="tab:orange", label="Sun-like star (5772 K blackbody)")
    a1.semilogy(wl, planet, color="tab:blue", label="Earth-like planet (288 K blackbody)")
    a1.set_ylabel(r"flux at 10 pc [$\mu$Jy]"); a1.legend(fontsize=8, loc="center right")
    contrast = planet / star
    a2.semilogy(wl, contrast, color="k")
    a2.set_ylim(1e-9, 1e-5)
    a2.set_ylabel("thermal contrast: planet / star"); a2.set_xlabel("wavelength [micron]")
    for ax in (a1, a2):
        ax.axvspan(6, 16, color="tab:green", alpha=0.15)
        ax.axvspan(4, 18.5, color="tab:green", alpha=0.07)
        ax.set_xlim(3, 25)
    # approximate positions of the main band centres (standard spectroscopy values)
    for name, c in [("H$_2$O 6.3", 6.3), ("CH$_4$ 7.7", 7.66), ("O$_3$ 9.6", 9.6), ("CO$_2$ 15", 15.0)]:
        a2.axvline(c, color="0.4", ls=":", lw=0.9)
        a2.text(c - 0.15, 1.3e-9, name, rotation=90, fontsize=8, va="bottom", ha="right")
    wien = 2897.8 / 288.0
    a1.axvline(wien, color="tab:blue", ls="--", lw=0.8)
    a1.text(wien + 0.3, a1.get_ylim()[0] * 3, "288 K blackbody peaks at ~%.1f um (Wien)" % wien,
            fontsize=8, color="tab:blue")
    i10 = np.argmin(np.abs(wl - 10.0))
    a2.annotate("contrast ~ %.0e at 10 um" % contrast[i10], xy=(10, contrast[i10]),
                xytext=(17, 2e-8), fontsize=8, arrowprops=dict(arrowstyle="->", color="0.3"))
    a1.text(0.02, 0.95, "dark green: LIFE requirement 6-16 um   light green: goal 4-18.5 um",
            transform=a1.transAxes, fontsize=8, va="top")
    fig.savefig(os.path.join(FIG_DIR, "fig3_why_midir.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Making fig3 (why mid-IR) ...")
    fig_why_midir()
    if os.path.exists(POST_FILE) and os.path.exists(DATA_FILE):
        samples = np.load(POST_FILE)["samples"]
        d = np.load(DATA_FILE)
        print("Making fig1 (corner) ...")
        fig_corner(samples)
        print("Making fig2 (spectrum) ...")
        fig_spectrum(samples, d)
    else:
        print("No posterior found yet: run make_data.py and run_retrieval.py first.")
    print("Figures are in", os.path.normpath(FIG_DIR))
