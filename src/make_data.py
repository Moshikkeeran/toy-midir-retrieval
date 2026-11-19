
import os
import numpy as np
import config as cfg
import forward_model as fm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "..", "data", "synthetic_spectrum.npz")

rng = np.random.default_rng(cfg.RNG_SEED)             
wl = cfg.make_wavelength_grid()


truth_flux = fm.forward_flux(cfg.TRUTH_VECTOR, wl, h2o_mode=cfg.TRUTH_H2O_PROFILE)


i_ref = np.argmin(np.abs(wl - cfg.LAMBDA_REF))
sigma = truth_flux[i_ref] / cfg.SNR_REF
noisy_flux = truth_flux + rng.normal(0.0, sigma, size=wl.size)


chi2_per_point = np.sum(((noisy_flux - truth_flux) / sigma) ** 2) / wl.size

np.savez(DATA_FILE, wavelength=wl, flux=noisy_flux, sigma=sigma, truth_flux=truth_flux,
         h2o_mode=cfg.TRUTH_H2O_PROFILE)

print("Synthetic spectrum written to", os.path.normpath(DATA_FILE))
print("  data points          :", wl.size)
print("  wavelength range     : %.2f - %.2f micron" % (wl[0], wl[-1]))
print("  noise sigma          : %.4f microJy (constant)" % sigma)
print("  S/N at %.1f micron   : %.1f" % (cfg.LAMBDA_REF, truth_flux[i_ref] / sigma))
print("  chi2 per point (truth): %.2f  (expect ~1 +/- 0.11)" % chi2_per_point)
print("  truth H2O profile    :", cfg.TRUTH_H2O_PROFILE)
