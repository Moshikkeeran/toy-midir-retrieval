"""
forward_model.py  --  parameters in, mid-infrared emission spectrum out.

WHAT THIS IS: a deliberately simple 1D thermal-emission model of an Earth-like
planet. It has the same ingredients as a real retrieval forward model
(pressure layers, temperature profile, opacities, radiative transfer), but the
opacities are TOY Gaussian bands, not real line-by-line data. See the README.

Pipeline inside forward_flux():
    (T0, kappa, abundances, R_p)
      -> temperature in each layer
      -> optical depth of each layer at each wavelength
      -> emission from each layer, attenuated by the layers above it
      -> add the surface, convert to flux of the planet seen from 10 pc
"""
import numpy as np
import config as cfg

# --- physical constants (SI) -------------------------------------------------
H_PLANCK = 6.62607015e-34   # J s
C_LIGHT = 2.99792458e8      # m / s
K_BOLTZ = 1.380649e-23      # J / K
AMU = 1.66053907e-27        # kg
PARSEC = 3.0856776e16       # m
R_EARTH = 6.371e6           # m
JY = 1.0e-26                # W m^-2 Hz^-1 (one Jansky)


def planck_nu(wavelength_um, temperature):
    """Planck function B_nu(T) in W m^-2 Hz^-1 sr^-1.
    wavelength_um : array, microns.   temperature : array or float, kelvin."""
    wl = np.asarray(wavelength_um) * 1e-6               # micron -> metre
    nu = C_LIGHT / wl                                   # frequency
    x = H_PLANCK * nu / (K_BOLTZ * np.asarray(temperature))
    return 2.0 * H_PLANCK * nu**3 / C_LIGHT**2 / np.expm1(x)   # expm1 = exp(x)-1


# --- TOY OPACITIES -----------------------------------------------------------
# Each band is a Gaussian in wavelength. Band CENTRES are the real positions of
# these molecules' main mid-IR bands. Band STRENGTHS and WIDTHS are rough,
# order-of-magnitude choices (cm^2 per molecule, converted to m^2 below).
# They do NOT depend on pressure or temperature (a real model's would).
def _gauss(wl, centre, sigma):
    return np.exp(-0.5 * ((wl - centre) / sigma) ** 2)

def _sigmoid(wl, centre, width):
    return 1.0 / (1.0 + np.exp(-(wl - centre) / width))

def cross_sections(wl):
    """Return dict species -> cross-section (m^2 / molecule) on wavelength grid wl."""
    cm2 = 1.0e-4  # 1 cm^2 = 1e-4 m^2
    sig = {}
    # H2O: 6.3 micron bending band + rotational band rising steeply beyond ~13 micron
    #      + a tiny flat "continuum" that makes the 8-12 micron window slightly opaque
    sig["H2O"] = cm2 * (5.0e-20 * _gauss(wl, 6.3, 0.30)
                        + 3.0e-20 * _sigmoid(wl, 18.0, 0.7)
                        + 2.0e-24)
    # CO2: the strong 15 micron bending band, plus the very strong 4.3 micron stretching band
    sig["CO2"] = cm2 * (1.0e-19 * _gauss(wl, 15.0, 0.6)
                        + 1.0e-18 * _gauss(wl, 4.3, 0.15))
    # CH4: the 7.66 micron band
    sig["CH4"] = cm2 * (1.0e-19 * _gauss(wl, 7.66, 0.35))
    return sig


# --- ATMOSPHERE STRUCTURE ----------------------------------------------------
def pressure_grid():
    """Return (P_mid, dP) for N_LAYERS layers, both in Pa. Index 0 = TOP layer."""
    edges = np.logspace(np.log10(cfg.P_TOP), np.log10(cfg.P_BOTTOM), cfg.N_LAYERS + 1) * 1e5
    p_mid = np.sqrt(edges[:-1] * edges[1:])      # geometric mean of layer edges
    dp = np.diff(edges)                           # pressure thickness of each layer
    return p_mid, dp

def temperature_profile(p_mid, t0, kappa):
    """T(P) = T0 (P / P_bottom)^kappa, but never colder than the stratosphere floor."""
    p_bottom = cfg.P_BOTTOM * 1e5
    return np.maximum(t0 * (p_mid / p_bottom) ** kappa, cfg.T_STRATOSPHERE)

def h2o_vmr_profile(p_mid, log_h2o, mode):
    """Volume mixing ratio of H2O in each layer."""
    x0 = 10.0 ** log_h2o
    if mode == "constant":
        return np.full_like(p_mid, x0)
    if mode == "falling":     # drops as (P/0.3 bar)^3 above 0.3 bar
        p_c = 0.3e5
        return x0 * np.minimum(1.0, (p_mid / p_c) ** 3)
    raise ValueError("unknown H2O profile mode: %s" % mode)


# --- THE FORWARD MODEL -------------------------------------------------------
def forward_flux(theta, wl, h2o_mode="constant", return_brightness=False):
    """
    theta = [T0, kappa, log_H2O, log_CO2, log_CH4, R_p]
    wl    = wavelength grid in microns
    Returns flux density of the whole planet seen from 10 pc, in microJansky.
    """
    t0, kappa, log_h2o, log_co2, log_ch4, r_p = theta
    p_mid, dp = pressure_grid()
    temp = temperature_profile(p_mid, t0, kappa)                # (n_layers,)

    # number of molecules per m^2 of air in each layer (all species share this)
    #   mass per area = dP / g ;  molecules of air = mass / (mu * amu)
    air_column = dp / (cfg.GRAVITY * cfg.MEAN_MOL_WEIGHT * AMU)  # (n_layers,)

    sig = cross_sections(wl)
    vmr = {"H2O": h2o_vmr_profile(p_mid, log_h2o, h2o_mode),
           "CO2": np.full_like(p_mid, 10.0 ** log_co2),
           "CH4": np.full_like(p_mid, 10.0 ** log_ch4)}

    # optical depth of each layer at each wavelength: sum over species of
    #   (vmr * air_column) * cross_section        -> shape (n_layers, n_wl)
    tau = np.zeros((p_mid.size, wl.size))
    for s in sig:
        tau += (vmr[s] * air_column)[:, None] * sig[s][None, :]

    mu = cfg.MU_DIFFUSIVITY
    trans_layer = np.exp(-tau / mu)                              # transmission of each layer
    # optical depth of everything ABOVE layer i = cumulative sum minus the layer itself.
    # (Working with sums of tau, not products of transmissions, avoids 0/0 when a band
    #  is so opaque that the transmission underflows to zero.)
    tau_above = np.cumsum(tau, axis=0) - tau
    above = np.exp(-tau_above / mu)
    # radiance from each layer: Planck(T_i) * emissivity (1 - transmission) * what is above
    b_layer = planck_nu(wl[None, :], temp[:, None])
    rad_layers = (b_layer * (1.0 - trans_layer) * above).sum(axis=0)
    # surface: blackbody at T0, attenuated by the whole atmosphere
    rad_surface = planck_nu(wl, t0) * np.exp(-tau.sum(axis=0) / mu)
    radiance = rad_layers + rad_surface                          # W m^-2 Hz^-1 sr^-1

    # flux of the planet at 10 pc:  F = pi * I * (R_p / d)^2
    solid = (r_p * R_EARTH / (cfg.DISTANCE_PC * PARSEC)) ** 2
    flux_ujy = np.pi * radiance * solid / JY * 1e6               # microJansky

    if return_brightness:                                        # brightness temperature of the
        return flux_ujy, radiance                                # radiance, handy for plotting
    return flux_ujy


def brightness_temperature(wl, radiance):
    """Temperature of the blackbody that would emit this radiance at each wavelength."""
    wlm = wl * 1e-6
    nu = C_LIGHT / wlm
    return H_PLANCK * nu / K_BOLTZ / np.log1p(2 * H_PLANCK * nu**3 / C_LIGHT**2 / radiance)
