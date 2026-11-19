
import numpy as np
import config as cfg


H_PLANCK = 6.62607015e-34   
C_LIGHT = 2.99792458e8      
K_BOLTZ = 1.380649e-23      
AMU = 1.66053907e-27        
PARSEC = 3.0856776e16       
R_EARTH = 6.371e6           
JY = 1.0e-26                


def planck_nu(wavelength_um, temperature):
    """Planck function B_nu(T) in W m^-2 Hz^-1 sr^-1.
    wavelength_um : array, microns.   temperature : array or float, kelvin."""
    wl = np.asarray(wavelength_um) * 1e-6               
    nu = C_LIGHT / wl                                   
    x = H_PLANCK * nu / (K_BOLTZ * np.asarray(temperature))
    return 2.0 * H_PLANCK * nu**3 / C_LIGHT**2 / np.expm1(x)   


def _gauss(wl, centre, sigma):
    return np.exp(-0.5 * ((wl - centre) / sigma) ** 2)

def _sigmoid(wl, centre, width):
    return 1.0 / (1.0 + np.exp(-(wl - centre) / width))

def cross_sections(wl):
    """Return dict species -> cross-section (m^2 / molecule) on wavelength grid wl."""
    cm2 = 1.0e-4  
    sig = {}

    sig["H2O"] = cm2 * (5.0e-20 * _gauss(wl, 6.3, 0.30)
                        + 3.0e-20 * _sigmoid(wl, 18.0, 0.7)
                        + 2.0e-24)
    
    sig["CO2"] = cm2 * (1.0e-19 * _gauss(wl, 15.0, 0.6)
                        + 1.0e-18 * _gauss(wl, 4.3, 0.15))
    
    sig["CH4"] = cm2 * (1.0e-19 * _gauss(wl, 7.66, 0.35))
    return sig


# --- ATMOSPHERE STRUCTURE ----------------------------------------------------
def pressure_grid():
    """Return (P_mid, dP) for N_LAYERS layers, both in Pa. Index 0 = TOP layer."""
    edges = np.logspace(np.log10(cfg.P_TOP), np.log10(cfg.P_BOTTOM), cfg.N_LAYERS + 1) * 1e5
    p_mid = np.sqrt(edges[:-1] * edges[1:])      
    dp = np.diff(edges)                           
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
    if mode == "falling":     
        p_c = 0.3e5
        return x0 * np.minimum(1.0, (p_mid / p_c) ** 3)
    raise ValueError("unknown H2O profile mode: %s" % mode)


# --- THE FORWARD MODEL -------------------------------------------------------
def emergent_flux(wl, temp, vmr, t_surface, r_p, return_brightness=False):
    """
    Radiative transfer for ANY atmosphere defined layer by layer.
      wl        : wavelength grid, microns
      temp      : temperature of each layer, K            (length N_LAYERS, top first)
      vmr       : dict {"H2O": array, "CO2": array, "CH4": array} of volume mixing ratios per layer
      t_surface : surface temperature, K
      r_p       : planet radius in Earth radii
    Returns the flux density of the whole planet seen from 10 pc, in microJansky.
    forward_flux() below builds `temp` and `vmr` from a few parameters and calls this;
    argo_profile_test.py builds them from a chemistry-model profile instead.
    """
    p_mid, dp = pressure_grid()


    air_column = dp / (cfg.GRAVITY * cfg.MEAN_MOL_WEIGHT * AMU)  

    sig = cross_sections(wl)

    tau = np.zeros((p_mid.size, wl.size))
    for s in sig:
        tau += (vmr[s] * air_column)[:, None] * sig[s][None, :]

    mu = cfg.MU_DIFFUSIVITY
    trans_layer = np.exp(-tau / mu)                             
    
    tau_above = np.cumsum(tau, axis=0) - tau
    above = np.exp(-tau_above / mu)
    
    b_layer = planck_nu(wl[None, :], temp[:, None])
    rad_layers = (b_layer * (1.0 - trans_layer) * above).sum(axis=0)
   
    rad_surface = planck_nu(wl, t_surface) * np.exp(-tau.sum(axis=0) / mu)
    radiance = rad_layers + rad_surface                          

    
    solid = (r_p * R_EARTH / (cfg.DISTANCE_PC * PARSEC)) ** 2
    flux_ujy = np.pi * radiance * solid / JY * 1e6               

    if return_brightness:                                         
        return flux_ujy, radiance                                
    return flux_ujy


def forward_flux(theta, wl, h2o_mode="constant", return_brightness=False):
    """
    theta = [T0, kappa, log_H2O, log_CO2, log_CH4, R_p]
    wl    = wavelength grid in microns
    Returns flux density of the whole planet seen from 10 pc, in microJansky.
    """
    t0, kappa, log_h2o, log_co2, log_ch4, r_p = theta
    p_mid, dp = pressure_grid()
    temp = temperature_profile(p_mid, t0, kappa)                # (n_layers,)
    vmr = {"H2O": h2o_vmr_profile(p_mid, log_h2o, h2o_mode),
           "CO2": np.full_like(p_mid, 10.0 ** log_co2),
           "CH4": np.full_like(p_mid, 10.0 ** log_ch4)}
    return emergent_flux(wl, temp, vmr, t0, r_p, return_brightness=return_brightness)


def brightness_temperature(wl, radiance):
    """Temperature of the blackbody that would emit this radiance at each wavelength."""
    wlm = wl * 1e-6
    nu = C_LIGHT / wlm
    return H_PLANCK * nu / K_BOLTZ / np.log1p(2 * H_PLANCK * nu**3 / C_LIGHT**2 / radiance)
