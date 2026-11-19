"""
config.py  --  every number in the project lives here.

"""
import numpy as np


LAMBDA_MIN = 4.0        
LAMBDA_MAX = 18.5       
R_SPEC = 100            # resolving power R = lambda / d_lambda

def make_wavelength_grid():
    """Log-spaced grid: each point is (1 + 1/R) times the previous one.
    On such a grid R = lambda/d_lambda is the same at every wavelength."""
    n = int(np.ceil(np.log(LAMBDA_MAX / LAMBDA_MIN) / np.log(1.0 + 1.0 / R_SPEC)))
    return LAMBDA_MIN * (1.0 + 1.0 / R_SPEC) ** np.arange(n + 1)

N_LAYERS = 50           # number of atmospheric layers
P_TOP = 1.0e-3          # bar, top of the model atmosphere
P_BOTTOM = 1.0          # bar, surface pressure (fixed in this toy)
GRAVITY = 9.81          # m s^-2 (Earth)
MEAN_MOL_WEIGHT = 28.97 # mean molecular weight of air, in atomic mass units
T_STRATOSPHERE = 210.0  # K, temperature floor above the "tropopause" (fixed)
MU_DIFFUSIVITY = 1.0 / np.sqrt(3.0)   # diffusivity-approximation angle


PARAM_NAMES = ["T0", "kappa", "log_H2O", "log_CO2", "log_CH4", "R_p"]
PARAM_LABELS = [r"$T_0$ [K]", r"$\kappa$", r"$\log_{10}$ H$_2$O",
                r"$\log_{10}$ CO$_2$", r"$\log_{10}$ CH$_4$", r"$R_p$ [$R_\oplus$]"]
TRUTH = dict(T0=288.0, kappa=0.15, log_H2O=-2.52, log_CO2=-3.4,
             log_CH4=-5.0, R_p=1.0)


PRIORS = dict(T0=(200.0, 350.0), kappa=(0.05, 0.35),
              log_H2O=(-6.0, -0.5), log_CO2=(-7.0, -0.5), log_CH4=(-9.0, -2.0),
              R_p=(0.5, 2.0))


SNR_REF = 20.0          # signal-to-noise at the reference wavelength below
LAMBDA_REF = 11.2       # micron, reference wavelength for the S/N definition
RNG_SEED = 4             
DISTANCE_PC = 10.0      # the planet is placed at 10 parsecs

# "constant": H2O mixing ratio is the same at all heights (what the retrieval assumes).
# "falling" : H2O drops steeply above 0.3 bar (closer to a real Earth).

TRUTH_H2O_PROFILE = "constant"


NLIVE = 500             # number of live points in nested sampling
DLOGZ = 0.1             # stopping tolerance on the evidence

TRUTH_VECTOR = np.array([TRUTH[k] for k in PARAM_NAMES])
