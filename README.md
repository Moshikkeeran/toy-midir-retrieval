# toy-midir-retrieval

A small, self-contained **toy Bayesian atmospheric retrieval** for a synthetic Earth-like
planet observed in the mid-infrared, written to learn the structure of retrieval frameworks
such as pyRetLIFE. 


1. `src/make_data.py` builds a synthetic emission spectrum over 4-18.5 micron at R = 100
   (the wavelength range of the LIFE concept) from known "truth" parameters, and adds white
   Gaussian noise (S/N = 20 at 11.2 micron).
2. `src/run_retrieval.py` recovers six parameters (surface temperature T0, temperature-profile
   exponent kappa, three constant mixing ratios of H2O, CO2 and CH4, and planet radius) with
   nested sampling (dynesty), using a Gaussian likelihood, uniform priors for T0, kappa and
   radius, and log-uniform priors for the abundances.
3. `src/make_figures.py` makes a corner plot, a posterior-predictive spectrum, and a context
   figure on why the mid-infrared is useful for characterising temperate planets.



## Limitations

- **Opacities are toy Gaussian bands**, not line-by-line or correlated-k data. Band centres
  are real; strengths and widths are order-of-magnitude choices, independent of pressure and
  temperature.
- 1D, cloud-free, constant mixing ratios, fixed surface pressure and stratospheric temperature,
  no stellar or instrument systematics, white noise.
- With one noise realisation, "truth inside the 68% interval" is expected for about two
  thirds of parameters; a proper calibration test would repeat the retrieval over many noise
  seeds.

## To run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/make_data.py
python src/run_retrieval.py      # a few minutes
python src/make_figures.py
```

Outputs go to `data/`, `results/` (posterior samples, `summary.txt`) and `figures/`.

