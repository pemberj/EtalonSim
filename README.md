# EtalonSim

A work-in-progress Python package for the analytical and numerical simulation of plane-mirror
Fabry-Pérot interferometers (etalons).

Originally developed for the analysis and design of etalons for optical frequency comb generation,
and to investigate the expected degradation of etalon finesse when illuminated with a Gaussian beam
rather than an idealised plane-wave beam.

`./etalonsim/` is the Python package itself, containing code that describes a simple etalon analytically,
and code describing plane-wave and Gaussian beams for use with numerical simulations of etalons.

The root directory of this repository contains various example scripts used in numerical simulations,
and to generate plots (example plots in `./plots/` with the same filenames as the scripts used to
generate them.)

Dependencies: `numpy`, `astropy`, `matplotlib`