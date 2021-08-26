"""
A script to generate the expected etalon effect arising from internal reflections in an
antireflection-coated beamsplitter cube.

Here the definition of a Gaussian beam is used to model a beam focused by a telescope.
In this (f/6) beam is a beamsplitter cube used to examine both the telescope's field
of view and to look at an optical fiber in the telescope's prime focus.

Due to the low reflectivity inside the cube, and the ever-changing curvature of the
beam's wavefronts at successive round-trip reflections, the effect is miniscule compared
to the same beamsplitter cube placed in a collimated (~plane-wave) beam.
"""

import os
import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from etalonsim.GaussianBeam import GaussianBeam
from etalonsim.plotting import jakeStyle
plt.style.use(jakeStyle)
#from colours import wavelength_to_rgb


def beam_waist(λ=0.5, f_number=6, n=1.5):
    """
    Calculate the beam waist (w_0) for a beam with an input f-number, far away from the beam waist
    (i.e. at z >> z_R)

    Args:
        lam (float, optional): [description]. Defaults to 0.5.
        f_number (int, optional): [description]. Defaults to 6.
    """

    return (λ / (np.pi * n * np.arcsin(1 / (2 * n * f_number)))).to("um")


def main():

    # I want to define the Gaussian beam for each wavelength based on an f-number rather than the
    # beam waist. The beam waist for a beam of the same f/# is different for each wavelength.
    f_number = 6

    mirror_spacing = 25.4 * u.mm # 1 inch beamsplitter cube
    num_round_trips = 50 # even 3 reflections gets very low power left indeed, here I go overboard
    mirror_reflectivity = 0.02 # 2% reflection from antireflection-coated outer cube surface

    reflectivity = np.array([mirror_reflectivity**i for i in range(1, num_round_trips)])
    baseline_transmission = 1 - np.sum(reflectivity)

    # Work out where to sample the beam - starting from the back surface of the BSC
    backfocus = 60.5 * u.mm
    start_z = -(backfocus + mirror_spacing)
    stop_z = start_z + (2*mirror_spacing) * (num_round_trips-1)

    # numpy doesn't enjoy working with astropy.units sometimes...
    Zs = np.linspace(start_z.to("mm").value, stop_z.to("mm").value, (num_round_trips-1)) * u.mm

    #λs = np.linspace(380, 930, 3000) * u.nm
    λs = np.linspace(499.99, 500.01, 100) * u.nm

    output = []

    for λ in λs:

        # Create a GaussianBeam for a given wavelength
        beam = GaussianBeam(λ=λ, E_0=1, w_0=beam_waist(λ=λ, f_number=f_number, n=1.5), n=1, debug_level=0)

        results = beam.summed_E_field(z=Zs)

        # Multiply by the reflectivity attenuation
        dimmed = np.array([result * R for result, R in zip(results, reflectivity)])

        # Get the intensity of the beam (I = E * E†)
        intensity = np.real(np.sum(dimmed) * np.conj(np.sum(dimmed)))

        output.append(intensity)

    output = np.array(output)

    fig = plt.figure()
    ax = fig.gca()
    
    # Don't use round numbers for this plot
    plt.rcParams["axes.autolimit_mode"] = "data"

    transmission = baseline_transmission - output
    ax.plot(λs, transmission, label="")

    # ax.set_title("Beamsplitter cube etalon, gaussian beam illumination")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative transmission")
    #ax.set_yticklabels([])
    ax.set_ylim(min(transmission), max(transmission))
    ax.set_xlim(min(λs).value, max(λs).value)
    # ax.legend()

    # plt.show()
    plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")



if __name__ == "__main__":
    main()
