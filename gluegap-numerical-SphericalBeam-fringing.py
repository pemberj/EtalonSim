"""

"""

import os
import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt

from etalonsim.SphericalBeam import SphericalBeam
from etalonsim.plotting import jakeStyle
plt.style.use(jakeStyle)
#from colours import wavelength_to_rgb


def main():

    # I want to define the Gaussian beam for each wavelength based on an f-number rather than the
    # beam waist. The beam waist for a beam of the same f/# is different for each wavelength.
    f_number = 10

    mirror_spacing = 10 * u.um # Glue gap thickness in beamsplitter cube
    num_round_trips = 30
    mirror_reflectivity = 0.01 # Calculated reflectivity due to refractive
    # index mismatch between the glue and the glass

    reflectivity = np.array([mirror_reflectivity**i for i in range(1, num_round_trips)])
    baseline_transmission = 1 - np.sum(reflectivity)

    # Work out where to sample the beam - starting from the back surface of the BSC
    backfocus = 60.5 * u.mm
    start_z = -(backfocus + mirror_spacing)
    stop_z = start_z + (2*mirror_spacing) * (num_round_trips-1)

    # numpy doesn't enjoy working with astropy.units sometimes...
    Zs = np.linspace(start_z.to("mm").value, stop_z.to("mm").value, (num_round_trips-1)) * u.mm

    # λs = np.linspace(380, 930, 500) * u.nm
    λs = np.linspace(480, 520, 200) * u.nm
    # λs = np.linspace(498, 502, 200) * u.nm

    output = []

    for λ in λs:

        beam = SphericalBeam(λ=λ, E_0=1, n=1, plane_normal=(0,0,1), debug_level=0)

        results = beam.E_field(z=Zs)

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

    plt.show()
    # plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")



if __name__ == "__main__":
    main()
