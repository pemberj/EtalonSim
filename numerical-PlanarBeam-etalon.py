"""
"""

import os
import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt

from etalonsim.Etalon import Etalon
from etalonsim.PlanarBeam import PlanarBeam
from etalonsim.helper import FWHM
from etalonsim.colours import wavelength_to_rgb
from etalonsim.plotting import jakeStyle
plt.style.use(jakeStyle)


def main():

    # Compute the transmission through the same etalon, but for plane-wave illumination!
    # For this, I am using just the x, y = (0, 0) coordinates of the Gaussian wave...

    mirror_spacing = 6*u.mm #mm
    round_trip = 2 * mirror_spacing
    round_trip_mm = round_trip.to("mm").value
    num_round_trips = 150 #number of reflections on the OUTPUT surface
    mirror_reflectivity = 0.95

    reflectivity = np.array([mirror_reflectivity**(n) for n in range(1, num_round_trips)])

    Zs = np.arange(0, round_trip_mm*(num_round_trips-1), round_trip_mm) * u.mm
    Xs = np.zeros_like(Zs)
    Ys = np.zeros_like(Zs)

    c = 6e-3 / 7692 * 1e9
    w = 0.005
    λs = np.linspace(c - w/2, c + w/2, 201) * u.nm

    output = []

    for λ in λs:

        beam = PlanarBeam(λ=λ, plane_normal=(0,0,1))

        results = np.real(beam.E_field(x=Xs, y=Ys, z=Zs))
        dimmed = reflectivity * results
        intensity = np.real(np.sum(dimmed) * np.conj(np.sum(dimmed)))

        # sqrt needed to get it to give the correct values... Not sure why
        output.append(intensity**0.5)

    fig = plt.figure()
    ax = fig.gca()

    analytic_etalon = Etalon()
    distance_between_peaks = analytic_etalon.delta_lambda
    peak_fwhm = np.diff(FWHM(λs.to("pm").value, np.real(output)))[0] * u.pm
    calculated_finesse = distance_between_peaks/peak_fwhm

    ax.plot(λs, output / np.max(output), label=f"Calculated finesse: {calculated_finesse:.1f}")
    ax.set_title("Plane-wave illumination (numerical simulation)")
    ax.set_xlabel("Wavelength (nm)")
    #ax.set_yticklabels([])
    ax.set_ylim(-0.1, 1.1)

    ax.legend()

    plt.show()
    # plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")


if __name__ == "__main__":
    main()