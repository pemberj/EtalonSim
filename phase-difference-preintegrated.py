import os
import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt

from etalonsim.GaussianBeam import GaussianBeam
from etalonsim.plotting import jakeStyle
plt.style.use(jakeStyle)




def main():

    mirror_spacing = 6*u.mm #mm
    round_trip = 2 * mirror_spacing
    round_trip_mm = round_trip.to("mm").value
    num_round_trips = 100 #number of reflections on the OUTPUT surface
    mirror_reflectivity = 0.95

    Zs = np.arange(0, (num_round_trips)*round_trip_mm, round_trip_mm) * u.mm

    reflectivity = np.array([mirror_reflectivity**(n) for n in range(1, len(Zs)+1)])

    # Don't use round numbers for this plot
    plt.rcParams["axes.autolimit_mode"] = "data"

    fig = plt.figure()
    ax = fig.gca()

    λ = mirror_spacing / int(mirror_spacing / (780*u.nm))

    r = 5 * u.mm  # radius at which to evaluate curvature, radial phase

    # waist_radii = [2.5, 3, 4, 5, 6] * u.mm
    waist_radii = np.linspace(0.04, 10, 250) * u.mm
    # Zoom in to close-to-zero region
    # waist_radii = np.linspace(0.05, 1.75, 50) * u.mm

    summed_average_phase_differences = []
    for w_0 in waist_radii:
        beam = GaussianBeam(λ=λ, w_0=w_0)

        gouy = np.angle(beam.E_gouy_integrated(r=r, z=Zs)).value * reflectivity
        radi = np.angle(beam.E_radial_integrated(r=r, z=Zs)).value * reflectivity

        summed_average_phase_differences.append(np.sum(gouy + radi))
        
    ax.plot(waist_radii, summed_average_phase_differences)

    # ax.set_yscale("log")

    ax.set_xlabel("Beam waist radius, $w_0$ [mm]")
    ax.set_ylabel(f"Sum of average phase\nover {num_round_trips} round trips")

    # plt.show()
    plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")


if __name__ == "__main__":

    main()