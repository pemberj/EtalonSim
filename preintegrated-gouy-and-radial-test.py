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

    max_radius = 6 * u.mm

    w_0 = 5 * u.mm

    λ = mirror_spacing / int(mirror_spacing / (780*u.nm))

    beam = GaussianBeam(λ=λ, w_0=w_0)

    Zs = np.arange(0, round_trip_mm * num_round_trips, round_trip_mm) * u.mm
    reflectivity = np.array([mirror_reflectivity**(n) for n in range(1, len(Zs)+1)])

    gouy = np.angle(beam.E_gouy_integrated(r=max_radius, z=Zs))
    radi = np.angle(beam.E_radial_integrated(r=max_radius, z=Zs))
    
    gouy = reflectivity * np.array(gouy)
    radi = reflectivity * np.array(radi)

    # Don't use round numbers for this plot
    plt.rcParams["axes.autolimit_mode"] = "data"

    fig = plt.figure()
    ax = fig.gca()

    ax.plot(Zs, gouy, label="Gouy phase")
    ax.plot(Zs, radi, label="Radial phase")
    # ax.plot(Zs, gouy + radi)

    # plt.show()
    plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")






if __name__ == "__main__":

    main()