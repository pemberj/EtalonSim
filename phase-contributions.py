"""
"""

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

    r = 5 * u.mm # maximum radius of beam

    w_0 = 4 * u.mm
    λ = mirror_spacing / int(mirror_spacing / (780*u.nm))

    beam = GaussianBeam(λ=λ, w_0=w_0)

    Zs = np.arange(0, beam.z_R.to("mm").value / 10, round_trip_mm) * u.mm
    reflectivity = np.array([mirror_reflectivity**(n) for n in range(1, len(Zs)+1)])

    gouy = np.angle(beam.E_gouy_integrated(r=r, z=Zs)).value * reflectivity
    radi = np.angle(beam.E_radial_integrated(r=r, z=Zs)).value * reflectivity

    # Don't use round numbers for this plot
    plt.rcParams["axes.autolimit_mode"] = "data"

    fig = plt.figure()
    ax = fig.gca()

    G = ax.plot(Zs, gouy, ls="--", label="Gouy phase")
    R = ax.plot(Zs, radi, ls="-.", color=G[0].get_color(), label="Radial phase")

    ax.plot(Zs, gouy + radi, color=G[0].get_color(), alpha=0.5, label="Sum")

    ax.legend()

    ax.set_title(f"Contributions of Gouy and radial phase,\nreduced by mirror reflection for $w_0=${w_0}")
    ax.set_xlabel(f"Propagation distance z [mm], {num_round_trips} round trips")
    ax.set_ylabel("Phase offset [cycles]")

    # plt.show()
    plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")






if __name__ == "__main__":

    main()