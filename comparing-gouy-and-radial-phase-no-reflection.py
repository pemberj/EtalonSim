import os
import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt

from etalonsim.GaussianBeam import GaussianBeam
from etalonsim.plotting import jakeStyle
plt.style.use(jakeStyle)




def main():

    mirror_spacing = 600*u.mm #mm # NOTE changed mirror spacing to see very large distance from waist
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

    waist_radii = [2, 2.5, 3, 4] * u.mm
    # Zoom in to close-to-zero region
    # waist_radii = np.linspace(0.5, 2.5, 15) * u.mm

    summed_average_phase_differences = []
    for w_0 in waist_radii:
        beam = GaussianBeam(λ=λ, w_0=w_0)

        gouy = np.angle(beam.E_gouy_integrated(r=r, z=Zs))# * reflectivity NOTE
        radi = np.angle(beam.E_radial_integrated(r=r, z=Zs))# * reflectivity NOTE

        summed_average_phase_differences.append(np.sum(gouy + radi))
        
        G = ax.plot(0, 0)
        ax.plot(Zs, np.abs(gouy), color=G[0].get_color(), ls="--")
        ax.plot(Zs, np.abs(radi), color=G[0].get_color(), ls="-.")
        ax.fill_between(
            Zs.to(u.mm).value,
            np.abs(gouy.to(u.rad).value),
            np.abs(radi.to(u.rad).value),
            color=G[0].get_color(),
            alpha=0.25,
            hatch="//",
            label=f"$w_0=${w_0} ($z_R\\approx${beam.z_R:.0f})"
            )

    ax.plot(0, 0, color="k", alpha=0.75, ls="--", label="Gouy phase")
    ax.plot(0, 0, color="k", alpha=0.75, ls="-.", label="radial phase (absolute)")
    
    ax.set_title(f"Gouy phase and radial phase\nintegrated out to $r={r}$")

    ax.set_xlabel("propagation distance z [mm]")
    ax.set_ylabel("average phase [cycles]")

    ax.legend()
    plt.show()

    # plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")



if __name__ == "__main__":

    main()