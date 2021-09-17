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

    r = 5 * u.mm  # radius at which to evaluate curvature, radial phase

    # Don't use round numbers for this plot
    plt.rcParams["axes.autolimit_mode"] = "data"

    fig = plt.figure()
    ax = fig.gca()

    λ = (mirror_spacing / int(mirror_spacing / (780*u.nm))).to("nm")
    z0s = np.linspace(-300, 50_000, 250)
    # z0s = np.linspace(-300, -100, 100)

    for w_0 in [0.5, 1, 2, 4] * u.mm:

        summed_average_phase_differences = []

        for z0 in z0s:

            Zs = np.arange(z0, z0+(num_round_trips)*round_trip_mm, round_trip_mm) * u.mm

            reflectivity = np.array([mirror_reflectivity**(n) for n in range(1, len(Zs)+1)])

            beam = GaussianBeam(λ=λ, w_0=w_0)

            gouy = np.angle(beam.E_gouy_integrated(r=r, z=Zs)).value * reflectivity
            radi = np.angle(beam.E_radial_integrated(r=r, z=Zs)).value * reflectivity

            summed_average_phase_differences.append(np.sum(gouy + radi))
            
        ax.plot((z0s*u.mm).to("m"), summed_average_phase_differences,
                label=f"$w_0=${w_0}, $\lambda=${λ:.0f}")

    ax.axhline(y=0, color="k", ls="--", alpha=0.5)
    
    # ax.set_yscale("log")
    # ax.set_ylim(-5, 5)

    ax.set_xlabel("Offset from beam waist, $z_0$ [m]")
    ax.set_ylabel(f"Sum of average phase\nover {num_round_trips} round trips [waves]")

    ax.legend()

    # plt.show()
    plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")


if __name__ == "__main__":

    main()