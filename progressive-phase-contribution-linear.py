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
    num_round_trips = 140 #number of reflections on the OUTPUT surface
    mirror_reflectivity = 0.95

    reflectivity = np.array([mirror_reflectivity**(n) for n in range(1, num_round_trips)])

    Zs = np.arange(0, round_trip_mm*(num_round_trips-1), round_trip_mm) * u.mm

    λ = mirror_spacing / int(mirror_spacing / (780*u.nm))
    
    # waist_radii = [1.25, 2, 3] * u.mm
    w_0 = 2 * u.mm

    fig = plt.figure(figsize=(10, 7))
    ax = fig.gca()

    beam = GaussianBeam(λ=λ, w_0=w_0)

    r = 4 * u.mm

    xr = np.linspace(-r, r, 31)
    yr = np.linspace(-r, r, 31)
    X, Y = np.meshgrid(xr, yr)
    dist_from_center = np.sqrt(X**2 + Y**2)
    circle_mask = dist_from_center < r

    # integrated_phases = []
    phases_avg = []
    phases_min = []
    phases_max = []
    for Z in Zs:
        E = beam.E_field(x=X[circle_mask], y=Y[circle_mask], z=Z).value / (2*np.pi)
        phi = np.angle(E)
        phases_min.append(np.min(phi))
        phases_max.append(np.max(phi))
        phases_avg.append(np.angle(np.mean(E)))
        
    # average = ax.plot(Zs, np.abs(phases_avg), label="Average phase (cycles)")
    # ax.fill_between(Zs, phases_min, phases_max, alpha=0.1, color=average[0].get_color())

    phases_avg *= reflectivity
    phases_min *= reflectivity
    phases_max *= reflectivity
    refd = ax.plot(Zs, phases_avg, label="Average phase contribution (scaled by reflection)")
    ax.fill_between(Zs, phases_min, phases_max, alpha=0.1, color=refd[0].get_color(),
                    label="Min/max phase")

    summed = np.sum(phases_avg)
    ax.fill_between(Zs, 0, phases_avg, color=refd[0].get_color(), alpha=0.25, hatch="//",
                    label=f"Cumulative phase contribution over\n{num_round_trips} round trips: {summed:.2f} cycles")

    # ax.plot(Zs, reflectivity, color="k", ls="--", alpha=0.5, label="Mirror reflection attenuation")

    ax.legend(loc="lower right")

    ax3 = ax.twiny()
    ax3.set_xlim(0, num_round_trips)
    ax3.set_xlabel("number of round-trips, n")

    ax.set_title(f"Gaussian beam etalon, waist radius {w_0}")
    ax.set_xlabel("propagation distance (mm)")
    ax.set_ylabel("cycles")

    # ax.set_ylim(-0.01, 0.01)
    ax.set_xlim(min(Zs).value, max(Zs).value)

    # plt.show()
    plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")



if __name__ == "__main__":
    main()
