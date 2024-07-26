import os
import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt

from etalonsim.helper import FWHM
from etalonsim.GaussianBeam import GaussianBeam
from etalonsim.PlanarBeam import PlanarBeam
from etalonsim.Etalon import Etalon
from etalonsim.plotting import jakeStyle
plt.style.use(jakeStyle)


def main():

    mirror_spacing = 6 * u.mm #mm
    round_trip = 2 * mirror_spacing
    round_trip_mm = round_trip.to("mm").value
    num_round_trips = 3000 #number of reflections on the OUTPUT surface
    mirror_reflectivity = 0.95

    analytic_etalon = Etalon(l=mirror_spacing, R_1=mirror_reflectivity)

    reflectivity = np.array([mirror_reflectivity**(n) for n in range(1, num_round_trips)])

    Zs = np.arange(0, round_trip_mm*(num_round_trips-1), round_trip_mm) * u.mm
    # two same-length placeholder arrays for the plane-wave version
    Xs = np.zeros_like(Zs)
    Ys = np.zeros_like(Zs)

    c = 6e-3 / 7692 * 1e9
    w = 0.01
    λs = np.linspace(c - w/2, c + w/2, 301) * u.nm

    # Three peaks:
    # λs = np.linspace(499.975, 500.025, 200) * u.nm

    output = {"gaussian": [],
              "planar": []}

    for λ in λs:

        beam = GaussianBeam(λ=λ, E_0=1, w_0=4*u.mm, n=1, debug_level=0)
        results = beam.summed_E_field(z=Zs)
        dimmed = results * reflectivity
        intensity = np.real(np.sum(dimmed) * np.conj(np.sum(dimmed)))
        output["gaussian"].append(intensity**0.5) # I don't know why taking the square root here makes it work as expected!

        # Now the same for the planar beam (only evaluated on-axis)
        beam = PlanarBeam(λ=λ, plane_normal=(0,0,1))
        results = np.real(beam.E_field(x=Xs, y=Ys, z=Zs))
        dimmed = results * reflectivity
        intensity = np.real(np.sum(dimmed) * np.conj(np.sum(dimmed)))
        output["planar"].append(intensity**0.5)


    fig = plt.figure()
    ax = fig.gca()

    distance_between_peaks = analytic_etalon.delta_lambda
    analytic_intensity = analytic_etalon.intensity(λs)

    analytic_fwhm = np.diff(FWHM(λs.to("pm").value, np.real(analytic_intensity)))[0] * u.pm
    analytic_finesse = distance_between_peaks/analytic_fwhm

    ax.plot(λs, analytic_intensity / np.max(analytic_intensity),
            label=f"Analytic\nreflection-limited peak\nFWHM={analytic_fwhm:.4f}\nFinesse={analytic_finesse:.2f}")

    for _, [beam_type, values] in enumerate(output.items()):        
        # Why am I getting a zero-offset in the numerical ones?
        # Corresponds to the very first mirror reflection
        values = values / np.max(values) - (1 - np.sqrt(mirror_reflectivity))
        
        peak_fwhm = np.diff(FWHM(λs.to("pm").value, values))[0] * u.pm
        calculated_finesse = distance_between_peaks/peak_fwhm

        ax.plot(λs, values / np.max(values), ls="--",
                label=f"{beam_type}\nFWHM={peak_fwhm:.4f}\n"\
                     +f"Finesse={calculated_finesse:.2f}")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalised intensity")
    #ax.set_yticklabels([])
    ax.set_xlim(min(λs).to("nm").value, max(λs).to("nm").value)
    ax.set_ylim(-0.05, 1.1)
    ax.legend()

    plt.show()
    # plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")
    


if __name__ == "__main__":
    main()
