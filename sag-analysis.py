import os
import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt

from etalonsim.GaussianBeam import GaussianBeam
from etalonsim.PlanarBeam import PlanarBeam
from etalonsim.plotting import jakeStyle
plt.style.use(jakeStyle)
from etalonsim.colours import wavelength_to_rgb


def main():

    mirror_spacing = 6 * u.mm

    target_λ = 780 * u.nm
    λ = mirror_spacing / int(mirror_spacing / target_λ)

    num_reflections = 60
    
    z_total = (2 * mirror_spacing) * (num_reflections)
    zs = np.arange(0, z_total.to("mm").value, 2 * mirror_spacing.to("mm").value) * u.mm
    
    #beam_diameters = [1.25, 2, 3] * u.mm
    beam_diameters = [2] * u.mm
    
    mirror_reflectivity = 0.95
    attenuation = np.array([mirror_reflectivity**(n / 2) for n in range(1, 2 * (num_reflections), 2)])

    # Radius at which to evaluate both sag and summed E-field
    r = 4 * u.mm


    # Don't use round numbers for this plot
    plt.rcParams["axes.autolimit_mode"] = "data"

    fig = plt.figure(figsize=(10, 7))
    ax = fig.gca()
    ax2 = ax.twinx()

    for w_0 in beam_diameters:
        beam = GaussianBeam(λ=λ, w_0=w_0)

        xr = np.linspace(-r, r, 101)
        yr = np.linspace(-r, r, 101)
        X, Y = np.meshgrid(xr, yr)
        dist_from_center = np.sqrt(X**2 + Y**2)
        circle_mask = dist_from_center < r
        
        summed_E = beam.summed_E_field(z=zs)

        dimmed = np.array([E * R for E, R in zip(summed_E, attenuation)])
        intensities = dimmed# * np.conj(dimmed)

        ax.step(zs, intensities / np.max(intensities), where="pre",
                label="(Normalised) integrated\nE-field after reflection")

        sag = beam.sag(zs, D=2*r)
        ax2.plot(zs, (sag / λ).si, "r--", label="Wavefront sag, waves")

        # ax.axvline(x=beam.z_R.to("mm").value, ls="--", color="k", alpha=0.25, label="Rayleigh length")

    ax.step(zs, attenuation, where="pre",
            color="k", ls="--", label="Mirror reflection attenuation")

    # Make a legend with lines from both 'axes'
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="upper center")

    ax3 = ax.twiny()
    ax3.set_xlim(ax.get_xlim() / (2 * mirror_spacing.value))
    ax3.set_xlabel("number of round-trips")

    ax.set_xlabel("propagation distance (mm)")
    ax.set_ylabel("")

    ax2.tick_params(axis="y", labelcolor="r")
    ax2.set_ylabel("wavefront sag (waves)", color="r")
    # ax.set_xlim(min(zs).value, max(zs).value)

    # plt.show()
    plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")



if __name__ == "__main__":
    main()
