import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm

from etalonsim.PlanarBeam import PlanarBeam
from etalonsim.SphericalBeam import SphericalBeam
from etalonsim.GaussianBeam import GaussianBeam

from scipy.interpolate import griddata

from etalonsim.plotting import jakeStyle
fm.fontManager.addfont(path="./Quicksand_regular.ttf")
plt.style.use(jakeStyle)



def main():

    source_spacing = 1 * u.mm
    distance_to_detector = 1 * u.m
    detector_num_pix = 10800
    pixel_size = 9 * u.um
    detector_x = detector_num_pix * pixel_size

    num_samples = 1_000

    xi = np.linspace(-detector_x/2, detector_x/2, num_samples)
    yi = np.linspace(-detector_x/2, detector_x/2, num_samples)
    x,y = np.meshgrid(xi, yi)

    # Beam waist radius => fiber mode field diameter / 2?
    w_0 = 4 * u.um / 2

    # HeNe laser
    λ = 632.8 * u.nm

    beam_1 = SphericalBeam(λ=λ)
    E_1 = beam_1.E_field(x - source_spacing / 2, y, distance_to_detector)
    beam_2 = SphericalBeam(λ=λ)
    E_2 = beam_2.E_field(x + source_spacing / 2, y, distance_to_detector)

    interference = E_1 + E_2
    print(interference)

    intensity = np.real(interference * np.conj(interference))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    ax.imshow(intensity)

    ax.set_xticks([])
    ax.set_yticks([])

    for s in ["top", "bottom", "left", "right"]:
        ax.spines[s].set_visible(False)

    plt.show()
    # plt.savefig("WEIRD.png")


if __name__ == "__main__":
    main()