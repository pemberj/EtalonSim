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

    source_spacing = 0.5 * u.mm
    distance_to_detector = 0.5 * u.m
    detector_num_pix = 10800
    pixel_size = 9 * u.um
    detector_x = detector_num_pix * pixel_size

    num_samples = 100_000

    x, y = (np.random.rand(2, num_samples) - 0.5) * detector_x

    # HeNe laser
    λ = 632.8 * u.nm

    z = [distance_to_detector.to("mm").value]*len(x) * u.mm

    beam_1 = SphericalBeam(λ=λ)
    E_1 = beam_1.E_field(x - source_spacing / 2, y, z)
    beam_2 = SphericalBeam(λ=λ)
    E_2 = beam_2.E_field(x + source_spacing / 2, y, z)

    interference = E_1 + E_2

    intensity = np.real(interference * np.conj(interference))

    # Set interpolation bounds
    xplot = np.linspace(-detector_x.to("mm").value/2, detector_x.to("mm").value/2, 2_000)
    yplot = np.linspace(-detector_x.to("mm").value/2, detector_x.to("mm").value/2, 2_000)

    # Interpolate data into a regular grid
    zplot = griddata((x.to("mm").value, y.to("mm").value), intensity, (xplot[None, :], yplot[:, None]), method="cubic")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    contour = ax.pcolormesh(xplot, yplot, zplot, cmap="afmhot")
    # ax.set_xticks([])
    # ax.set_yticks([])

    # for s in ["top", "bottom", "left", "right"]:
    #     ax.spines[s].set_visible(False)

    plt.show()


if __name__ == "__main__":
    main()