import os
import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt

from etalonsim.Etalon import Etalon
from etalonsim.helper import FWHM
from etalonsim.plotting import jakeStyle
plt.style.use(jakeStyle)





def main():

    etalon = Etalon()

    fig = plt.figure()
    ax = fig.gca()

    # Single peak:
    # λs = np.linspace(499.998, 500.002, 500) * u.nm
    c = 6e-3 / 7692 * 1e9
    w = 0.005
    λs = np.linspace(c - w/2, c + w/2, 501) * u.nm

    print("Analytic finesse:", etalon.finesse)

    # Three peaks:
    # λs = np.linspace(499.975, 500.025, 200) * u.nm

    intensity = np.real(etalon.intensity(λs))
    peak_fwhm = np.diff(FWHM(λs.to("pm").value, intensity))[0] * u.pm

    ax.plot(λs, intensity / np.max(intensity), label="Analytic\nreflection-limited peak\n"\
                                                   +f"FWHM={peak_fwhm:.4f}")

    print("Finesse calculated from FWHM and Δλ:", etalon.delta_lambda / peak_fwhm)

    etalon.print_info()
    
    ax.legend()
    ax.set_title("Plane-wave illumination (analytic description)")
    ax.set_xlabel("Wavelength (nm)")
    #ax.set_yticklabels([])
    ax.set_ylim(-0.1, 1.1)
    
    # plt.show()

    plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")



if __name__ == "__main__":
    main()
