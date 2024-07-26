import os
import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt

from etalonsim.Etalon import Etalon
from etalonsim.plotting import jakeStyle
plt.style.use(jakeStyle)


def main():

    etalon = Etalon(n=1.5, l=25.4*u.mm, θ=0*u.degree, R_1=0.02, R_2=None)

    print(etalon.finesse)

    fig = plt.figure()
    ax = fig.gca()

    λs = np.linspace(499.99, 500.01, 200) * u.nm

    intensity = etalon.intensity(λs)

    ax.plot(λs, intensity / np.max(intensity))

    ax.set_title("Plane-wave illumination")
    ax.set_xlabel("Wavelength (nm)")
    # ax.set_yticklabels([])
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(min(λs).value, max(λs).value)
    
    plt.show()
    # plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")
    


if __name__ == "__main__":
    main()
