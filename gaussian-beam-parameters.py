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

    """
    mirror_spacing = 6*u.mm #mm
    round_trip = 2 * mirror_spacing
    round_trip_mm = round_trip.to("mm").value
    num_round_trips = 100 #number of reflections on the OUTPUT surface
    mirror_reflectivity = 0.95

    r = 5 * u.mm # maximum radius of beam

    w_0 = 4 * u.mm
    """

    mirror_spacing = 6 * u.mm
    num_round_trips = 100
    λ = mirror_spacing / int(mirror_spacing / (780*u.nm))

    waist_radii = np.linspace(0.005, 15, 80) * u.mm

    z_Rs = []
    R_halfs = [] # beam wavefront radius in middle of etalon
    w_halfs = []
    Gouy_halfs = []
    radial_halfs = []
    amp_halfs = []
    for w_0 in waist_radii:
        beam = GaussianBeam(λ=λ, w_0=w_0)
        z_Rs.append(beam.z_R.si.value)
        R_halfs.append(beam.R(z=mirror_spacing * num_round_trips/2).si.value)
        w_halfs.append(beam.w(z=mirror_spacing * num_round_trips/2).si.value)
        Gouy_halfs.append(beam.Gouy_phase(z=mirror_spacing * num_round_trips/2))
        radial_halfs.append(beam.radial_phase(r=5*u.mm, z=mirror_spacing * num_round_trips/2))
        amp_halfs.append(beam.amplitude(r=5*u.mm, z=mirror_spacing * num_round_trips/2))



    fig = plt.figure()
    ax = fig.gca()

    ax.plot(waist_radii.value, z_Rs/max(z_Rs), label="$z_R$, Rayleigh length")
    ax.plot(waist_radii.value, R_halfs/max(R_halfs), label="R($z_{mid}$), radius of wavefront curvature")
    ax.plot(waist_radii.value, w_halfs/max(w_halfs), label="w($z_{mid}$), beam radius")
    ax.plot(waist_radii.value, Gouy_halfs/max(Gouy_halfs), label="$\\Psi$($z_{mid}$), Gouy phase")
    ax.plot(waist_radii.value, radial_halfs/max(radial_halfs), label="radial($z_{mid}$), radial phase @ r=5mm")
    ax.plot(waist_radii.value, amp_halfs/max(amp_halfs), label="amp($z_{mid}$), E-field amplitude @ r=5mm")

    ax.legend()

    ax.set_title("Normalised beam properties as a function of waist radius $w_0$", size=18)
    ax.set_xlabel("Beam waist $w_0$")
    # ax.set_ylabel("Phase offset [cycles]")

    ax.set_xlim(0, max(waist_radii).value)
    ax.set_ylim(-0.05, 1.5)

    # plt.show()
    plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")






if __name__ == "__main__":

    main()