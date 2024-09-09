"""
"""

import numpy as np
from numpy.typing import ArrayLike
import astropy.units as u
from astropy.units import Quantity
from dataclasses import dataclass

import pprint
pp = pprint.PrettyPrinter(indent=4)


@dataclass
class SphericalBeam():
    """
    An implementation of a very simple spherical electromagnetic wave
    """
    
    λ: Quantity = 633 * u.nm
    E_0: int | float = 1
    n: int | float = 1
    plane_normal: ArrayLike = (0,0,1)
    debug_level: int | float = 0
    
    def __post_init__(self) -> None:
        self.k_scalar = (2* np.pi / self.λ) * np.linalg.norm(self.plane_normal)
        self.k_vector = self.k_scalar * self.plane_normal

        if self.debug_level > 0:
            pp.pprint(self.__dict__)
    

    def E_field(
        self,
        x: Quantity = 0 * u.mm,
        y: Quantity = 0 * u.mm,
        z: Quantity = 0 * u.mm,
        ) -> ArrayLike:
        """
        https://en.wikipedia.org/wiki/Electromagnetic_wave_equation
        """

        x, y, z = x.to("mm").value, y.to("mm").value, z.to("mm").value
        
        if isinstance(z, float | int):
            z = np.ones_like(x) * z
            
        if isinstance(x, float | int) & isinstance(y, float | int) == 1:
            x = [x] * len(z)
            y = [y] * len(z)
        
        # length of vectors x,y,z
        rho_unitless = np.sqrt(np.sum(np.array([x, y, z])**2, axis=0))
        k_scalar_unitless = self.k_scalar.to("1/mm").value

        return self.E_0 * np.exp(-1j * k_scalar_unitless * rho_unitless)
    

    def summed_E_field(
        self,
        z: Quantity | ArrayLike,
        ) -> np.complex128 | ArrayLike:
        """
        Sum the electric field for a range of z-values
        """

        Es = self.E_field(z=z)
        return Es
        return np.sum(Es, axis=0)




def test():

    from matplotlib import pyplot as plt
    from plotting import jakeStyle
    plt.style.use(jakeStyle)


    xi = np.linspace(-10*u.um, 10*u.um, 500)
    yi = np.linspace(-10*u.um, 10*u.um, 500)
    x, y = np.meshgrid(xi, yi)

    beam1 = SphericalBeam(
        λ = 633*u.nm,
        E_0 = 1,
        n = 1,
        plane_normal = (0,0,1),
        debug_level = 0,
        )
    
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title("The electric field amplitude for a single z-slice")
    ax.imshow(
        np.real(beam1.E_field(x=x, y=y, z=1*u.um)\
      * np.conj(beam1.E_field(x=x, y=y, z=1*u.mm)))
        )
    
    # Create a second beam at slightly different wavelength
    beam2 = SphericalBeam(
        λ = 630*u.nm,
        E_0 = 1,
        n = 1,
        plane_normal = (0,0,1),
        debug_level=0,
        )
    Zs = np.linspace(0, 0.2, 1400) * u.mm
    Es = beam1.E_field(z=Zs) + beam2.E_field(z=Zs)
    Is = np.real(Es * np.conj(Es))

    # Don't use round numbers for this plot
    plt.rcParams["axes.autolimit_mode"] = "data"

    fig = plt.figure()
    ax = fig.gca()
    ax.set_title("Real and imaginary parts of the electric field amplitude,"+\
                 "\nand the beam intensity as a function of z-distance")
    ax.plot(Zs, np.real(Es), alpha=0.25)
    ax.plot(Zs, np.imag(Es), alpha=0.25)
    ax.plot(Zs, Is)
    
    plt.show()



if __name__ == "__main__":

    test()