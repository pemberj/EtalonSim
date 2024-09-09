import numpy as np
from numpy.typing import ArrayLike
import astropy.units as u
import pprint
pp = pprint.PrettyPrinter(indent=4)

class SphericalBeam():
    """
    An implementation of a very simple spherical electromagnetic wave
    """

    def __init__(self, λ=633*u.nm, E_0=1, n=1, plane_normal=(0,0,1), debug_level=0):

        self.λ = λ
        self.E_0 = E_0
        self.n = n
        self.plane_normal = plane_normal
        self.debug_level = debug_level
        
        self.k_scalar = (2* np.pi / λ) * np.linalg.norm(self.plane_normal)
        self.k_vector = self.k_scalar * self.plane_normal

        if debug_level > 0:
            pp.pprint(self.__dict__)


    def E_field(self, x=0*u.mm, y=0*u.mm, z=0*u.mm):
        """
        https://en.wikipedia.org/wiki/Electromagnetic_wave_equation
        """

        x, y, z = x.to("mm").value, y.to("mm").value, z.to("mm").value
        
        if isinstance(z, int| float):
            z = np.ones_like(x) * z
            
        if isinstance(x, float | int) & isinstance(y, float | int) == 1:
            x = [x] * len(z)
            y = [y] * len(z)
        
        # length of vectors x,y,z
        rho_unitless = np.sqrt(np.sum(np.array([x, y, z])**2, axis=0))
        k_scalar_unitless = self.k_scalar.to("1/mm").value

        return self.E_0 * np.exp(-1j * k_scalar_unitless * rho_unitless)




def test():

    from matplotlib import pyplot as plt
    from plotting import jakeStyle
    plt.style.use(jakeStyle)


    xi = np.linspace(-10*u.um, 10*u.um, 500)
    yi = np.linspace(-10*u.um, 10*u.um, 500)
    x, y = np.meshgrid(xi, yi)

    beam1 = SphericalBeam(λ=633*u.nm, E_0=1, n=1, plane_normal=(0,0,1), debug_level=0)
    
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title("The electric field amplitude for a single z-slice")
    ax.imshow(np.real(beam1.E_field(x=x, y=y, z=1*u.um) * np.conj(beam1.E_field(x=x, y=y, z=1*u.mm))))
    
    # Create a second beam at slightly different wavelength
    beam2 = SphericalBeam(λ=630*u.nm, E_0=1, n=1, plane_normal=(0,0,1), debug_level=0)
    Zs = np.linspace(0, 0.2, 1400) * u.mm
    Es = beam1.E_field(z=Zs) + beam2.E_field(z=Zs)
    Is = np.real(Es * np.conj(Es))

    # Don't use round numbers for this plot
    plt.rcParams["axes.autolimit_mode"] = "data"

    fig = plt.figure()
    ax = fig.gca()
    ax.set_title("Real and imaginary parts of the electric field amplitude,\nand the beam intensity as a function of z-distance")
    ax.plot(Zs, np.real(Es), alpha=0.25)
    ax.plot(Zs, np.imag(Es), alpha=0.25)
    ax.plot(Zs, Is)
    
    plt.show()



if __name__ == "__main__":

    test()