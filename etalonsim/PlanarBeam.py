import numpy as np
import astropy.units as u
import pprint
pp = pprint.PrettyPrinter(indent=4)

class PlanarBeam():
    """
    An implementation of a very simple plane electromagnetic wave
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
        
        # Position vector r
        r_unitless = np.array([x, y, z])
        # r = r_unitless * u.mm
        k_vector_unitless = self.k_vector.to("1 / mm").value

        return self.E_0 * np.exp(-1j * np.dot(k_vector_unitless, r_unitless))




def test():

    from matplotlib import pyplot as plt
    from plotting import jakeStyle
    plt.style.use(jakeStyle)

    beam1 = PlanarBeam(λ=633*u.nm, E_0=1, n=1, plane_normal=(0,0,1), debug_level=0)
    beam2 = PlanarBeam(λ=632*u.nm, E_0=1, n=1, plane_normal=(0,0,1), debug_level=0)
    Zs = np.linspace(0, 0.2, 1400) * u.mm
    Es = beam1.E_field(z=Zs) + beam2.E_field(z=Zs)
    Is = np.real(Es * np.conj(Es))

    # Don't use round numbers for this plot
    plt.rcParams["axes.autolimit_mode"] = "data"

    plt.plot(Zs, np.real(Es), alpha=0.25)
    plt.plot(Zs, np.imag(Es), alpha=0.25)
    plt.plot(Zs, Is)
    plt.show()



if __name__ == "__main__":

    test()