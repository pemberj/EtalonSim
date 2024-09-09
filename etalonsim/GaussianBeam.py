"""
A simple implementation of Gaussian Beams using Python.

Dependencies: numpy, astropy (for units), matplotlib (for plots)
"""

import numpy as np
from numpy.typing import ArrayLike
import astropy.units as u
from astropy.units import Quantity
from dataclasses import dataclass


@dataclass
class GaussianBeam():
    """
    """

    λ: Quantity = 633 * u.nm
    E_0: float = 1
    n: float = 1
    w_0: Quantity = 10 * u.um
    debug_level: int | float = 0
    

    def __post_init__(self) -> None:
        if self.debug_level > 0:
            for p, val in self.__dict__.items():
                print(f"{p}: {val}")


    @property
    def k(self) -> float:
        # k, wavenumber
        return ((2 * np.pi * self.n) / self.λ).si


    @property
    def z_R(self) -> Quantity:
        # z_R, Rayleigh range, sometimes z_0
        # (not to be confused with a distance to the beam waist)
        return ((np.pi * self.n * self.w_0**2) / self.λ).si


    def q(self, z: Quantity = 0 * u.mm) -> float:
        # q, complex beam parameter
        return z + 1j * self.z_R


    def w_squared(self, z: Quantity = 0 * u.mm) -> Quantity:
        # Square of the beam width

        _z = (z / self.z_R).si

        if self.debug_level > 0:
            print(f"z / self.z_0: {_z}")
            print(f"w(z)^2: {(self.w_0**2 * (1 + (_z)**2)).si}")

        return (self.w_0**2 * (1 + (_z)**2)).si


    def w(self, z: Quantity = 0*u.mm):
        """
        w, beam width
        Defined as the radius at which the field amplitudes fall to 1/e of their
        axial values (i.e., where the intensity values fall to 1/e2 of their
        axial values)
        """

        if self.debug_level > 0:
            print(f"w(z): {np.sqrt(self.w_squared(z)).si}")

        return np.sqrt(self.w_squared(z)).si


    def R(self, z: Quantity = 0*u.mm):
        # R, radius of curvature of the wavefront

        _z = (self.z_R / z).si

        if self.debug_level > 0:
            print(f"z_0 / z: {_z}")
            print(f"R(z): {(z * (1 + (_z)**2)).si}")

        R = (z * (1 + (_z)**2)).si
        # Account for infinite radius of curvature at beam waist
        R[np.isnan(R)] = np.inf * u.m

        return R


    def curvature(self, z: Quantity = 0 * u.mm) -> Quantity:
        # Curvature of the wavefront at a distance z from the beam waist

        return 1 / self.R(z)
    

    def sag(self, z: Quantity = 0 * u.mm, D: Quantity = 4 * u.mm) -> Quantity:
        """
        Returns the sag (sagitta) between the wavefront at a distance z from the
        beam waist and at a diameter D from the propagation axis

        Modified definition to avoid a singularity at z=0, infinite radius
        """

        R = self.R(z)

        return R - np.sign(R) * np.sqrt(R**2 - (D/2)**2)


    def amplitude(
        self,
        r: Quantity | None = None,
        x: Quantity = 0 * u.mm,
        y: Quantity = 0 * u.mm,
        z: Quantity = 0 * u.mm,
        ) -> float:
        """
        The amplitude of the electric field at the given coordinates
        """
        if r is None:
            r = np.sqrt(x**2 + y**2).si
        else:
            r = r.si

        return self.E_0 * (self.w_0 / self.w(z)).si * np.exp((-r**2 / (self.w_squared(z))).si)


    def Gouy_phase(self, z: Quantity = 0 * u.mm,) -> float:
        """
        The Gouy phase at a distance z from the beam waist
        """

        if self.debug_level > 0:
            print(f"Gouy phase, arctan(z/z_0): {np.arctan((z / self.z_R).si)}")

        return np.arctan((z / self.z_R).si) / u.radian


    def longitudinal_phase(self, z: Quantity = 0 * u.mm,) -> float:
        """
        The longitudinal phase of the E field at a distance z from the beam waist
        """

        return (self.k * z).si


    def radial_phase(
        self,
        r: Quantity | None = None,
        x: Quantity = 0 * u.mm,
        y: Quantity = 0 * u.mm,
        z: Quantity = 0 * u.mm
        ) -> float:
        """
        The radial phase of the E field at the given coordinates
        """
        if r is None:
            r = np.sqrt(x**2 + y**2)
        else:
            r = r.si

        return (self.k.si * (r**2).si) / (2 * self.R(z).si)


    def E_field(
        self,
        x: Quantity = 0 * u.mm,
        y: Quantity = 0 * u.mm,
        z: Quantity = 0 * u.mm
        ) -> ArrayLike:
        """
        Returns the complex E field of the wave, encoding amplitude and phase,
        at the given coordinates
        """

        return self.amplitude(x, y, z)\
            * np.exp(-1j * (
                self.longitudinal_phase(z)\
              + self.radial_phase(x, y, z)\
              - self.Gouy_phase(z)
              ))


    def E_field_partial(
        self,
        r: Quantity | None = None,
        x: Quantity = 0 * u.mm,
        y: Quantity = 0 * u.mm,
        z: Quantity = 0 * u.mm,
        include="amplitude longitudinal"
        ) -> ArrayLike:
        """
        A function to construct partial Gaussian beams, for instance including only Gouy phase or
        only radial phase, for inspection of the contributions of each phase component, while taking
        in to account the amplitude variations of these contributions.

        Args:
            r, radius from the optical axis in units of length. Defaults to None.
            Coordinates x, y, z
            include (str, optional): [description]. Defaults to "amplitude longitudinal".
        """

        if r is None:
            r = np.sqrt(x**2 + y**2)
        else:
            r = r.si

        E = 1.

        if "amplitude" in include.lower():
            # print("Computing amplitude")
            E = E * self.amplitude(r=r, z=z)

        if "longitudinal" in include.lower():
            # print("Computing longitudinal phase")
            E = E * np.exp(-1j * self.longitudinal_phase(z=z))
            
        if "radial" in include.lower():
            # print("Computing radial phase")
            E = E * np.exp(-1j * self.radial_phase(r=r, z=z))

        if "gouy" in include.lower():
            # print("Computing Gouy phase")
            E = E * np.exp(+1j * self.Gouy_phase(z=z))

        return E


    def E_gouy_integrated(
        self,
        r: Quantity | float = np.inf,
        z: Quantity = 0 * u.mm) -> float:
        """
        A function that returns the E-field incorporating only the Gouy phase
        component integrated from r=0 to the given r in the plane a given
        distance z from the beam waist.
        """

        ret = self.E_0 * self.w_0 * self.w(z=z)\
            * np.pi * np.exp(1j * self.Gouy_phase(z=z))

        if np.isinf(r):
            return ret

        else:
            exponent = -((r**2) / (self.w(z)**2)).si
            ret *= (1 - np.exp(exponent))
            return ret


    def E_radial_integrated(
        self,
        r: Quantity | float = np.inf,
        z: Quantity = 0 * u.mm) -> float:
        """
        A function that returns the E-field incorporating only the radial phase
        component integrated from r=0 to the given radius r in the plane a given
        distance z from the beam waist.
        """

        ret = self.E_0 * (self.w_0 / self.w(z=z)) * np.pi\
            / ((1 / (self.w(z=z)**2)) + ((1j * self.k) / (2 * self.R(z=z))))

        if np.isinf(r):
            return ret

        else:
            exponent1 = -r**2 * (1 / self.w(z=z)**2).si
            exponent2 = -r**2 * ((1j * self.k) / (2 * self.R(z=z))).si
            exponent = exponent1 + exponent2
            ret *= (1 - np.exp(exponent))
            return ret


    def summed_E_field(
        self,
        z: Quantity = 0 * u.mm,
        exclude: str = "none",
        debug_level: float | int = 0
        ) -> ArrayLike:
        """
        For a beam symmetric about the optical axis. Integrating over r from 0
        to infinity gives the summed (inegrated) E field across a plane at z.
        Used for calculating the interference of full waves in devices like a
        Fabry-Perot etalon.
        """

        A = ((u.m**2 / self.w(z)**2) + 1j*((self.k * u.m**2) / (2 * self.R(z))))
        integral_over_r = np.pi / A

        # baseline result, including only the amplitude and longitudinal phase,
        # ie. a plane wave beam
        ret = self.E_0 * (self.w_0 / self.w(z))\
            * np.exp(-1j * self.longitudinal_phase(z))

        try:
            if "gouy" not in exclude.lower():
                if debug_level > 0: print("Including Gouy phase")
                ret *= np.exp(1j * self.Gouy_phase(z))

            else: print("Returning results without Gouy phase...")

            if "radial" not in exclude.lower():
                if debug_level > 0:
                    print("Including radial effects (wavefront curvature, "+\
                          "intensity variations)")
                ret *= integral_over_r

            else: print("Returning results with no radial effects...")

            return ret.si

        except Exception as e:
            print(e)
            print("Assuming full Gaussian beam...")
            
            return (self.E_0 * (self.w_0 / self.w(z))\
                             * np.exp(-1j * self.longitudinal_phase(z))\
                             * np.exp(1j * self.Gouy_phase(z))\
                             * integral_over_r).si



def test():
    
    ...



if __name__ == "__main__":

    test()