import numpy as np
import astropy.units as u
from astropy.units import Quantity
from dataclasses import dataclass

# Description of plane wave illumination of an etalon a la Wikipedia

@dataclass
class Etalon():
    """
    Basic analytic etalon description
    """
    
    n: float = 1
    l: Quantity = 6 * u.mm
    θ: Quantity = 0 * u.degree
    R_1: float = 0.95
    R_2: float = None
    
    
    def __post_init__(self):
        if not self.R_2:
            self.R_2 = self.R_1


    @property    
    def R(self) -> float:
        return np.sqrt(self.R_1 * self.R_2)


    @property
    def F_coef(self) -> float:
        return (4 * self.R) / ((1 - self.R)**2)


    @property
    def finesse(self) -> float:
        return np.pi / (2 * np.arcsin(1 / np.sqrt(self.F_coef)))


    @property
    def delta_lambda(self, λ=780*u.nm) -> Quantity:
        """
        Free spectral range of the etalon, defaults to λ = 780nm
        """
        n_g = 1
        return ((λ ** 2) / (2 * n_g * self.l * np.cos(self.θ) + λ)).to("pm")


    @property
    def R_max(self) -> float:
        """
        Maximum reflectivity, R_max
        """
        return (4 * self.R) / ((1 + self.R) ** 2)


    def delta(self, λ: Quantity = 780 * u.nm) -> Quantity | float:
        return ((2 * np.pi / λ) * 2 * self.n * self.l * np.cos(self.θ)).si.value


    def Transmittance(self, λ) -> float:
        """
        Transmittance function, sometimes T_e
        Transmitted beam intensity as a function of wavelength
        """
        return 1 / (1 + self.F_coef * (np.sin(self.delta(λ) / 2))**2)


    def intensity(self, λ) -> float:
        return self.Transmittance(λ)


    @property
    def T(self) -> float:
        return 1 - self.R


    def amplitude(self, m: int = 6_000, λ: Quantity = 780 * u.nm) -> np.complex128:
        # Returns complex amplitude of the electric field
        return (self.T * (self.R ** m) * np.exp(1j * m * self.delta(λ))).si.value


    def summed_amplitudes(self, λ: Quantity = 780 * u.nm) -> np.complex128:
        """
        The geometric infinite sum of ampllitudes over m from 0 to infinity
        """
        return (self.T / (1 - self.R * np.exp(1j * self.delta(λ)))).si.value
    
    
    def m(self, λ: Quantity = 780 * u.nm) -> int:

        return int(self.l / λ)
    
    
    def λ(self, m: int = 6_000) -> Quantity:

        return (self.l / m).to(u.nm)
        


    def print_info(self) -> None:

        for p, val in self.__dict__.items():
            print(f"{p}: {val}")
            
            
            
def main() -> None:
    
    e = Etalon(l = 6*u.mm)
    
    ms = range(e.m(λ = 1*u.um), e.m(λ = 374.977*u.nm))
    peak_λs = [e.λ(m).to(u.nm).value for m in ms] * u.nm
    

if __name__ == "__main__":
    
    main()
