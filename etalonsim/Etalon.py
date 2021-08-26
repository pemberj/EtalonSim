import numpy as np
import astropy.units as u

# Description of plane wave illumination of an etalon a la Wikipedia

class Etalon():
    """
    Basic analytic etalon description
    """

    def __init__(self, n=1, l=6*u.mm, θ=0*u.degree, R_1=0.95, R_2=None):
        self.n = n
        self.l = l
        self.θ = θ
        self.R_1 = R_1
        if R_2 is None:
            self.R_2 = R_1
        else:
            self.R_2 = R_2
        self.R = np.sqrt(self.R_1 * self.R_2)

    @property
    def F_coef(self):
        return (4 * self.R) / ((1 - self.R)**2)

    @property
    def finesse(self):
        return np.pi / (2 * np.arcsin(1 / np.sqrt(self.F_coef)))

    @property
    def delta_lambda(self, λ=780*u.nm):
        """
        Free spectral range of the etalon, defaults to λ = 780nm
        """
        n_g = 1
        return ((λ ** 2) / (2 * n_g * self.l * np.cos(self.θ) + λ)).to("pm")

    @property
    def R_max(self):
        """
        Maximum reflectivity, R_max
        """
        return (4 * self.R) / ((1 + self.R) ** 2)

    def delta(self, λ):
        return ((2 * np.pi / λ) * 2 * self.n * self.l * np.cos(self.θ)).si.value

    def Transmittance(self, λ):
        """
        Transmittance function, sometimes T_e
        Transmitted beam intensity as a function of wavelength
        """
        return 1 / (1 + self.F_coef * (np.sin(self.delta(λ) / 2))**2)

    def intensity(self, λ):
        return self.Transmittance(λ)

    @property
    def T(self):
        return 1 - self.R

    def amplitude(self, m, λ):
        return self.T * (self.R ** m) * np.exp(1j * m * self.delta(λ))

    def summed_amplitudes(self, λ):
        """
        The geometric infinite sum of ampllitudes over m from 0 to infinity
        """
        return self.T / (1 - self.R * np.exp(1j * self.delta(λ)))


    def print_info(self):

        for p, val in self.__dict__.items():
            print(f"{p}: {val}")