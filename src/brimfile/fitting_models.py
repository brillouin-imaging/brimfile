from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numpy.typing import NDArray

import numpy as np
from collections.abc import Callable

from enum import Enum

class FitModel(Enum):
    Undefined = "Undefined"
    Lorentzian = "Lorentzian"
    DHO = "DHO"
    Gaussian = "Gaussian"
    Voigt = "Voigt"
    Custom = "Custom"

def get_fit_model(model: FitModel) -> Callable[..., NDArray]:
    """
    Get the function corresponding to the given fit model
    """
    fit_model: Callable[..., NDArray]
    match model:
        case FitModel.Lorentzian:
            fit_model = lorentzian
        case FitModel.DHO:
            fit_model = dho
        case FitModel.Gaussian:
            fit_model = gaussian
        case FitModel.Voigt:
            fit_model = voigt
        case _:
            raise NotImplementedError(f"Fit model {model} is not implemented yet")
    return fit_model

#%% Fit model definitions

def lorentzian(x: NDArray[np.floating | np.integer], 
               nu0: float, gamma: float, a: float, b: float) -> NDArray:
    """Model of a simple lorentzian lineshape

    Parameters
    ----------
    x : array
        The frequency array
    nu0 : float
        The center position of the function
    gamma : float
        The FWHM linewidth of the function
    a : float
        The amplitude of the peak
    b : float
        The constant offset of the data

    Returns
    -------
    array
        The value of the function at each point of the frequency array
    """
    return b + a/( (2*(x-nu0)/gamma) ** 2 + 1)

def dho(x: NDArray[np.floating | np.integer],
        nu0: float, gamma: float, a: float, b: float) -> NDArray:
    """Model of a Damped Harmonic Oscillator (DHO) lineshape

    The function is normalized so that the peak maximum equals ``a + b``
    in the underdamped regime (``nu0 > gamma / sqrt(2)``).

    Parameters
    ----------
    x : array
        The frequency array
    nu0 : float
        The resonance frequency (center position) of the oscillator
    gamma : float
        The damping coefficient (linewidth) of the oscillator
    a : float
        The amplitude of the peak (peak maximum above baseline)
    b : float
        The constant offset of the data

    Returns
    -------
    array
        The value of the function at each point of the frequency array
    """
    return b + a * gamma**2 * (nu0**2 - gamma**2 / 4) / ((x**2 - nu0**2)**2 + (x * gamma)**2)

def gaussian(x: NDArray[np.floating | np.integer], 
             nu0: float, gamma: float, a: float, b: float) -> NDArray:
    """Model of a simple gaussian lineshape

    Parameters
    ----------
    x : array
        The frequency array
    nu0 : float
        The center position of the function
    gamma : float
        The FWHM linewidth of the function
    a : float
        The amplitude of the peak
    b : float
        The constant offset of the data

    Returns
    -------
    array
        The value of the function at each point of the frequency array
    """
    return b + a * np.exp(-4*np.log(2)*((x-nu0)/gamma)**2)

def voigt(x: NDArray[np.floating | np.integer],
          nu0: float, gamma_lorentz: float, gamma_gauss: float, a: float, b: float) -> NDArray:
    """Model of a simple Voigt-like lineshape.

    This implementation uses a pseudo-Voigt approximation (Thompson et al.)

    Parameters
    ----------
    x : array
        The frequency array
    nu0 : float
        The center position of the function
    gamma_lorentz : float
        The FWHM linewidth of the lorentzian component of the function
    gamma_gauss : float
        The FWHM linewidth of the gaussian component of the function
    a : float
        The amplitude of the peak
    b : float
        The constant offset of the data

    Returns
    -------
    array
        The value of the function at each point of the frequency array
    """
    if gamma_lorentz <= 0 and gamma_gauss <= 0:
        # Degenerate zero-width limit: return an impulse-like peak at the center.
        x_arr = np.asarray(x, dtype=float)
        return b + a * np.isclose(x_arr, nu0, rtol=0.0, atol=np.finfo(float).eps).astype(float)

    if gamma_lorentz <= 0:
        return gaussian(x, nu0, gamma_gauss, a, b)

    if gamma_gauss <= 0:
        return lorentzian(x, nu0, gamma_lorentz, a, b)

    x_arr = np.asarray(x, dtype=float)
    dx = x_arr - nu0

    # Effective FWHM approximation for the Voigt profile.
    f_l = float(gamma_lorentz)
    f_g = float(gamma_gauss)
    f = (
        f_g**5
        + 2.69269 * f_g**4 * f_l
        + 2.42843 * f_g**3 * f_l**2
        + 4.47163 * f_g**2 * f_l**3
        + 0.07842 * f_g * f_l**4
        + f_l**5
    ) ** (1.0 / 5.0)

    ratio = f_l / f
    eta = np.clip(
        1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3,
        0.0,
        1.0,
    )

    gaussian_part = np.exp(-4.0 * np.log(2.0) * (dx / f) ** 2)
    lorentzian_part = 1.0 / (1.0 + (2.0 * dx / f) ** 2)

    return b + a * (eta * lorentzian_part + (1.0 - eta) * gaussian_part)