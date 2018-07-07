import numpy as np
from scipy.special import wofz


def gaussian(x, sigma):

    return np.sqrt(0.5/np.pi)/sigma * np.exp(-(x/sigma)**2 * 0.5)

def gaussian_hwhm(x, alpha):
    """ Return Gaussian line shape at x with HWHM alpha """
    return (np.sqrt(np.log(2) / np.pi) / alpha *
            np.exp(-(x / alpha)**2 * np.log(2)))

def lorentzian(x, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return gamma / np.pi / (x**2 + gamma**2)

def voigt(x, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))

    return (np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) /
            sigma /np.sqrt(2*np.pi))


def gamma_cabaret(x, xlow, xhigh, ylow, yhigh, slope=0):

    dx = x[1] - x[0]

    mask = (x>=xlow) & (x<xhigh)

    X = x[mask] - xlow
    dX = X[-1] + dx
    slope_ = (yhigh - ylow) / dX
    ramp = X * slope_ + ylow

    val = np.ones_like(x) * ylow

    val[mask] = ramp
    val[x>=xhigh] = yhigh + slope * (x[x>=xhigh] - xhigh)

    return val
