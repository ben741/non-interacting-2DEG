# -*- coding: utf-8 -*-
r"""
Created on Wed Jan  4 14:57:27 2017

@author: benjamin

This code is meant to gather together a few different calculations I have done
in one place, with proper documentation and testing. The idea is to first
generate a density of states corresponding to broadened Landau levels, and then
use it to calculate:
- electrical conductivity, sigma_xx
- specific heat, C
- thermopower, S_xx and S_xy (i.e. Mott thermopower)
- $\partial{\mu}/\partial{T}$, which is equal to $\partial{\S}/\partial{N}$
- spectral diffusion in the presence of an electric field
and later, perhaps
- thermal conductuctivity?

You could use this code to first calculate conducitivity, and by comparing to
experiment choose the most appropriate model for the broadening. Then, using
the same DOS that best matches the conductance, calculate other quantities of
interest such as C or S_xx.

TODO:
- move specific heat code into this file
- move thermopower code into this file
- move
"""

from __future__ import division
import numpy as np

# some things are just more convenient without the np prefix...
from numpy import exp, pi

# all times in ns
# all masses in kg
# all energies in K (i.e. multipy by k_b to get Joules)

# physical constants
k_b = 1.38064852e-23
q_e = 1.60217662e-19 # electron charge
h = 6.62607004e-34
hbar = h/(2*pi)
m_e = 9.10938356e-31

# GaAs constants
m_star = 0.067 * m_e

def filling(B, n_e):
    """ Calculate the filling factor for a given B-field and electon density
    Below is an example, which functions as a doctest to confirm that there
    have been no changes to its behaviour

    >>> print ('%.5f'%filling (1, 1e15))
    4.13567

    """
    return n_e *h/(q_e * B)

def omega_c(B, m=m_star, q=q_e):
    """ Calculate the cyclotron frequency of charged particles with mass m and
    charge q at magnetic field B
    Below is an example, which functions as a doctest to confirm that there
    have been no changes to its behaviour

    >>> print ('%.5f'%omega_c (1))
    2625104511528.22217

    """
    return q * B / m # in 1/s

def mag_length(B, q=q_e):
    """ Calculate the magnetic length in meters

    >>> print ('%.5e'%mag_length(1))
    2.56556e-08
    """

    return np.sqrt(hbar/(q * B))


def fermi(eps, mu, T):
    """ Calculate the Fermi distribution with chemical potential mu at
    temperature T
    eps and mu have to be in units of Kelvin
    >>> import numpy
    >>> print('%.5f'%fermi (0.6, 0.5, 0.1))
    0.26894
    """
    return 1/(1 + np.exp((eps - mu)/(T)))

def deriv(y, x):
    """ Calculate numerical derivative without reducing array length by
    padding with zeros at the ends, which is fine if y is flat there.

    >>> import numpy
    >>> x = numpy.linspace(0, 1, 5)
    >>> y = x**2
    >>> deriv(y,x)
    array([ 0. ,  0.5,  1. ,  1.5,  0. ])
    """
    answer = np.zeros(len(y))
    answer[1:-1] = (y[2:] - y[:-2])/(x[2:] - x[:-2])
    return answer

def generate_DOS(eps, B, tau_q, LL_energies=None):
    """ Calculate the density of states for non-interacting electrons
    at magnetic field B with quantum lifetime tau_q
    Implements equation 4 of Zhang et al. PRB 80, 045310 (2009)

    >>> import numpy as np
    >>> generate_DOS(np.array([0.5, 1.2]), 1.0, 1e-12)
    array([ 0.25191318,  0.32785897])
    """
    # calculate cyclotron frequency, convert into energy in units of Kelvin
    E_c = omega_c(B) * hbar / k_b # in K

    # by default, take spinless Landau levels with gaps of E_c
    # I'm not sure about the added 0.5, which is not included in Zhang but is
    # in other references.
    if LL_energies is None:
        LL_energies = E_c * np.arange(0.5, 1000, 1)

    # sigma squared for the Gaussian
    sigma2 = 0.5 * E_c * hbar / (np.pi * tau_q * k_b) # sigma squared

    # the prefactor normalizes the height of the Gaussian, accounting for
    # the broadening given by sigma2
    prefactor = np.sqrt(omega_c(B) * tau_q)

    # Sum over Gaussians centred at E_c *N. This could be done more
    # pythonically or more efficiently
    # Should also make it so you can pass in your own Landau level spacings,
    # so that you can use spin-split LLs
    return_value = np.zeros(len(eps))
    for eps_0 in LL_energies:
        return_value += exp(-(eps - eps_0)**2 / (2 * sigma2))
    return  prefactor * return_value


def specific_heat(eps, DOS, T, mu=None):
    """ Numerically calculate specific heat of the 2DEG
    works by calculating total energy U at temperatures slightly above and
    below T and approximating C = dU/dT

    Example: calculate the specific heat for a flat density of states
    
    >>> import numpy as np
    >>> eps = np.linspace (0,100,100)
    >>> dens = np.ones(100)
    >>> print '%.5f'%specific_heat(eps, dens, 10, mu = 50)
    27.91817
    """
    # generate high and low temperatures, which are +/- 5% from temperature we're interested in
    dT = T* 0.1
    T_h = T + 0.5 * dT
    T_l = T - 0.5 * dT

    if mu is None:
        # these functions are yet to be moved into this file.
        mu_high = get_mu_at_T(B, T_h, DOS)
        mu_low = get_mu_at_T(B, T_l, DOS)
    else:
        mu_high = mu
        mu_low = mu
   
    dU = np.trapz((fermi(eps, mu_high, T_h)-fermi(eps, mu_low, T_l)) * (eps)* DOS, x = eps)
    # previously used (eps-mu_low) instead of (eps) in above
    
    return dU/dT


# run doctests when this file is executed as a script
if __name__ == "__main__":
    import doctest
    doctest.testmod()
