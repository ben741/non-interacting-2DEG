# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 14:57:27 2017

@author: benjamin
"""

from __future__ import division
import numpy as np

# all times in ns
# all masses in kg
# all energies in K (i.e. multipy by k_b to get Joules)

# physical constants
k_b = 1.38064852e-23
elec = 1.60217662e-19
h = 6.62607004e-34
hbar = h/ (2 * np.pi)
m_e = 9.10938356e-31

# GaAs constants
m_star = 0.067 * m_e

def filling (B, n_e):
    """ Calculate the filling factor for a given B-field and electon density
    Below is an example, which functions as a doctest to confirm that there
    have been no changes to its behaviour

    >>> print ('%.5f'%filling (1, 1e15))
    4.13567

    """
    return n_e *h/(elec * B)

def omega_c(B, m=m_star, q=elec):
    """ Calculate the cyclotron frequency of charge particles with mass m and
    charge q at magnetic field B
    Below is an example, which functions as a doctest to confirm that there
    have been no changes to its behaviour

    >>> print ('%.5f'%omega_c (1))
    2625104511528.22217

    """
    return q * B / m # in 1/s

def fermi(eps, mu, T):
    """ Calculate the Fermi distribution with chemical potential mu at
    temperature T
    eps and mu have to be in units of Kelvin
    >>> import numpy
    >>> print ('%.5f'%fermi (0.6, 0.5, 0.1))
    0.26894
    """
    return 1/(1 + np.exp((eps - mu)/(T)))




if __name__ == "__main__":
    import doctest
    doctest.testmod()
