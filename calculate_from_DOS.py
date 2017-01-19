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

Sections:
A. Utility functions, like Fermi distribution, its derivative etc.
B. Density of States generation
C. Specific heat calculation
D. Conductance & spectral diffusion

Units:
- Energies are in units of k_b T
- lengths are in units of meters


TODO:
- verify conductance calculations (sigma_DC, sigma_nl)
- write function to generate eps appropriately based on T, E_f etc.
- write simple calculation of E_f at 0 field
- move thermopower code into this file
- move other calcs here
- specific heat returns NaN instead of 0 at T = 0

"""

from __future__ import division
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
# some things are just more convenient without the np prefix...
from numpy import exp, sqrt, pi



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

# dimensionless DOS to actual units of 1/(K m^2)
nu0 = m_star/(pi * hbar**2) * k_b


###############################################################################
# A. General-purpose functions
###############################################################################

def E_fermi(n_e):
    """ Calculate the zero field Fermi energy """
    return n_e / nu0 # in K

def v_fermi(n_e):
    """ Calculate the zero field Fermi velocity """
    return sqrt(2*E_fermi(n_e) * k_b / m_star)

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

    # T=0 case handled separately since (eps-mu)/T is NaN, not +inf
    # or -inf as required to generate step function
    if T == 0:
        return np.piecewise(eps, [eps < mu, eps == mu, eps > mu], [1, 0.5, 0])

    # suppress overflow warnings for this calculation. Numpy handles +inf
    # and -inf gracefully in this calculation.
    old_settings = np.seterr()
    np.seterr(over='ignore')
    result = 1/(1 + np.exp((eps - mu)/T))
    np.seterr(**old_settings)

    return result

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

def gauss(x, x0, gamma):
    """ returns a gaussian function with integral = 1
    """
    sigma = gamma / sqrt(2.0)
    
    A = 1/ (sigma * sqrt(2*pi))
    return (A * exp (-0.5 * (x-x0)**2/sigma**2))

def lorentz(x, x0, gamma):
    """ returns a Lorentzian function with integral = 1
    """    
    return (0.5/pi) * gamma / ((x-x0)**2 + 0.25 * gamma**2)
    
def get_mu_at_T(reduced_DOS, T, n_e=3e15, precision=1e-15):
    """ Find the chemical potential for a given density and temperature

    Example: calculate mu, in K, at T = 1 mK at zero field for n_e = 3e15 m^-2
    >>> eps = np.linspace (0, 500, 10000)
    >>> mu = get_mu_at_T([eps, np.ones(len(eps))], 1e-3, 3e15)
    >>> print '%.3f'%mu
    124.390
    """
    eps, dens = reduced_DOS
    
    mu = np.amax(eps)/2
    mu_step = np.amax(eps)/4
    n_e_calc = 0

    DOS = nu0 * dens
    while mu_step > mu * precision:
        n_e_calc = simps(fermi(eps, mu, T) * DOS, x=eps)

        if abs(n_e - n_e_calc) < 0.00001 * n_e:
            break
        elif n_e > n_e_calc:
            mu = mu + mu_step
        elif n_e < n_e_calc:
            mu = mu - mu_step
        mu_step = mu_step/2.

    return mu
    
###############################################################################
# B. Calculation of the Density of States
###############################################################################

def generate_eps(T_low, T_high, n_e, factor = 10):
    """ Generate an array for epsilon which is centred around the Fermi 
    energy and has sufficient range and resolution for specific heat 
    calculations.
    """
    
    E_f = E_fermi (n_e)
    eps_min = E_f - factor * T_high
    eps_max = E_f + factor * T_high
    eps_step = T_low / factor
    
    return np.arange (eps_min, eps_max+eps_step, eps_step)


def generate_DOS(B, tau_q, eps=None, LL_energies=None, T_low=0.1, T_high=1,
                            n_e=3e15, factor=10, tau_q_dep=lambda B:1,
                            broadening='Gaussian'):
    """ Calculate the density of states for non-interacting electrons
    at magnetic field B with quantum lifetime tau_q
    Used to implement; equation 4 of Zhang et al. PRB 80, 045310 (2009)
    now more like Piot et. al. PRB 72 245325 (2005).

    The result is dimensionless and needs to be multiplied by
    nu0 = m/(pi * hbar **2) * k_b to obtain the DOS in units of 1/(K m^2)

    This is not a very useful example. It just tests that there have been no
    changes made to the function by calling it for a short, arbitrary array.
    >>> import numpy as np
    >>> eps = np.array([0.5, 1.2])
    >>> generate_DOS(1.0, 1e-12, eps=eps)
    [array([ 0.5,  1.2]), array([ 0.00063729,  0.00110615])]
    """
    # calculate cyclotron frequency, convert into energy in units of Kelvin
    E_c = omega_c(B) * hbar / k_b # in K

    if broadening == 'Gaussian':
        broaden = lambda eps, eps_0, gamma: gauss(eps, eps_0, gamma)
        eps_width = 6
    elif broadening == 'Lorentzian':
        broaden = lambda eps, eps_0, gamma: lorentz(eps, eps_0, gamma)
        eps_width = 30
        
    # by default, take spinless Landau levels with gaps of E_c
    # I'm not sure about the added 0.5, which is not included in Zhang but is
    # in other references such as Kobayakawa

    if eps is None:
        eps = generate_eps(T_low, T_high, n_e, factor)

    # precalculate sigma squared for the Gaussian
    #sigma2 = 0.5 * E_c * hbar / (np.pi * tau_q * k_b) # sigma squared
    #sigma = sqrt(sigma2)
    gamma = 0.5 * hbar/(k_b * tau_q)
    sigma = gamma/sqrt(2)
    
    ### we could also intelligently choose Landau levels to sum over
    ### let's commit first before modifying this...
        
    if LL_energies is None:
        # choose LLs only in a range such that their broadening reaches
        # all the way to the fermi level.
    
        E_min = max (np.amin (eps) - gamma * eps_width, E_c)
        E_max = np.amax(eps)  + gamma * eps_width
        LL_max = np.ceil(E_max/E_c)
        LL_min = np.floor(E_min/E_c)
        LL_energies = E_c * np.arange(LL_min, LL_max+1, 1)
        

    # the prefactor normalizes the height of the Gaussian, accounting for
    # the broadening given by sigma2
    #prefactor = np.sqrt(omega_c(B) * tau_q)

    # Sum over Gaussians centred at E_c *N. This could be done more
    # pythonically or more efficiently
    # Should also make it so you can pass in your own Landau level spacings,
    # so that you can use spin-split LLs
    return_value = np.zeros(len(eps))
    for eps_0 in LL_energies:
        #return_value += exp(-(eps - eps_0)**2 / (2 * sigma**2))
    
        ## broaden should return a gaussian with area = 1. However, each 
        ## gaussian accounts for an area 
        return_value += E_c * broaden(eps, eps_0, sigma)
    #return  [eps, prefactor * return_value]
    return  [eps, return_value]




###############################################################################
# C. Specific heat
###############################################################################
def specific_heat(reduced_DOS, T, mu=None, n_e=1e15, dT_frac=0.01):
    """ Numerically calculate specific heat of the 2DEG
    works by calculating total energy U at temperatures slightly above and
    below T and approximating C = dU/dT

    Example: calculate the specific heat for a flat density of states

    >>> import numpy as np
    >>> eps = np.linspace (0,500,1000)
    >>> dens = np.ones(len(eps))
    >>> C = k_b * specific_heat([eps, dens], 1)
    >>> print 'Numerical: %.5e'%C
    Numerical: 1.09548e-09
    >>> print "Analytical: %.5e"%(pi * m_star * k_b **2 / (3 * hbar **2))
    Analytical: 1.09548e-09
    """
    # generate high and low temperatures, which are +/- 5% from T
    dT = T * dT_frac
    T_h = T + 0.5 * dT
    T_l = T - 0.5 * dT

    [eps, dens] = reduced_DOS
    if mu is None:
        # these functions are yet to be moved into this file.
        mu_high = get_mu_at_T([eps, dens], T_h, n_e=n_e)
        mu_low = get_mu_at_T([eps, dens], T_l, n_e=n_e)
    else:
        mu_high = mu
        mu_low = mu

    dU = simps((fermi(eps, mu_high, T_h)-fermi(eps, mu_low, T_l))
                  * (eps-mu_low)* dens, x=eps)
                  
    #dU = np.trapz((fermi(eps, mu_high, T_h)-fermi(eps, mu_low, T_l))
    #              * (eps-mu_low)* DOS, x = eps)              
    # previously used (eps-mu_low) instead of (eps) in above. Need to think
    # about this a bit more.                 

    # commented factors would convert to J/(K m**2)
    return dU/dT * nu0 


###############################################################################
# D. Conductance and spectral diffusion
###############################################################################

def sigma_DC(B, tau_tr, v_f, q=q_e):
    """ Calculate sigma_DC as defined in  Zhang et al. PRB 80, 045310 (2009)

    >>> print '%.3e'%sigma_DC(1, 10e-12, v_f=v_fermi(3e15))
    1.828e-05
    """
    return (q **2 * nu0/k_b * v_f**2 * tau_tr / 
            (2 * (1 + omega_c(B)**2 * tau_tr**2)))


def sigma_nl(B, tau_tr, reduced_DOS, f_dist, v_f):
    """ Calculate sigma_nl as defined in  Zhang et al. PRB 80, 045310 (2009)
    This calculates the conductance in quasi-equilibrium if f_dist=fermi, but
    can also calculate non-equilibrium transport if some other f_dist is given.

    """
    [eps, dens] = reduced_DOS
    return simps(sigma_DC(B, tau_tr, v_f) * dens**2
                    * -1 * deriv(f_dist, eps), x=eps)


###############################################################################
# Z. When the file is run directly....
###############################################################################
# run doctests when this file is executed as a script
if __name__ == "__main__":
    import doctest
    doctest.testmod()
