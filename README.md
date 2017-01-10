# non-interacting-2DEG
Numerically calculate quantities such as specific heat and conductance from a density of states

In recent years, there have been a number of proposed experiments that would use measurements of the entropy in the 5/2 FQH state to detect whether it is non-Abelian or not. Those proposals are laid out in the following papers:

http://dx.doi.org/10.1103/PhysRevB.79.115317 (Thermopower)

http://dx.doi.org/10.1103/PhysRevLett.102.176807 (dS/dN via Maxwell relations)

http://dx.doi.org/10.1103/10.1103/PhysRevLett.105.086801 (Adiabatic cooling)

http://dx.doi.org/10.1103/10.1103/10.1103/PhysRevB.85.195107 (Thermopower in Corbino)

In my lab work, I have focused on measurement of specific heat as an additional means to probe the entropy of the 2DEG. Preliminary results are at https://arxiv.org/abs/1605.02344. Any of these measurements must be validated, and are also of fundamental interest, away from the FQHE in the IQHE and SdH regimes. As part of my thesis, I explore the relative merits of these different "entropy probes", and this code gathers together the various calculations into a single framework. The general idea is to

1) Calculate a reasonable guess for the density of states (broadening and its field dependence, gaussian, Lorentzian or semi-elliptical)

2) Validate the DOS by using it to calculate electrical conductance and comparing that to experiment

3) Use the same DOS to calculate other quantities such as the specific heat (others to be added in this code later)

Other highly relevant references are:

http://dx.doi.org/10.1038/ncomms8298 (dS/dN at higher temperatures)

http://dx.doi.org/10.7566/JPSJ.82.053702 (Measurement of thermopower in Corbino)

http://dx.doi.org/10.1103/PhysRevB.87.075302 and references therein (Measurement of thermopower at 5/2)

http://dx.doi.org/10.1103/PhysRevB.80.045310 (Spectral diffusion)

