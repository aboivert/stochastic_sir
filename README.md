# Implementation of SIR stochastic model

Use of the stochastic simulator GEMFSim to represent the propagation of an epidemic through a graph using the SIR model.

Several simulation files are available:

-normal.py : no restrictive measures

-lockdown.py : implementation of a lockdown lifted as soon as the percentage of infected people is lower than seuildeconf

-stop&go.py : stop&go strategy : alternating lockdowns

-curfew.py : implementation of a curfew until the epidemic is over

-curfew_light_lockdown.py : implementation of a curfew until the end of the epidemic and a light lockdown simultaneously, which is lifted as soon as the percentage of infected people is lower than seuildeconf

If you want to use our version, please download myGEMF.py and the strategy you want to use. 

Attention, all files record the population curves, the R0 and the values used to display the curves. It is possible to deactivate this.

Original code and additional resources available at this address : https://www.ece.k-state.edu/netse/software/
