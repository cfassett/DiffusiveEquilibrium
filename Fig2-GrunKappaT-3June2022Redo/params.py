from math import *


gravity=1.62        # m/s2
strength=1.0e4     # Pa
targdensity=1500.0  #kg/m3 (rho)
impdensity=1500.0   #kg/m3 (delta)
velocity=18000.0    # m/s    COULD choose a distribution?   
alpha=45.0          # impact angle degrees
effvelocity=velocity*sin(radians(alpha))
nu=(1.0/3.0)           # ~1/3 to 0.4
mu=0.4593             # ~0.4 to 0.55    Varying mu makes a big difference in scaling, 0.41 from Williams et al. would predict lower fluxes / longer equilibrium times and a discontinuity with Neukum
K1=0.132
K2=0.26  
Kr=1.1*1.3  #Kr and KrRim  
densityratio=(targdensity/impdensity)

surfacearea = 1.0 #km2

visibilitythreshold=0.04
