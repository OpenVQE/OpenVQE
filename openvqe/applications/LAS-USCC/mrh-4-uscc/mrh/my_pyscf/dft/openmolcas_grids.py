import numpy

# Update 04/07/2022:
# OpenMolcas seems to have changed something about its angular grids
# in the last few months. The smallest possible angular grid
# in OpenMolcas distributes 5 points on each radial shell, as opposed
# to 6 in PySCF. I need to figure this out.

# quasi_ultrafine is nearly the equivalent of
#   grid input
#   nr=100
#   lmax=41
#   rquad=ta
#   nopr
#   noro
#   end of grid input
# in OpenMolcas version 21.06
# It is similar to "grid=ultrafine"

om_ta_alpha = [0.8, 0.9, # H, He
    1.8, 1.4, # Li, Be
        1.3, 1.1, 0.9, 0.9, 0.9, 0.9, # B - Ne
    1.4, 1.3, # Na, Mg
        1.3, 1.2, 1.1, 1.0, 1.0, 1.0, # Al - Ar
    1.5, 1.4, # K, Ca
            1.3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1, # Sc - Zn
        1.1, 1.0, 0.9, 0.9, 0.9, 0.9] # Ga - Kr
def om_treutler_ahlrichs(n, chg, *args, **kwargs):
    '''
    "Treutler-Ahlrichs" as implemented in OpenMolcas
    '''
    r = numpy.empty(n)
    dr = numpy.empty(n)
    alpha = om_ta_alpha[chg-1]
    step = 2.0 / (n+1) # = numpy.pi / (n+1)
    ln2 = alpha / numpy.log(2)
    for i in range(n):
        x = (i+1)*step - 1 # = numpy.cos((i+1)*step)
        r [i] = -ln2*(1+x)**.6 * numpy.log((1-x)/2)
        dr[i] = (step #* numpy.sin((i+1)*step) 
                * ln2*(1+x)**.6 *(-.6/(1+x)*numpy.log((1-x)/2)+1/(1-x)))
    return r[::-1], dr[::-1]

quasi_ultrafine = {'atom_grid': (99,590),
    'radi_method': om_treutler_ahlrichs,
    'prune': False,
    'radii_adjust': None}


