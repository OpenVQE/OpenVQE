from pyscf import gto, scf, mcscf
from pyscf.lib import logger
from pyscf.data.nist import BOHR
from pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from scipy import linalg
import numpy as np
import math

# Energy calculation
logger.TIMER_LEVEL = logger.INFO
h2co_casscf66_631g_xyz = '''C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125'''
mol = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = False, verbose = logger.INFO, output = 'h2co_sa2_tpbe66_631g_grad1.log')
mf = scf.RHF (mol).run ()
mc = mcpdft.CASSCF (mf, 'tPBE', 6, 6, grids_level=9)
mc.fcisolver = csf_solver (mol, smult = 1)
mc.state_average_([0.5,0.5])
mc.kernel ()

# mc.nuc_grad_method for MC-PDFT objects already points to a state-specific solver
# Just select which root!

# Gradient calculation
mc_grad = mc.nuc_grad_method ()
dE = mc_grad.kernel (state = 1)
print ("SA(2) tPBE(6,6)/6-31g second root gradient of formaldehyde at the CASSCF(6,6)/6-31g geometry:")
for ix, row in enumerate (dE):
    print ("{:1s} {:11.8f} {:11.8f} {:11.8f}".format (mol.atom_pure_symbol (ix), *row))
print ("OpenMolcas's opinions (note that there is no way to use the same grid in PySCF and OpenMolcas)")
print (("Analytical implementation:\n"
"C -0.17842556 -0.00000000 -0.00000000\n"
"O  0.18358344 -0.00000000  0.00000000\n"
"H -0.00257894 -0.00000000 -0.01065860\n"
"H -0.00257894  0.00000000  0.01065860\n"
"Numerical algorithm:\n"
"C -0.178517   -0.000000   -0.000000\n"
"O  0.183663    0.000000    0.000000\n"
"H -0.002573    0.000000   -0.010651\n"
"H -0.002573    0.000000    0.010651\n"))

