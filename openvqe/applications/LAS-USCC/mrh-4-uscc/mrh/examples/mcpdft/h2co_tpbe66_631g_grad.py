from pyscf import gto, scf, mcscf
from pyscf.lib import logger
from pyscf.data.nist import BOHR
from pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from scipy import linalg
import numpy as np
import math

logger.TIMER_LEVEL = logger.INFO

# Energy calculation
h2co_casscf66_631g_xyz = '''C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125'''
mol = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = False, verbose = logger.INFO, output = 'h2co_tpbe66_631g_grad.log')
mf = scf.RHF (mol).run ()
mc = mcpdft.CASSCF (mf, 'tPBE', 6, 6, grids_level=9)
mc.fcisolver = csf_solver (mol, smult = 1)
mc.kernel ()

# Gradient calculation
mc_grad = mc.nuc_grad_method ()
dE = mc_grad.kernel ()
print ("tPBE(6,6)/6-31g gradient of formaldehyde at the CASSCF(6,6)/6-31g geometry:")
for ix, row in enumerate (dE):
    print ("{:1s} {:11.8f} {:11.8f} {:11.8f}".format (mol.atom_pure_symbol (ix), *row))
print ("OpenMolcas's opinions (note that there is no way to use the same grid in PySCF and OpenMolcas)")
print (("Analytical implementation:\n"
"C -0.02790613  0.00000000 -0.00000000\n"
"O  0.05293225 -0.00000000  0.00000000\n"
"H -0.01251306 -0.00000000 -0.01568837\n"
"H -0.01251306 -0.00000000  0.01568837\n"
"Numerical algorithm:\n"
"C -0.028019    0.000000   -0.000000\n"
"O  0.053037   -0.000000    0.000000\n"
"H -0.012509   -0.000000   -0.015737\n"
"H -0.012509   -0.000000    0.015737\n"))


