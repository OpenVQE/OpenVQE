from pyscf import gto, scf, mcscf
from pyscf.lib import logger
from pyscf.data.nist import BOHR
from mrh.my_pyscf.fci import csf_solver
#from mrh.my_pyscf.grad import sacasscf
from scipy import linalg
import numpy as np
import math

# Energy calculation
h2co_casscf66_631g_xyz = '''C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125'''
mol = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = False, verbose = logger.INFO, output = 'h2co_sa2_casscf66_631g_grad0.log')
mf = scf.RHF (mol).run ()
mc = mcscf.CASSCF (mf, 6, 6)
mc.fcisolver = csf_solver (mol, smult = 1)
mc.state_average_([0.5,0.5])
mc.kernel ()

# Gradient calculation (PySCF update: PySCF can now do this natively)
#mc_grad = sacasscf.Gradients (mc)
mc_grad = mc.nuc_grad_method ()
dE = mc_grad.kernel (state = 0)
print ("SA(2) CASSCF(6,6)/6-31g first root gradient of formaldehyde at the CASSCF(6,6)/6-31g geometry:")
for ix, row in enumerate (dE):
    print ("{:1s} {:11.8f} {:11.8f} {:11.8f}".format (mol.atom_pure_symbol (ix), *row))
print ("OpenMolcas's opinions")
print (("Analytical implementation:\n"
"C -0.07115711 -0.00000000 -0.00000000\n"
"O  0.07528782  0.00000000  0.00000000\n"
"H -0.00206535  0.00000000 -0.00004778\n"
"H -0.00206535 -0.00000000  0.00004778\n"
"Numerical algorithm:\n"
"C -0.071257 -0.000000  0.000000\n"
"O  0.075383  0.000000  0.000000\n"
"H -0.002063  0.000000 -0.000061\n"
"H -0.002063  0.000000  0.000061\n"))


