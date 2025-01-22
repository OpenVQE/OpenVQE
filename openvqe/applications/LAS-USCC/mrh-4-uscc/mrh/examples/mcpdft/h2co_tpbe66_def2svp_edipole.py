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
h2co_xyz = '''C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125'''
mol = gto.M (atom = h2co_xyz, basis = 'def2svp', symmetry = False, verbose = logger.INFO, output = 'h2co_tpbe66_def2svp_edipole.log')
mf = scf.RHF (mol).run ()
mc = mcpdft.CASSCF (mf, 'tPBE', 6, 6, grids_level=9)
mc.fcisolver = csf_solver (mol, smult = 1)
mc.kernel ()

# Electric Dipole calculation
dipole = mc.dip_moment(unit='Debye')
print ("MC-PDFT electric dipole moment Debye \n {:8.5f} {:8.5f} {:8.5f}".format (*dipole))
print ("Numerical MC-PDFT electric dipole moment from GAMESS [Debye] \n 2.09361 0.00000 0.00000 ")

