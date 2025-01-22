from pyscf import gto, scf, mcscf
from pyscf.lib import logger
from pyscf.data.nist import BOHR
from pyscf.geomopt.geometric_solver import kernel as optimize
from pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from scipy import linalg
import numpy as np
import math

# PySCF has no native geometry optimization driver
# To do this, you'll need to install the geomeTRIC optimizer
# It is at https://github.com/leeping/geomeTRIC
# "pip install geometric" may also work

# Convenience functions to get the internal coordinates for human inspection
def bond_length (carts, i, j):
    return linalg.norm (carts[i] - carts[j])
def bond_angle (carts, i, j, k):
    rij = carts[i] - carts[j]
    rkj = carts[k] - carts[j]
    res = max (min (1.0, np.dot (rij, rkj) / linalg.norm (rij) / linalg.norm (rkj)), -1.0)
    return math.acos (res) * 180 / math.pi
def h2co_geom_analysis (carts):
    print ("rCO = {:.4f} Angstrom".format (bond_length (carts, 1, 0)))
    print ("rCH1 = {:.4f} Angstrom".format (bond_length (carts, 2, 0)))
    print ("rCH2 = {:.4f} Angstrom".format (bond_length (carts, 3, 0)))
    print ("tOCH1 = {:.2f} degrees".format (bond_angle (carts, 1, 0, 2)))
    print ("tOCH2 = {:.2f} degrees".format (bond_angle (carts, 1, 0, 3)))
    print ("tHCH = {:.2f} degrees".format (bond_angle (carts, 3, 0, 2)))

# Energy calculation at initial geometry
h2co_casscf66_631g_xyz = '''C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125'''
mol = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = False, verbose = logger.INFO, output = 'h2co_tpbe66_631g_opt.log')
mf = scf.RHF (mol).run ()
mc = mcpdft.CASSCF (mf, 'tPBE', 6, 6, grids_level=6)
mc.fcisolver = csf_solver (mol, smult = 1)
mc.kernel ()

# Geometry optimization (my_call is optional; it just prints the geometry in internal coordinates every iteration)
def my_call (env):
    carts = env['mol'].atom_coords () * BOHR
    h2co_geom_analysis (carts)
conv_params = {
    'convergence_energy': 1e-6,  # Eh
    'convergence_grms': 3e-5,    # Eh/Bohr
    'convergence_gmax': 4.5e-5,  # Eh/Bohr
    'convergence_drms': 1.2e-5,  # Angstrom
    'convergence_dmax': 1.8e-5,  # Angstrom
}
conv, mol_eq = optimize (mc, callback=my_call, **conv_params)

molcas_geom = np.asarray ([[0.549600, 0.000000, 0.000000],
[-0.692516, 0.000000,  0.000000],
[ 1.139765, 0.000000,  0.939946],
[ 1.139765, 0.000000, -0.939946]])
print ("tPBE(6,6)/6-31g optimized geometry of formaldehdye:")
h2co_geom_analysis (mol_eq.atom_coords () * BOHR)
print ("OpenMolcas's opinion using analytical gradient implementation (note OpenMolcas and PySCF have different quadrature grids):")
h2co_geom_analysis (molcas_geom)





