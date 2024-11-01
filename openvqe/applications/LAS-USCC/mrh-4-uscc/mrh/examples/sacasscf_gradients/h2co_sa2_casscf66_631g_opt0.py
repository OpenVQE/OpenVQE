from pyscf import gto, scf, mcscf
from pyscf.lib import logger
from pyscf.data.nist import BOHR
from pyscf.geomopt.geometric_solver import kernel as optimize
from mrh.my_pyscf.fci import csf_solver
#from mrh.my_pyscf.grad import sacasscf
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
def out_of_plane_angle (carts, i, j, k, l):
    eji = carts[j] - carts[i]
    eki = carts[k] - carts[i]
    eli = carts[l] - carts[i]
    eji /= linalg.norm (eji)
    eki /= linalg.norm (eki)
    eli /= linalg.norm (eli)
    return -math.asin (np.dot (eji, (np.cross (eki, eli) / math.sin (bond_angle (carts, j, i, k) * math.pi / 180)))) * 180 / math.pi
def h2co_geom_analysis (carts):
    print ("rCO = {:.4f} Angstrom".format (bond_length (carts, 1, 0)))
    print ("rCH1 = {:.4f} Angstrom".format (bond_length (carts, 2, 0)))
    print ("rCH2 = {:.4f} Angstrom".format (bond_length (carts, 3, 0)))
    print ("tOCH1 = {:.2f} degrees".format (bond_angle (carts, 1, 0, 2)))
    print ("tOCH2 = {:.2f} degrees".format (bond_angle (carts, 1, 0, 3)))
    print ("tHCH = {:.2f} degrees".format (bond_angle (carts, 3, 0, 2)))
    print ("eta = {:.2f} degrees".format (out_of_plane_angle (carts, 0, 2, 3, 1)))

# Energy calculation at initial geometry
h2co_casscf66_631g_xyz = '''C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125'''
mol = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = False, verbose = logger.INFO, output = 'h2co_sa2_casscf66_631g_opt0.log')
mf = scf.RHF (mol).run ()
mc = mcscf.CASSCF (mf, 6, 6)
mc.fcisolver = csf_solver (mol, smult = 1)
mc.state_average_([0.5,0.5])
mc.kernel ()

# Geometry optimization (my_call is optional; it just prints the geometry in internal coordinates every iteration)
print ("Initial geometry: ")
h2co_geom_analysis (mol.atom_coords () * BOHR)
print ("Initial energy: {:.8e}".format (mc.e_states[0]))
def my_call (env):
    carts = env['mol'].atom_coords () * BOHR
    h2co_geom_analysis (carts)
conv_params = {
    'convergence_energy': 1e-6,  # Eh
    'convergence_grms': 5.0e-5,  # Eh/Bohr
    'convergence_gmax': 7.5e-5,  # Eh/Bohr
    'convergence_drms': 1.0e-4,  # Angstrom
    'convergence_dmax': 1.5e-4,  # Angstrom
}
# PySCF update: PySCF can now do this natively
#mc_opt = sacasscf.Gradients (mc).as_scanner (state = 0).optimizer ()
mc_opt = mc.nuc_grad_method ().as_scanner (state = 0).optimizer ()
mc_opt.callback = my_call
mol_eq = mc_opt.kernel (params = conv_params)
molcas_geom = np.asarray ([[ 0.54965420,0.00000000, 0.00000000],
[-0.71017341,0.00000000, 0.00000000],
[ 1.11163660,0.00000000, 0.92144690],
[ 1.11163660,0.00000000,-0.92144690]])
print ("SA(2) CASSCF(6,6)/6-31g optimized geometry of first root of formaldehdye:")
h2co_geom_analysis (mol_eq.atom_coords () * BOHR)
print ("OpenMolcas's opinion using analytical gradient implementation:")
h2co_geom_analysis (molcas_geom)





