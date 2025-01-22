#from rhf import monkeypatch_setup
#monkeypatch_teardown = monkeypatch_setup ()
import math
import numpy as np
from scipy import linalg
from pyscf import scf, gto, lib, mcscf, df
from pyscf import mcpdft
from pyscf.df.grad import mcpdft as mcpdft_grad
from mrh.my_pyscf.grad import numeric as numeric_grad

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


h2co_casscf66_631g_xyz = '''C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125'''
mol = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = False, verbose = lib.logger.INFO, output = 'h2co_tpbe66_631g_opt.log')
mf_conv = scf.RHF (mol).run ()
mc_conv = mcpdft.CASSCF (mf_conv, 'tPBE', 6, 6, grids_level=6)
mc_conv.conv_tol = 1e-10
mc_conv.kernel ()

mf = scf.RHF (mol).density_fit (auxbasis = df.aug_etb (mol)).run ()
mc = mcpdft.CASSCF (mf, 'tPBE', 6, 6, grids_level=6)
mc.conv_tol = 1e-10
mc.kernel ()

def my_call (env):
    carts = env['mol'].atom_coords () * lib.param.BOHR
    h2co_geom_analysis (carts)
conv_params = {
    'convergence_energy': 1e-6,  # Eh
    'convergence_grms': 5.0e-5,  # Eh/Bohr
    'convergence_gmax': 7.5e-5,  # Eh/Bohr
    'convergence_drms': 1.0e-4,  # Angstrom
    'convergence_dmax': 1.5e-4,  # Angstrom
}

mc_opt_conv = mc_conv.nuc_grad_method ().as_scanner ().optimizer ()
mc_opt_conv.callback = my_call
mol_eq_conv = mc_opt_conv.kernel (params = conv_params)

mc_opt_df = mcpdft_grad.Gradients (mc).as_scanner ().optimizer ()
mc_opt_df.callback = my_call
mol_eq_df = mc_opt_df.kernel (params = conv_params)

print ("Conventional optimized geometry:")
h2co_geom_analysis (mol_eq_conv.atom_coords () * lib.param.BOHR)

print ("DF optimized geometry:")
h2co_geom_analysis (mol_eq_df.atom_coords () * lib.param.BOHR)

