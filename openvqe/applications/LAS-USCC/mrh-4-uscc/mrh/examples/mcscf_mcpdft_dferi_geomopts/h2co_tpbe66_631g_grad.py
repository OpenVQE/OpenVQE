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
mol = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = False, verbose = lib.logger.INFO, output = 'h2co_tpbe66_631g_grad.log')
mf_conv = scf.RHF (mol).run ()
mc_conv = mcpdft.CASSCF (mf_conv, 'tPBE', 6, 6, grids_level=6)
mc_conv.conv_tol = 1e-10
mc_conv.kernel ()


mf = scf.RHF (mol).density_fit (auxbasis = df.aug_etb (mol)).run ()
mc_df = mcpdft.CASSCF (mf, 'tPBE', 6, 6, grids_level=6)
mc_df.conv_tol = 1e-10
mc_df.kernel ()

try:
    de_num = np.load ('h2co_tpbe66_631g_grad_num.npy')
    de_conv_num, de_df_num = list (de_num)
except OSError as e:
    de_conv_num = numeric_grad.Gradients (mc_conv).kernel ()
    de_df_num = numeric_grad.Gradients (mc_df).kernel ()
    np.save ('h2co_tpbe66_631g_grad_num.npy', np.stack ((de_conv_num, de_df_num), axis=0))
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

de_conv = mc_conv.nuc_grad_method ().kernel ()
de_df = mcpdft_grad.Gradients (mc_df).kernel ()

def printable_grad (arr):
    arr[np.abs (arr)<1e-10] = 0.0
    line_fmt = ' '.join (('{:13.9f}',)*3)
    return '\n'.join ((line_fmt.format (*row) for row in arr))

print ("Analytic gradient with conventional ERIs:\n", printable_grad (de_conv)[1:])
print ("Numeric gradient with conventional ERIs:\n", printable_grad (de_conv_num)[1:])
print ("Gradient error with conventional ERIs = {:.6e}".format (linalg.norm (de_conv - de_conv_num)))
print ("Analytic gradient with DF ERIs:\n", printable_grad (de_df)[1:])
print ("Numeric gradient with DF ERIs:\n", printable_grad (de_df_num)[1:])
print ("Gradient error with DF ERIs = {:.6e}".format (linalg.norm (de_df - de_df_num)))
print ("Gradient disagreements of state 0: Analytic = {:.6e} ; Numeric = {:.6e}".format (
    linalg.norm (de_df - de_conv), linalg.norm (de_df_num - de_conv_num)))


